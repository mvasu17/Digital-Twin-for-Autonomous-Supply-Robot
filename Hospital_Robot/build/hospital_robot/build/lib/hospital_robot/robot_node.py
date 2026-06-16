#!/usr/bin/env python3
"""
=============================================================
HOSPITAL ROBOT NODE — A* Pathfinding + ML Stock Prediction
=============================================================
WHAT THIS NODE DOES:
1. Subscribes to /odom to know where the robot is
2. Every 10 seconds, queries ML model for each room's stock
3. If refill needed → compute A* path → move robot room by room
4. Posts position updates to the Flask dashboard

WHY A* ALGORITHM?
- A* is the gold standard for robot pathfinding
- It uses a heuristic (straight-line distance) to find shortest path
- Unlike Dijkstra's, it's faster because it prioritizes nodes closer to goal
- Formula: f(n) = g(n) + h(n)
  * g(n) = cost from start to current node
  * h(n) = estimated cost from current node to goal (heuristic)
  * f(n) = total estimated cost

GRID LAYOUT (matches Gazebo world):
- Grid cell size: 0.5m × 0.5m
- World: -15 to +15 (X), -8 to +8 (Y)
- Grid size: 60 × 32 cells
- Walls are marked as obstacles in the grid
=============================================================
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import math
import heapq
import time
import threading
import requests
import joblib
import os
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────
# A* PATHFINDING IMPLEMENTATION
# ─────────────────────────────────────────────────────────

class AStarGrid:
    """
    A* pathfinding on a 2D grid.
    
    The grid represents our hospital floor:
    - 0 = free space (corridors, rooms)
    - 1 = obstacle (walls, furniture)
    
    Each cell = 0.5m in the real world.
    """

    def __init__(self, width, height, cell_size=0.5):
        self.width     = width       # Grid columns
        self.height    = height      # Grid rows
        self.cell_size = cell_size   # Meters per grid cell
        
        # Origin: world (0,0) maps to grid (width//2, height//2)
        self.origin_x  = width  // 2
        self.origin_y  = height // 2
        
        # Initialize empty grid (all free)
        self.grid = [[0] * height for _ in range(width)]
        
        # Build the hospital obstacle map
        self._build_hospital_map()

    def _build_hospital_map(self):
        """
        Add walls as obstacles.
        Wall positions match the Gazebo world file.
        
        Room layout:
          Pharmacy: x ∈ [-14, -2],  y ∈ [-6, 6]
          ICU:      x ∈ [-6,   6],  y ∈ [-6, 6]
          General:  x ∈ [2,   14],  y ∈ [-6, 6]
        
        Doorways (openings in walls) at y ∈ [-2, 2] between rooms.
        """
        # Add outer boundary walls
        for gx in range(self.width):
            for gy in range(self.height):
                wx, wy = self.grid_to_world(gx, gy)
                
                # Outer boundary
                if abs(wx) > 14.5 or abs(wy) > 6.8:
                    self.grid[gx][gy] = 1
                    continue
                
                # ── Pharmacy east wall (x ≈ -2.15), doorway at y ∈ [-2, 2] ──
                if -2.5 < wx < -1.8 and abs(wy) > 2.2:
                    self.grid[gx][gy] = 1

                # ── ICU west wall (x ≈ -5.7), doorway at y ∈ [-2, 2] ──
                if -6.0 < wx < -5.4 and abs(wy) > 2.2:
                    self.grid[gx][gy] = 1

                # ── ICU east wall (x ≈ 5.7), doorway at y ∈ [-2, 2] ──
                if 5.4 < wx < 6.0 and abs(wy) > 2.2:
                    self.grid[gx][gy] = 1

                # ── General ward west wall (x ≈ 2.15), doorway at y ∈ [-2, 2] ──
                if 1.8 < wx < 2.5 and abs(wy) > 2.2:
                    self.grid[gx][gy] = 1

                # ── North/South room walls ──
                # Pharmacy N/S walls
                if -14.0 < wx < -2.0 and abs(wy) > 5.6:
                    self.grid[gx][gy] = 1
                # ICU N/S walls
                if -5.85 < wx < 5.85 and abs(wy) > 5.6:
                    self.grid[gx][gy] = 1
                # General Ward N/S walls
                if 2.0 < wx < 14.0 and abs(wy) > 5.6:
                    self.grid[gx][gy] = 1

                # ── ICU beds (obstacles) ──
                for bx in [-3.0, 0.0, 3.0]:
                    if abs(wx - bx) < 1.1 and 1.5 < wy < 2.8:
                        self.grid[gx][gy] = 1

                # ── GW beds (obstacles) ──
                for bx in [5.0, 8.0, 11.0]:
                    if abs(wx - bx) < 1.1 and 1.5 < wy < 2.8:
                        self.grid[gx][gy] = 1

    def world_to_grid(self, wx, wy):
        """Convert world coordinates (meters) to grid indices."""
        gx = int(round(wx / self.cell_size)) + self.origin_x
        gy = int(round(wy / self.cell_size)) + self.origin_y
        gx = max(0, min(self.width - 1,  gx))
        gy = max(0, min(self.height - 1, gy))
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid indices to world coordinates (meters)."""
        wx = (gx - self.origin_x) * self.cell_size
        wy = (gy - self.origin_y) * self.cell_size
        return wx, wy

    def is_free(self, gx, gy):
        """Check if a cell is obstacle-free."""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.grid[gx][gy] == 0
        return False

    def heuristic(self, a, b):
        """
        Euclidean distance heuristic.
        WHY: Euclidean is admissible (never overestimates) for diagonal movement,
        which gives us optimal paths.
        """
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def get_neighbors(self, gx, gy):
        """
        8-directional movement (cardinal + diagonal).
        Diagonal moves cost √2, cardinal cost 1.0.
        """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if self.is_free(nx, ny):
                    cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    neighbors.append((nx, ny, cost))
        return neighbors

    def find_path(self, start_wx, start_wy, goal_wx, goal_wy):
        """
        A* search from start to goal.
        
        Returns: list of (world_x, world_y) waypoints, or [] if no path.
        
        Algorithm steps:
        1. Convert world coords to grid cells
        2. Open set = priority queue sorted by f = g + h
        3. Pop lowest-f node, check if goal reached
        4. Expand neighbors, update costs if better path found
        5. Reconstruct path by tracing parents back to start
        """
        start = self.world_to_grid(start_wx, start_wy)
        goal  = self.world_to_grid(goal_wx,  goal_wy)

        if not self.is_free(*start):
            # Find nearest free cell to start
            for r in range(1, 5):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        ns = (start[0]+dx, start[1]+dy)
                        if self.is_free(*ns):
                            start = ns
                            break

        if not self.is_free(*goal):
            # Find nearest free cell to goal
            for r in range(1, 8):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        ng = (goal[0]+dx, goal[1]+dy)
                        if self.is_free(*ng):
                            goal = ng
                            break

        # Priority queue: (f_score, node)
        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from  = {}
        g_score    = {start: 0.0}
        f_score    = {start: self.heuristic(start, goal)}
        closed_set = set()

        iterations = 0
        max_iter   = 10000  # Safety limit

        while open_set and iterations < max_iter:
            iterations += 1
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            # GOAL REACHED → reconstruct path
            if current == goal or self.heuristic(current, goal) < 1.5:
                path = []
                while current in came_from:
                    wx, wy = self.grid_to_world(*current)
                    path.append((wx, wy))
                    current = came_from[current]
                path.reverse()
                # Convert to world coords and smooth
                return self._smooth_path(path)

            # Expand neighbors
            for nx, ny, move_cost in self.get_neighbors(*current):
                if (nx, ny) in closed_set:
                    continue
                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)]   = tentative_g
                    f = tentative_g + self.heuristic((nx, ny), goal)
                    f_score[(nx, ny)]   = f
                    heapq.heappush(open_set, (f, (nx, ny)))

        return []  # No path found

    def _smooth_path(self, path):
        """
        Remove redundant waypoints to create a smoother path.
        Keeps only points where direction changes significantly.
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            # Check angle change
            dx1 = path[i][0]   - path[i-1][0]
            dy1 = path[i][1]   - path[i-1][1]
            dx2 = path[i+1][0] - path[i][0]
            dy2 = path[i+1][1] - path[i][1]
            
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            
            if abs(angle1 - angle2) > 0.15:  # ~8 degrees
                smoothed.append(path[i])
        
        smoothed.append(path[-1])
        return smoothed


# ─────────────────────────────────────────────────────────
# MAIN ROS2 ROBOT NODE
# ─────────────────────────────────────────────────────────

class HospitalRobotNode(Node):
    """
    Main robot node that:
    1. Maintains current position from odometry
    2. Periodically checks stock levels using ML model
    3. Sends navigation goals using A* planned paths
    4. Updates the dashboard with its position/status
    """

    # Room center positions in the Gazebo world
    ROOM_POSITIONS = {
        "pharmacy":     (-8.0,  0.0),
        "icu":          ( 0.0,  0.0),
        "general_ward": ( 8.0,  0.0),
    }

    # Room types for ML model
    ROOM_TYPES = {
        "pharmacy":     2,
        "icu":          1,
        "general_ward": 0,
    }

    def __init__(self):
        super().__init__('hospital_robot_node')

        self.get_logger().info("="*50)
        self.get_logger().info(" Hospital Robot Node Starting...")
        self.get_logger().info("="*50)

        # ── Publishers ──
        # cmd_vel sends velocity commands to the robot
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # ── Subscribers ──
        # Odometry gives us the robot's current position
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )

        # ── Robot state ──
        self.current_x     = 0.0
        self.current_y     = 0.0
        self.current_yaw   = 0.0
        self.is_moving     = False
        self.current_task  = None
        self.battery       = 100.0
        self.status        = "idle"

        # ── A* grid ──
        # Grid: 60×32 cells, 0.5m each = 30m × 16m world
        self.astar_grid = AStarGrid(width=60, height=32, cell_size=0.5)
        self.get_logger().info("[✓] A* grid initialized (60×32, 0.5m cells)")

        # ── ML model ──
        self.model    = None
        self.metadata = {}
        self._load_model()

        # ── Dashboard URL ──
        self.dashboard_url = "http://localhost:5000"

        # ── Stock check timer (every 10 seconds) ──
        self.stock_timer = self.create_timer(10.0, self.check_stocks_callback)

        # ── Position update timer (every 0.5 seconds) ──
        self.pos_timer = self.create_timer(0.5, self.update_dashboard)

        # ── Navigation thread ──
        self.nav_thread = None
        self.nav_lock   = threading.Lock()

        self.get_logger().info("[✓] Robot node ready. Waiting for stock checks...")

    def _load_model(self):
        """Load the trained ML model."""
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', '..', '..', '..', 'ml_model', 'stock_model.pkl'
        )
        model_path = os.path.normpath(model_path)

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.get_logger().info(f"[✓] ML model loaded from {model_path}")
        else:
            self.get_logger().warn(f"[!] Model not found at {model_path}")
            self.get_logger().warn("    Run: python3 ml_model/train_model.py")

    def odom_callback(self, msg):
        """
        Update robot position from odometry message.
        Odometry gives us x, y, and orientation quaternion.
        We convert quaternion → yaw angle for heading.
        """
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Convert quaternion to yaw (rotation around Z axis)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Slowly drain battery (simulation)
        if self.is_moving:
            self.battery = max(0, self.battery - 0.005)

    def predict_refill(self, room_key, current_stock):
        """
        Use ML model to predict if refill is needed.
        Falls back to threshold rule if model unavailable.
        """
        if self.model is None:
            return current_stock < 30, 0.9 if current_stock < 30 else 0.1

        now     = datetime.now()
        hour    = now.hour
        dow     = now.weekday() + 1
        shift   = 0 if 6 <= hour < 14 else (1 if 14 <= hour < 22 else 2)
        room_t  = self.ROOM_TYPES.get(room_key, 0)

        import random
        patient_count   = random.randint(8, 25)
        emergency_cases = random.randint(0, 3) if room_t == 1 else 0
        prev_usage      = max(1.0, (100 - current_stock) / 10.0)

        features = [[hour, patient_count, current_stock, shift,
                     emergency_cases, round(prev_usage, 1), dow, room_t]]

        try:
            pred = self.model.predict(features)[0]
            prob = self.model.predict_proba(features)[0][1]
            return bool(pred), round(float(prob), 3)
        except Exception as e:
            self.get_logger().warn(f"Prediction error: {e}")
            return current_stock < 30, 0.9

    def check_stocks_callback(self):
        """
        Timer callback: check all room stocks and dispatch robot if needed.
        Called every 10 seconds.
        
        WHY not continuous? We don't want the robot running around
        constantly. 10 seconds is fast enough for demo purposes
        (in real deployment this might be every few minutes).
        """
        if self.is_moving:
            return  # Don't interrupt active navigation

        # Ask dashboard for current stock levels
        try:
            response = requests.get(f"{self.dashboard_url}/api/state", timeout=2)
            state    = response.json()
            stocks   = state.get("stocks", {})
        except Exception:
            # Dashboard not running — use simulated values
            stocks = {
                "pharmacy":     {"current_stock": 85},
                "icu":          {"current_stock": 20},
                "general_ward": {"current_stock": 45},
            }

        # Check each room using ML model
        tasks_needed = []
        for room_key, stock_info in stocks.items():
            current_stock = stock_info.get("current_stock", 50)
            refill_needed, confidence = self.predict_refill(room_key, current_stock)

            self.get_logger().info(
                f"  [{room_key}] Stock={current_stock:.1f}  "
                f"Refill={'YES' if refill_needed else 'no'}  "
                f"Confidence={confidence:.0%}"
            )

            if refill_needed and confidence > 0.5:
                tasks_needed.append((room_key, current_stock, confidence))

        # Sort by urgency: lowest stock first
        tasks_needed.sort(key=lambda x: x[1])

        if tasks_needed:
            top_task = tasks_needed[0]
            self.get_logger().info(
                f"[!] Most urgent: {top_task[0]} (stock={top_task[1]:.1f})"
            )
            # Launch navigation in background thread
            self.nav_thread = threading.Thread(
                target=self.navigate_to_room,
                args=(top_task[0],),
                daemon=True
            )
            self.nav_thread.start()
        else:
            self.get_logger().info("[✓] All stocks OK")

    def navigate_to_room(self, room_key):
        """
        Navigate robot from current position to target room using A*.
        
        Steps:
        1. Plan A* path from current pos to room center
        2. Follow path waypoint by waypoint
        3. At each waypoint: rotate to face it, then drive forward
        4. When at destination: simulate refilling
        """
        with self.nav_lock:
            goal_x, goal_y = self.ROOM_POSITIONS[room_key]

            self.get_logger().info(
                f"[→] Navigating to {room_key} at ({goal_x:.1f}, {goal_y:.1f})"
            )

            # ── STEP 1: A* PATH PLANNING ──
            path = self.astar_grid.find_path(
                self.current_x, self.current_y,
                goal_x, goal_y
            )

            if not path:
                self.get_logger().warn(f"[!] No path found to {room_key}! Using direct line.")
                path = [(goal_x, goal_y)]

            self.get_logger().info(f"[A*] Path found: {len(path)} waypoints")

            # Publish path to dashboard
            self.status       = "moving"
            self.current_task = room_key
            self.is_moving    = True

            # ── STEP 2: FOLLOW PATH ──
            for i, (wp_x, wp_y) in enumerate(path):
                self.get_logger().info(
                    f"    Waypoint {i+1}/{len(path)}: ({wp_x:.2f}, {wp_y:.2f})"
                )
                
                success = self.move_to_waypoint(wp_x, wp_y)
                if not success:
                    self.get_logger().warn(f"    Could not reach waypoint {i+1}")
                    break

                time.sleep(0.1)  # Small pause between waypoints

            # ── STEP 3: REFILL ──
            self.get_logger().info(f"[✓] Arrived at {room_key} — refilling stock...")
            self.status = "refilling"
            self.stop_robot()

            # Simulate refilling time
            time.sleep(3.0)

            # Notify dashboard that refill is done
            try:
                requests.post(
                    f"{self.dashboard_url}/api/robot_update",
                    json={
                        "x": self.current_x,
                        "y": self.current_y,
                        "status": "refilled",
                        "task": room_key,
                        "battery": self.battery
                    },
                    timeout=2
                )
            except Exception:
                pass

            self.get_logger().info(f"[✓] {room_key} refill complete!")

            # Return to center (ICU)
            self.status       = "moving"
            self.current_task = "returning"
            return_path = self.astar_grid.find_path(
                self.current_x, self.current_y, 0.0, 0.0
            )
            for wp_x, wp_y in (return_path or [(0.0, 0.0)]):
                self.move_to_waypoint(wp_x, wp_y)

            self.stop_robot()
            self.status       = "idle"
            self.current_task = None
            self.is_moving    = False

    def move_to_waypoint(self, target_x, target_y, tolerance=0.4):
        """
        Move robot to a specific (x, y) position.
        
        Strategy:
        1. Rotate in place to face the target
        2. Drive forward until within tolerance
        
        Returns True if reached, False if timeout.
        """
        max_time = 30.0  # seconds
        start_t  = time.time()

        while True:
            if time.time() - start_t > max_time:
                return False

            dx = target_x - self.current_x
            dy = target_y - self.current_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < tolerance:
                return True  # Reached waypoint

            # ── Angle to target ──
            target_yaw = math.atan2(dy, dx)
            yaw_error  = target_yaw - self.current_yaw

            # Normalize to [-π, π]
            while yaw_error >  math.pi: yaw_error -= 2 * math.pi
            while yaw_error < -math.pi: yaw_error += 2 * math.pi

            cmd = Twist()

            # ── Rotate if not facing target ──
            if abs(yaw_error) > 0.15:
                cmd.angular.z = 0.8 * yaw_error
                cmd.angular.z = max(-1.2, min(1.2, cmd.angular.z))
                cmd.linear.x  = 0.0
            else:
                # ── Drive forward ──
                cmd.linear.x  = min(0.5, 0.4 * dist)
                cmd.angular.z = 0.5 * yaw_error  # Gentle correction while moving

            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)

    def stop_robot(self):
        """Send zero velocity to stop the robot."""
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def update_dashboard(self):
        """
        Post current robot position to Flask dashboard.
        Called every 0.5 seconds by timer.
        """
        try:
            requests.post(
                f"{self.dashboard_url}/api/robot_update",
                json={
                    "x":       round(self.current_x, 3),
                    "y":       round(self.current_y, 3),
                    "status":  self.status,
                    "task":    self.current_task,
                    "battery": round(self.battery, 1),
                    "path":    []
                },
                timeout=1
            )
        except Exception:
            pass  # Dashboard might not be running


def main(args=None):
    rclpy.init(args=args)
    node = HospitalRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down robot node...")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
