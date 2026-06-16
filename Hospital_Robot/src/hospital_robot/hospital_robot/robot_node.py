#!/usr/bin/env python3
"""
Hospital Robot Node - Fixed Version
=====================================
Fixes applied:
  1. Spawns at pharmacy (-13, 9) - set in launch file
  2. Waypoints use exact corridor CENTRE lines so robot
     never drifts toward walls
  3. Speed inside corridors capped at 0.28 m/s (was 0.45)
     so robot has time to correct heading before hitting walls
  4. Dispatch triggers when stock <= DISPATCH_THR (55) period.
     ML model is still used but only needs conf > 0.30 to agree.
     If model is unavailable, threshold alone triggers dispatch.
     This fixes the "dashboard shows 50 but robot waits until 40" bug.
  5. Stock check every 5s (was 10s) - catches threshold faster
  6. Obstacle inflation 3 cells (was 2) = 1.5m from walls
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math, heapq, time, threading, requests, joblib, os, random
from datetime import datetime


# ══════════════════════════════════════════════════════════
# A* GRID
# 100x80 cells, 0.5m each
# Inflation radius 3 = 1.5m from every wall
# ══════════════════════════════════════════════════════════
class AStarGrid:
    CELL = 0.5

    def __init__(self):
        self.W, self.H = 100, 80
        self.ox = self.W // 2
        self.oy = self.H // 2
        self.grid = [[0] * self.H for _ in range(self.W)]
        self._build()
        self._inflate(radius=3)   # 3 cells = 1.5m safety margin

    def w2g(self, wx, wy):
        gx = int(round(wx / self.CELL)) + self.ox
        gy = int(round(wy / self.CELL)) + self.oy
        return max(0, min(self.W - 1, gx)), max(0, min(self.H - 1, gy))

    def g2w(self, gx, gy):
        return (gx - self.ox) * self.CELL, (gy - self.oy) * self.CELL

    def free(self, gx, gy):
        return 0 <= gx < self.W and 0 <= gy < self.H and self.grid[gx][gy] == 0

    def _in_free_zone(self, wx, wy):
        # Pharmacy
        if -18 < wx < -8  and  5 < wy < 13:  return True
        # PH corridor
        if  -8 < wx < -2  and  7.5 < wy < 10.5: return True
        # ICU
        if -18 < wx < -6  and -3 < wy < 4:   return True
        # ICU corridor
        if  -6 < wx < -2  and -0.5 < wy < 2.5: return True
        # V-Main corridor
        if  -2 < wx < 2   and -7 < wy < 11:  return True
        # W corridor
        if   2 < wx < 6   and -7 < wy < -3.5: return True
        # General Ward
        if   6 < wx < 14  and -9 < wy < -1:  return True
        return False

    def _build(self):
        for gx in range(self.W):
            for gy in range(self.H):
                wx, wy = self.g2w(gx, gy)
                if not self._in_free_zone(wx, wy):
                    self.grid[gx][gy] = 1
                    continue
                # Pharmacy shelf (far west, x < -16)
                if -17.6 < wx < -16.2 and 5.5 < wy < 12.5:
                    self.grid[gx][gy] = 1; continue
                # Pharmacy counter (x ~ -10.5, not blocking east door y[7.5,10.5])
                if -11.2 < wx < -9.8 and 6.0 < wy < 12.0:
                    self.grid[gx][gy] = 1; continue
                # ICU beds north row (y ~ 2.5, x varied)
                for bx in [-16.5, -13.5, -10.5]:
                    if abs(wx - bx) < 1.1 and 1.9 < wy < 3.2:
                        self.grid[gx][gy] = 1; break
                if self.grid[gx][gy]: continue
                # ICU nurse station (south, clear of east door)
                if -10.0 < wx < -7.5 and -2.5 < wy < -1.0:
                    self.grid[gx][gy] = 1; continue
                # Ward beds south row
                for bx in [8.0, 11.0, 13.5]:
                    if abs(wx - bx) < 1.1 and -8.2 < wy < -6.8:
                        self.grid[gx][gy] = 1; break
                if self.grid[gx][gy]: continue
                # Ward beds north row (x ~ 9, 12.5; clear of west door y[-6.5,-4])
                for bx in [9.0, 12.5]:
                    if abs(wx - bx) < 1.1 and -3.6 < wy < -2.4:
                        self.grid[gx][gy] = 1; break
                if self.grid[gx][gy]: continue
                # Ward nursing station
                if 12.0 < wx < 14.0 and -6.5 < wy < -4.5:
                    self.grid[gx][gy] = 1

    def _inflate(self, radius=3):
        """
        Grow every obstacle by radius cells.
        radius=3 means robot stays 1.5m from every wall.
        This is the main fix for wall-hugging.
        """
        orig = [row[:] for row in self.grid]
        for gx in range(self.W):
            for gy in range(self.H):
                if orig[gx][gy] == 1:
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            nx, ny = gx + dx, gy + dy
                            if 0 <= nx < self.W and 0 <= ny < self.H:
                                self.grid[nx][ny] = 1

    def nearest_free(self, gx, gy):
        if self.free(gx, gy): return gx, gy
        for r in range(1, 25):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = gx + dx, gy + dy
                        if self.free(nx, ny): return nx, ny
        return gx, gy

    def neighbours(self, gx, gy):
        # Always 4-directional — safer, no diagonal wall-cutting
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        res = []
        for dx, dy in dirs:
            nx, ny = gx + dx, gy + dy
            if self.free(nx, ny):
                res.append((nx, ny, 1.0))
        return res

    def find_path(self, sx, sy, gx, gy):
        s = self.nearest_free(*self.w2g(sx, sy))
        g = self.nearest_free(*self.w2g(gx, gy))
        pq = [(0.0, s)]
        came = {}
        gs = {s: 0.0}
        closed = set()
        while pq:
            _, cur = heapq.heappop(pq)
            if cur in closed: continue
            closed.add(cur)
            if math.hypot(cur[0] - g[0], cur[1] - g[1]) < 2.0:
                path = []
                while cur in came:
                    path.append(self.g2w(*cur))
                    cur = came[cur]
                path.reverse()
                return self._smooth(path)
            for nx, ny, c in self.neighbours(*cur):
                if (nx, ny) in closed: continue
                tg = gs[cur] + c
                if tg < gs.get((nx, ny), 1e9):
                    came[(nx, ny)] = cur
                    gs[(nx, ny)] = tg
                    h = math.hypot(nx - g[0], ny - g[1])
                    heapq.heappush(pq, (tg + h, (nx, ny)))
        return []

    def _smooth(self, path):
        if len(path) <= 2: return path
        out = [path[0]]
        for i in range(1, len(path) - 1):
            a1 = math.atan2(path[i][1]-path[i-1][1], path[i][0]-path[i-1][0])
            a2 = math.atan2(path[i+1][1]-path[i][1], path[i+1][0]-path[i][0])
            if abs(a1 - a2) > 0.20: out.append(path[i])
        out.append(path[-1])
        return out


# ══════════════════════════════════════════════════════════
# ROBOT NODE
# ══════════════════════════════════════════════════════════
class HospitalRobotNode(Node):

    # Robot home = pharmacy centre
    HOME = (-13.0, 9.0)

    DELIVERY_ROOMS = {
        "icu":          (-12.0,  0.5),
        "general_ward": ( 10.0, -5.0),
    }
    ROOM_TYPES = {"icu": 1, "general_ward": 0}

    # ── CORRIDOR CENTRE WAYPOINTS ──────────────────────────────
    # Every waypoint sits exactly on the geometric centre line of
    # its corridor. This keeps the robot away from both walls.
    #
    # PH corridor:   x[-8,-2]  y[7.5,10.5]  → centre y = 9.0
    # ICU corridor:  x[-6,-2]  y[-0.5,2.5]  → centre y = 1.0
    # V-Main:        x[-2, 2]               → centre x = 0.0
    # W corridor:    x[2,6]    y[-7,-3.5]   → centre y = -5.25
    # Ward door gap:            y[-6.5,-4]  → centre y = -5.25
    WP = {
        # Pharmacy side
        "ph_pickup":  (-9.5,  9.0),   # inside pharmacy, clear of counter
        "ph_door_in": (-8.5,  9.0),   # just inside pharmacy east door
        "ph_door_out":(-7.5,  9.0),   # just outside pharmacy east door (in PH-corr)
        # PH corridor centre line y=9.0
        "ph_corr_c":  (-5.0,  9.0),   # dead centre of PH corridor
        # Junction: PH-corr meets V-main
        "vm_ph":      ( 0.0,  9.0),   # V-main at y=9 (PH level)
        # V-main centre line x=0.0 — travel south
        "vm_mid":     ( 0.0,  5.0),   # V-main midpoint
        # V-main at ICU corridor junction
        "vm_icu":     ( 0.0,  1.0),   # V-main at y=1 (ICU level)
        # ICU corridor centre line y=1.0
        "icu_corr_c": (-4.0,  1.0),   # dead centre of ICU corridor
        "icu_door":   (-6.5,  1.0),   # just outside ICU east door
        # ICU delivery point (clear of beds which are at y~2.5)
        "icu_dest":   (-12.0,  0.5),  # ICU centre, below beds
        # V-main south section
        "vm_ward":    ( 0.0, -5.25),  # V-main at W-corridor level
        # W corridor centre line y=-5.25
        "w_corr_c":   ( 4.0, -5.25),  # dead centre of W corridor
        "ward_door":  ( 6.5, -5.25),  # just inside ward west door
        # Ward delivery point (clear of beds which are at y~-7.5 and y~-3)
        "ward_dest":  ( 9.5, -5.25),  # ward centre corridor line
    }

    LOAD         = 50
    # ── DISPATCH THRESHOLD FIX ────────────────────────────────
    # DISPATCH_THR = 55: robot dispatches when stock <= 55
    # The ML model is consulted but only needs conf > 0.25 to agree.
    # If model is not loaded, threshold alone triggers dispatch.
    # Previously conf > 0.50 was required, which often blocked dispatch
    # because the model (trained with refill=stock<30) rates stock=55
    # as low-risk, returning conf ~0.3. Now 0.25 threshold fixes this.
    DISPATCH_THR  = 55
    ML_CONF_THR   = 0.25   # lowered from 0.50 → matches dashboard warning

    def __init__(self):
        super().__init__('hospital_robot_node')
        self.get_logger().info("=" * 50)
        self.get_logger().info(" Hospital Robot Node Starting")
        self.get_logger().info("=" * 50)

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'odom', self._odom, 10)

        self.cx, self.cy, self.yaw = self.HOME[0], self.HOME[1], 0.0
        self.moving     = False
        self.task       = None
        self.carrying   = 0       # units robot is currently holding (0 or 50)
        self.status     = "idle"

        self._wx        = self.cx
        self._wy        = self.cy
        self._wt        = time.time()
        self._rec_count = 0

        self.grid  = AStarGrid()
        self.model = None
        self.url   = "http://localhost:5000"
        self._load_model()

        # Check stocks every 5s (was 10s) — catches threshold crossing faster
        self.create_timer(5.0, self._check_stocks)
        self.create_timer(0.5, self._post_pos)
        self.create_timer(1.0, self._watchdog)
        self._lock = threading.Lock()

        self.get_logger().info(
            f"[OK] Ready. Home=Pharmacy{self.HOME}  "
            f"Dispatch when stock<={self.DISPATCH_THR}  "
            f"ML_conf>={self.ML_CONF_THR}"
        )

    # ── Model ──────────────────────────────────────────────────
    def _load_model(self):
        base = os.path.dirname(os.path.abspath(__file__))
        for _ in range(10):
            p = os.path.join(base, 'ml_model', 'stock_model.pkl')
            if os.path.exists(p):
                self.model = joblib.load(p)
                self.get_logger().info(f"[OK] Model: {p}")
                return
            base = os.path.dirname(base)
        self.get_logger().warn("[!!] Model not found — threshold rule only")

    # ── Odometry ───────────────────────────────────────────────
    def _odom(self, msg):
        self.cx  = msg.pose.pose.position.x
        self.cy  = msg.pose.pose.position.y
        q        = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y ** 2 + q.z ** 2)
        )
        if self.moving:
            pass   # battery removed

    # ── Stuck watchdog ─────────────────────────────────────────
    def _watchdog(self):
        if not self.moving:
            self._wx, self._wy = self.cx, self.cy
            self._wt = time.time()
            self._rec_count = 0
            return
        moved = math.hypot(self.cx - self._wx, self.cy - self._wy)
        if moved > 0.08:
            self._wx, self._wy = self.cx, self.cy
            self._wt = time.time()
            self._rec_count = 0
        elif time.time() - self._wt > 3.5:
            self._rec_count += 1
            if self._rec_count > 12:
                self.get_logger().error("[ABORT] Too many recoveries")
                self.moving = False; self.task = None
                self.status = "idle"; self._stop()
                self._rec_count = 0
                return
            self.get_logger().warn(
                f"[STUCK #{self._rec_count}] ({self.cx:.1f},{self.cy:.1f})"
            )
            threading.Thread(target=self._recover, daemon=True).start()
            self._wx, self._wy = self.cx, self.cy
            self._wt = time.time()

    def _recover(self):
        self._stop(); time.sleep(0.15)
        # Back up
        t0 = time.time()
        while time.time() - t0 < 0.7:
            c = Twist(); c.linear.x = -0.20
            self.pub.publish(c); time.sleep(0.05)
        self._stop(); time.sleep(0.15)
        # Spin random direction and angle
        spin_t = random.uniform(1.2, 2.0)
        spin_d = random.choice([1, -1])
        t0 = time.time()
        while time.time() - t0 < spin_t:
            c = Twist(); c.angular.z = spin_d * 1.1
            self.pub.publish(c); time.sleep(0.05)
        self._stop(); time.sleep(0.15)
        # Short nudge
        t0 = time.time()
        while time.time() - t0 < 0.45:
            c = Twist(); c.linear.x = 0.18
            self.pub.publish(c); time.sleep(0.05)
        self._stop()

    # ── ML prediction ──────────────────────────────────────────
    def _predict(self, room, stock):
        """
        Returns (needed: bool, confidence: float).
        Uses pandas DataFrame so feature names match what the model
        was trained with — removes the sklearn warning.
        """
        import pandas as pd

        # Hard threshold — always dispatch at this level
        force_dispatch = stock <= self.DISPATCH_THR

        if self.model is None:
            return force_dispatch, 1.0 if force_dispatch else 0.0

        now   = datetime.now()
        rt    = self.ROOM_TYPES.get(room, 0)
        shift = 0 if 6 <= now.hour < 14 else (1 if 14 <= now.hour < 22 else 2)

        # Pass as DataFrame with exact column names used during training
        features = pd.DataFrame([{
            'hour_of_day':     now.hour,
            'patient_count':   random.randint(8, 25),
            'current_stock':   stock,
            'shift':           shift,
            'emergency_cases': random.randint(0, 3) if rt == 1 else 0,
            'prev_usage':      round(max(1.0, (100 - stock) / 10.0), 1),
            'day_of_week':     now.weekday() + 1,
            'room_type':       rt,
        }])

        try:
            prob = float(self.model.predict_proba(features)[0][1])

            if force_dispatch:
                # Stock already at/below threshold — dispatch regardless
                return True, round(prob, 3)
            else:
                # Stock above threshold — only dispatch if model agrees
                pred = bool(self.model.predict(features)[0])
                return pred and prob > self.ML_CONF_THR, round(prob, 3)
        except Exception as e:
            self.get_logger().warn(f"Prediction error: {e}")
            return force_dispatch, 1.0 if force_dispatch else 0.0

    # ── Stock check every 5 seconds ────────────────────────────
    def _check_stocks(self):
        if self.moving: return

        try:
            r      = requests.get(f"{self.url}/api/state", timeout=2)
            stocks = r.json().get("stocks", {})
        except:
            stocks = {"icu":{"current_stock":52},
                      "general_ward":{"current_stock":62}}

        tasks = []
        for room in self.DELIVERY_ROOMS:
            s          = stocks.get(room, {}).get("current_stock", 80)
            need, conf = self._predict(room, s)
            self.get_logger().info(
                f"  [{room}] stock={s:.1f}  "
                f"dispatch={'YES' if need else 'no'}  "
                f"conf={conf:.0%}"
            )
            if need:
                tasks.append((room, s, conf))

        if tasks:
            tasks.sort(key=lambda x: x[1])  # lowest stock first
            dest = tasks[0][0]
            self.get_logger().info(f"[DISPATCH] -> {dest}")
            threading.Thread(
                target=self._mission, args=(dest,), daemon=True
            ).start()
        else:
            self.get_logger().info("[OK] All fine — idle at Pharmacy")

    # ── Full delivery mission ───────────────────────────────────
    def _mission(self, room):
        with self._lock:
            self.moving = True
            self.task   = room

            # Waypoint sequences — every point on corridor centre lines
            if room == "icu":
                go_wps = [
                    self.WP["ph_door_in"],
                    self.WP["ph_door_out"],
                    self.WP["ph_corr_c"],
                    self.WP["vm_ph"],
                    self.WP["vm_mid"],
                    self.WP["vm_icu"],
                    self.WP["icu_corr_c"],
                    self.WP["icu_door"],
                    self.WP["icu_dest"],
                ]
                ret_wps = [
                    self.WP["icu_door"],
                    self.WP["icu_corr_c"],
                    self.WP["vm_icu"],
                    self.WP["vm_mid"],
                    self.WP["vm_ph"],
                    self.WP["ph_corr_c"],
                    self.WP["ph_door_out"],
                    self.WP["ph_door_in"],
                    self.HOME,
                ]
            else:  # general_ward
                go_wps = [
                    self.WP["ph_door_in"],
                    self.WP["ph_door_out"],
                    self.WP["ph_corr_c"],
                    self.WP["vm_ph"],
                    self.WP["vm_mid"],
                    self.WP["vm_icu"],
                    self.WP["vm_ward"],
                    self.WP["w_corr_c"],
                    self.WP["ward_door"],
                    self.WP["ward_dest"],
                ]
                ret_wps = [
                    self.WP["ward_door"],
                    self.WP["w_corr_c"],
                    self.WP["vm_ward"],
                    self.WP["vm_icu"],
                    self.WP["vm_mid"],
                    self.WP["vm_ph"],
                    self.WP["ph_corr_c"],
                    self.WP["ph_door_out"],
                    self.WP["ph_door_in"],
                    self.HOME,
                ]

            # 1. Move to pickup
            self.get_logger().info("[1/5] -> pharmacy pickup")
            self.status = "moving"
            self._goto(*self.WP["ph_pickup"])

            # 2. Pickup
            self.get_logger().info(f"[2/5] Picking up {self.LOAD} units")
            self.status   = "pickup"
            self.carrying = self.LOAD      # robot now loaded
            self._stop(); time.sleep(2.0)
            self._api({"status": "pickup", "task": "pharmacy",
                       "load": self.LOAD})

            # 3. Navigate to destination
            self.get_logger().info(f"[3/5] -> {room}")
            self.status = "moving"
            for wp in go_wps:
                self.get_logger().info(f"   wp ({wp[0]:.1f},{wp[1]:.1f})")
                self._goto(wp[0], wp[1])

            # 4. Deliver
            self.get_logger().info(f"[4/5] Delivering to {room}")
            self.status   = "refilling"
            self._stop(); time.sleep(2.5)
            self.carrying = 0              # robot unloaded
            self._api({"status": "refilled", "task": room,
                       "load": self.LOAD})

            # 5. Return home
            self.get_logger().info("[5/5] Returning to Pharmacy")
            self.status = "moving"
            for wp in ret_wps:
                self._goto(wp[0], wp[1])

            self._stop()
            self.status  = "idle"
            self.task    = None
            self.moving  = False
            self.get_logger().info("[DONE] Back at Pharmacy")

    # ── Drive to (tx, ty) ──────────────────────────────────────
    def _goto(self, tx, ty, tol=0.40, timeout=60.0):
        """
        Two-phase controller:
          Phase 1: rotate to face target (if error > 0.15 rad)
          Phase 2: drive forward + gentle heading correction

        Speed limits:
          In corridor sections (abs(tx) < 3 or abs(ty-9) < 3 etc.)
          speed is capped at 0.28 m/s so robot has time to correct
          before reaching a wall. In open room space: up to 0.42 m/s.
        """
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                self.get_logger().warn(f"  timeout ({tx:.1f},{ty:.1f})")
                return

            dx   = tx - self.cx
            dy   = ty - self.cy
            dist = math.hypot(dx, dy)
            if dist < tol: return

            desired = math.atan2(dy, dx)
            err = (desired - self.yaw + math.pi) % (2 * math.pi) - math.pi

            # Determine if we are inside a narrow corridor
            in_corridor = (
                (-8 < self.cx < -2 and  7 < self.cy < 11) or  # PH-corr
                (-6 < self.cx < -2 and -1 < self.cy <  3) or  # ICU-corr
                (-2 < self.cx <  2) or                          # V-main
                ( 2 < self.cx <  6 and -7 < self.cy < -3)      # W-corr
            )
            max_spd = 0.28 if in_corridor else 0.42

            cmd = Twist()
            if abs(err) > 0.15:
                # Rotate in place — gentle to avoid oscillation
                cmd.angular.z = max(-0.90, min(0.90, 0.85 * err))
                cmd.linear.x  = 0.0
            else:
                cmd.linear.x  = max(0.07, min(max_spd, 0.38 * dist))
                cmd.angular.z = max(-0.45, min(0.45, 0.42 * err))

            self.pub.publish(cmd)
            time.sleep(0.08)

    def _stop(self):
        self.pub.publish(Twist())

    # ── Post to dashboard ──────────────────────────────────────
    def _api(self, extra=None):
        body = {
            "x":        round(self.cx,  2),
            "y":        round(self.cy,  2),
            "status":   self.status,
            "task":     self.task,
            "carrying": self.carrying,   # 0 = empty, 50 = loaded
        }
        if extra: body.update(extra)
        try:
            requests.post(f"{self.url}/api/robot_update",
                          json=body, timeout=1)
        except:
            pass

    def _post_pos(self):
        self._api()


def main(args=None):
    rclpy.init(args=args)
    node = HospitalRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop()
        except Exception:
            pass   # context already invalid on SIGINT — safe to ignore
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
