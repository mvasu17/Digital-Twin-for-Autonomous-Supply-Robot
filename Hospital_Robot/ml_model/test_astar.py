#!/usr/bin/env python3
"""
=============================================================
STANDALONE A* PATHFINDING VISUALIZER
=============================================================
Run this to TEST the A* algorithm without needing ROS2 or Gazebo.
It will show a matplotlib visualization of the hospital map
and the path the robot would take.

Usage:
  python3 test_astar.py
=============================================================
"""

import math
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class AStarGrid:
    """Same A* implementation as in robot_node.py"""

    def __init__(self, width=60, height=32, cell_size=0.5):
        self.width     = width
        self.height    = height
        self.cell_size = cell_size
        self.origin_x  = width  // 2
        self.origin_y  = height // 2
        self.grid = [[0] * height for _ in range(width)]
        self._build_hospital_map()

    def _build_hospital_map(self):
        for gx in range(self.width):
            for gy in range(self.height):
                wx, wy = self.grid_to_world(gx, gy)
                if abs(wx) > 14.5 or abs(wy) > 6.8:
                    self.grid[gx][gy] = 1; continue
                if -2.5 < wx < -1.8 and abs(wy) > 2.2:
                    self.grid[gx][gy] = 1
                if -6.0 < wx < -5.4 and abs(wy) > 2.2:
                    self.grid[gx][gy] = 1
                if 5.4 < wx < 6.0 and abs(wy) > 2.2:
                    self.grid[gx][gy] = 1
                if 1.8 < wx < 2.5 and abs(wy) > 2.2:
                    self.grid[gx][gy] = 1
                if -14.0 < wx < -2.0 and abs(wy) > 5.6:
                    self.grid[gx][gy] = 1
                if -5.85 < wx < 5.85 and abs(wy) > 5.6:
                    self.grid[gx][gy] = 1
                if 2.0 < wx < 14.0 and abs(wy) > 5.6:
                    self.grid[gx][gy] = 1
                for bx in [-3.0, 0.0, 3.0]:
                    if abs(wx - bx) < 1.1 and 1.5 < wy < 2.8:
                        self.grid[gx][gy] = 1
                for bx in [5.0, 8.0, 11.0]:
                    if abs(wx - bx) < 1.1 and 1.5 < wy < 2.8:
                        self.grid[gx][gy] = 1

    def world_to_grid(self, wx, wy):
        gx = int(round(wx / self.cell_size)) + self.origin_x
        gy = int(round(wy / self.cell_size)) + self.origin_y
        return max(0, min(self.width-1, gx)), max(0, min(self.height-1, gy))

    def grid_to_world(self, gx, gy):
        return (gx - self.origin_x) * self.cell_size, (gy - self.origin_y) * self.cell_size

    def is_free(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height and self.grid[gx][gy] == 0

    def heuristic(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def get_neighbors(self, gx, gy):
        neighbors = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx == 0 and dy == 0: continue
                nx, ny = gx+dx, gy+dy
                if self.is_free(nx, ny):
                    cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    neighbors.append((nx, ny, cost))
        return neighbors

    def find_path(self, swx, swy, gwx, gwy):
        start = self.world_to_grid(swx, swy)
        goal  = self.world_to_grid(gwx, gwy)

        # Find nearest free cell to goal
        if not self.is_free(*goal):
            for r in range(1, 8):
                found = False
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        ng = (goal[0]+dx, goal[1]+dy)
                        if self.is_free(*ng):
                            goal = ng; found = True; break
                    if found: break
                if found: break

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score   = {start: 0.0}
        closed    = set()
        expanded  = []

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur in closed: continue
            closed.add(cur)
            expanded.append(cur)

            if self.heuristic(cur, goal) < 1.5:
                path = []
                while cur in came_from:
                    path.append(self.grid_to_world(*cur))
                    cur = came_from[cur]
                path.reverse()
                return path, expanded

            for nx, ny, cost in self.get_neighbors(*cur):
                if (nx,ny) in closed: continue
                tg = g_score[cur] + cost
                if tg < g_score.get((nx,ny), 1e9):
                    came_from[(nx,ny)] = cur
                    g_score[(nx,ny)]   = tg
                    f = tg + self.heuristic((nx,ny), goal)
                    heapq.heappush(open_set, (f, (nx,ny)))

        return [], expanded


def visualize():
    print("Building A* grid...")
    grid = AStarGrid(width=60, height=32, cell_size=0.5)

    # Test: Robot moves Pharmacy → ICU → General Ward → back
    missions = [
        ((-8.0, 0.0), (0.0, -1.0),  "Pharmacy → ICU",          'cyan'),
        ((0.0, -1.0), (8.0, 0.0),   "ICU → General Ward",      'lime'),
        ((8.0, 0.0),  (0.0, 0.0),   "General Ward → Return",   'orange'),
    ]

    # ── Build grid image ──
    img = np.zeros((grid.height, grid.width, 3))
    for gx in range(grid.width):
        for gy in range(grid.height):
            wx, wy = grid.grid_to_world(gx, gy)
            if grid.grid[gx][gy] == 1:
                img[gy, gx] = [0.3, 0.3, 0.35]  # Wall: dark gray
            elif -14 < wx < -2:
                img[gy, gx] = [0.05, 0.1, 0.25]   # Pharmacy: dark blue
            elif -6 < wx < 6:
                img[gy, gx] = [0.25, 0.05, 0.05]  # ICU: dark red
            elif 2 < wx < 14:
                img[gy, gx] = [0.05, 0.2, 0.05]   # General: dark green
            else:
                img[gy, gx] = [0.08, 0.1, 0.12]   # Corridor

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0a0e1a')

    # ── LEFT: Full map with all paths ──
    ax = axes[0]
    ax.set_facecolor('#0a0e1a')
    ax.imshow(img, origin='lower', extent=[-15, 15, -8, 8], aspect='equal')

    colors = ['cyan', 'lime', 'orange']
    total_nodes = 0

    for (start, goal, label, color) in missions:
        path, expanded = grid.find_path(*start, *goal)
        total_nodes += len(expanded)

        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, '-', color=color, linewidth=2.5, label=label, zorder=5)
            ax.plot(px[0], py[0], 'o', color=color, markersize=10, zorder=6)
            ax.plot(px[-1], py[-1], 's', color=color, markersize=10, zorder=6)

            # Draw arrows along path
            for i in range(0, len(path)-1, max(1, len(path)//6)):
                dx = path[i+1][0] - path[i][0]
                dy = path[i+1][1] - path[i][1]
                ax.annotate('', xy=(path[i][0]+dx*0.6, path[i][1]+dy*0.6),
                           xytext=(path[i][0], path[i][1]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Room labels
    for name, (cx, cy), color in [
        ("💊 PHARMACY",    (-8, 0),  '#3b82f6'),
        ("🏥 ICU",         ( 0, 0),  '#ef4444'),
        ("🛏 GENERAL WARD",(8, 0),  '#22c55e'),
    ]:
        ax.text(cx, cy, name, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

    ax.set_title('Hospital Map — A* Paths', color='white', fontsize=13, pad=10)
    ax.set_xlabel('X (meters)', color='#94a3b8')
    ax.set_ylabel('Y (meters)', color='#94a3b8')
    ax.tick_params(colors='#64748b')
    ax.legend(loc='upper right', facecolor='#1e293b', labelcolor='white', fontsize=9)
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values(): spine.set_color('#1e293b')

    # ── RIGHT: A* expansion detail (ICU → General) ──
    ax2 = axes[1]
    ax2.set_facecolor('#0a0e1a')
    ax2.imshow(img, origin='lower', extent=[-15, 15, -8, 8], aspect='equal', alpha=0.7)

    path2, expanded2 = grid.find_path(0.0, -1.0, 8.0, 0.0)

    # Show expanded nodes (the cells A* visited)
    if expanded2:
        ex = [(grid.grid_to_world(c[0],c[1])[0]) for c in expanded2]
        ey = [(grid.grid_to_world(c[0],c[1])[1]) for c in expanded2]
        ax2.scatter(ex, ey, c='#6366f1', s=6, alpha=0.3, label=f'Nodes explored ({len(expanded2)})', zorder=3)

    if path2:
        px = [p[0] for p in path2]
        py = [p[1] for p in path2]
        ax2.plot(px, py, '-', color='lime', linewidth=3, label=f'Optimal path ({len(path2)} pts)', zorder=5)
        ax2.plot(px[0], py[0], 'go', markersize=14, label='Start (ICU)', zorder=6)
        ax2.plot(px[-1],py[-1],'r*', markersize=14, label='Goal (Gen Ward)', zorder=6)

    ax2.set_title('A* Expansion Detail: ICU → General Ward', color='white', fontsize=12, pad=10)
    ax2.set_xlabel('X (meters)', color='#94a3b8')
    ax2.tick_params(colors='#64748b')
    ax2.legend(loc='upper left', facecolor='#1e293b', labelcolor='white', fontsize=9)
    ax2.grid(True, alpha=0.15, color='white')
    for spine in ax2.spines.values(): spine.set_color('#1e293b')

    plt.suptitle('Hospital Robot — A* Pathfinding Visualization',
                 color='white', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    print(f"\n A* Statistics:")
    print(f"   Total nodes expanded: {total_nodes}")
    print(f"   Grid size: {grid.width}×{grid.height} = {grid.width*grid.height} cells")
    print(f"   Cell size: {grid.cell_size}m")
    print(f"   Paths found: {len(missions)}/3")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'astar_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    print(f"\n[✓] Visualization saved: {output_path}")
    plt.show()


if __name__ == '__main__':
    import os
    visualize()
