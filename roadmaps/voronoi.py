import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
from collections import deque
import heapq

# ==============================================================================
# SHARED ENVIRONMENT CONFIGURATION (18x12 coordinate space)
# ==============================================================================
GRID_COLS = 18
GRID_ROWS = 12

# Obstacle 1: Top-left L/C-shaped obstacle
OBS_1_CELLS = [
    (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (11, 10),
    (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9), (11, 9),
    (2, 8), (3, 8), (4, 8), (5, 8), (6, 8),
    (2, 7), (3, 7), (4, 7), (5, 7),
    (2, 6), (3, 6), (4, 6)
]

# Obstacle 2: Top-right rectangular obstacle
OBS_2_CELLS = [
    (14, 10), (15, 10),
    (14, 9), (15, 9),
    (14, 8), (15, 8)
]

# Obstacle 3: Bottom stepped obstacle
OBS_3_CELLS = [
    (10, 4), (11, 4), (12, 4), (13, 4), (14, 4),
    (7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (12, 3), (13, 3), (14, 3),
    (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (12, 2), (13, 2), (14, 2)
]

ALL_OBSTACLE_GROUPS = [OBS_1_CELLS, OBS_2_CELLS, OBS_3_CELLS]

# Start and Goal coordinates
START_CELL = (1, 10)
GOAL_CELL = (16, 5)
START_POS = (START_CELL[0] + 0.5, START_CELL[1] + 0.5)
GOAL_POS = (GOAL_CELL[0] + 0.5, GOAL_CELL[1] + 0.5)

# Polygonal obstacles for continuous Sensor-Based method
OBSTACLE_POLYGONS = [
    Polygon([(4, 11), (12, 11), (12, 9), (7, 9), (7, 8), (6, 8), (6, 7), (5, 7), (5, 6), (2, 6), (2, 9), (3, 9), (3, 10), (4, 10)]),
    Polygon([(14, 11), (16, 11), (16, 8), (14, 8)]),
    Polygon([(10, 5), (15, 5), (15, 2), (5, 2), (5, 3), (7, 3), (7, 4), (10, 4)])
]

# Environment boundary walls treated as obstacles in Sensor-Based mode
BOUNDARY_WALLS = [
    LineString([(0, 0), (GRID_COLS, 0)]),             # Bottom wall
    LineString([(GRID_COLS, 0), (GRID_COLS, GRID_ROWS)]),  # Right wall
    LineString([(GRID_COLS, GRID_ROWS), (0, GRID_ROWS)]),  # Top wall
    LineString([(0, GRID_ROWS), (0, 0)])              # Left wall
]

# Combined list of all detectable geometries (internal obstacles + boundary walls)
SENSOR_OBJECTS = OBSTACLE_POLYGONS + BOUNDARY_WALLS


# ==============================================================================
# 1. SENSOR-BASED VORONOI CONSTRUCTION GIF
# ==============================================================================
def generate_sensor_based_gif(output_filename="voronoi_sensor_based.gif"):
    print("Computing sensor-based Voronoi trajectory (with boundary walls as obstacles)...")

    def sense_environment(pt_tuple):
        """Returns sorted list of (distance, closest_point_coords, object_id) for all obstacles and walls."""
        p = Point(pt_tuple)
        readings = []
        for idx, geom in enumerate(SENSOR_OBJECTS):
            cp, _ = nearest_points(geom, p)
            dist = np.hypot(pt_tuple[0] - cp.x, pt_tuple[1] - cp.y)
            readings.append((dist, (cp.x, cp.y), idx))
        readings.sort(key=lambda item: item[0])
        return readings

    # Step A: Gradient ascent from START_POS until equidistant to the two closest objects
    gradient_path = [START_POS]
    curr = np.array(START_POS, dtype=float)
    step_size = 0.12

    for _ in range(40):
        readings = sense_environment(tuple(curr))
        d1, cp1, _ = readings[0]
        d2, cp2, _ = readings[1]
        if abs(d1 - d2) < 0.08:
            break
        # Ascend gradient of distance from the single closest obstacle/wall
        vec = curr - np.array(cp1)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            curr = curr + step_size * (vec / norm)
            gradient_path.append(tuple(curr))

    voronoi_entry_pt = gradient_path[-1]

    # Step B: Sample continuous space to extract Voronoi skeleton nodes
    res = 0.15
    xs = np.arange(0.3, GRID_COLS - 0.3, res)
    ys = np.arange(0.3, GRID_ROWS - 0.3, res)

    voronoi_nodes = {}
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            p = Point(x, y)
            if any(poly.buffer(0.12).contains(p) for poly in OBSTACLE_POLYGONS):
                continue
            readings = sense_environment((x, y))
            d1, _, _ = readings[0]
            d2, _, _ = readings[1]
            # Equidistance condition d(q_i, q) == d(q_j, q)
            if abs(d1 - d2) < 0.14:
                voronoi_nodes[(ix, iy)] = (x, y)

    def nearest_voronoi_key(target_pt):
        return min(
            voronoi_nodes.keys(),
            key=lambda k: np.hypot(voronoi_nodes[k][0] - target_pt[0], voronoi_nodes[k][1] - target_pt[1])
        )

    start_v_key = nearest_voronoi_key(voronoi_entry_pt)
    goal_v_key = nearest_voronoi_key(GOAL_POS)

    # Dijkstra search along the Voronoi skeleton (allowing 2-cell jumps to prevent micro-gaps)
    pq = [(0.0, start_v_key)]
    came_from = {start_v_key: None}
    cost_so_far = {start_v_key: 0.0}

    while pq:
        _, current_k = heapq.heappop(pq)
        if current_k == goal_v_key:
            break
        cx, cy = current_k
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0:
                    continue
                nk = (cx + dx, cy + dy)
                if nk in voronoi_nodes:
                    p1, p2 = voronoi_nodes[current_k], voronoi_nodes[nk]
                    step_cost = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    # Slightly penalize 2-step jumps to prefer contiguous 1-step nodes
                    penalty = 1.0 if max(abs(dx), abs(dy)) == 1 else 1.25
                    new_cost = cost_so_far[current_k] + step_cost * penalty
                    if nk not in cost_so_far or new_cost < cost_so_far[nk]:
                        cost_so_far[nk] = new_cost
                        heapq.heappush(pq, (new_cost, nk))
                        came_from[nk] = current_k

    # Reconstruct and subsample the Voronoi trace for smooth animation speed
    raw_trace = []
    curr_k = goal_v_key
    while curr_k is not None:
        raw_trace.append(voronoi_nodes[curr_k])
        curr_k = came_from.get(curr_k)
    raw_trace.reverse()
    voronoi_trace = raw_trace[::2] if len(raw_trace) > 2 else raw_trace
    if voronoi_trace[-1] != raw_trace[-1]:
        voronoi_trace.append(raw_trace[-1])

    # Step C: Smooth interpolation from the Voronoi exit point to GOAL_POS
    last_v_pt = np.array(voronoi_trace[-1])
    goal_arr = np.array(GOAL_POS)
    dist_to_goal = np.linalg.norm(goal_arr - last_v_pt)
    num_exit_steps = max(2, int(dist_to_goal / 0.25))
    exit_path = [
        tuple(last_v_pt + (i / num_exit_steps) * (goal_arr - last_v_pt))
        for i in range(1, num_exit_steps + 1)
    ]

    # Precompute sensor readings for all trajectory points to optimize frame rendering
    full_trajectory = []
    for pt in gradient_path:
        full_trajectory.append((pt, "gradient", sense_environment(pt)))
    for pt in voronoi_trace:
        full_trajectory.append((pt, "voronoi", sense_environment(pt)))
    for pt in exit_path:
        full_trajectory.append((pt, "depart", sense_environment(pt)))

    # Animation setup
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    frames_init = 10
    frames_move = len(full_trajectory)
    frames_hold = 15
    total_frames = frames_init + frames_move + frames_hold

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(0, GRID_COLS)
        ax.set_ylim(0, GRID_ROWS)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Highlight environment boundary as a physical obstacle wall
        boundary_rect = patches.Rectangle(
            (0, 0), GRID_COLS, GRID_ROWS, fill=False, edgecolor='black', linewidth=4.0, zorder=3
        )
        ax.add_patch(boundary_rect)

        # Draw internal polygonal obstacles
        for poly in OBSTACLE_POLYGONS:
            x_coords, y_coords = poly.exterior.xy
            patch = patches.Polygon(
                list(zip(x_coords, y_coords)), closed=True,
                facecolor='#D3D3D3', edgecolor='black', linewidth=1.5, zorder=2
            )
            ax.add_patch(patch)

        # Draw Start and Goal points
        ax.scatter(*START_POS, c='blue', s=90, zorder=6)
        ax.text(START_POS[0], START_POS[1] + 0.35, 'start', fontsize=11, ha='center', va='bottom', fontweight='bold')
        ax.scatter(*GOAL_POS, c='blue', s=90, zorder=6)
        ax.text(GOAL_POS[0], GOAL_POS[1] - 0.35, 'goal', fontsize=11, ha='center', va='top', fontweight='bold')

        if frame_idx < frames_init:
            ax.set_title("Sensor-Based Voronoi: Initial Environment & Boundary Walls", fontsize=12)
            return

        step_idx = min(frame_idx - frames_init, len(full_trajectory) - 1)
        robot_pos, mode, readings = full_trajectory[step_idx]

        # Extract traversed sub-paths up to current step
        grad_pts = [pt for pt, m, _ in full_trajectory[:step_idx + 1] if m == "gradient"]
        vor_pts = [pt for pt, m, _ in full_trajectory[:step_idx + 1] if m == "voronoi"]
        dep_pts = [pt for pt, m, _ in full_trajectory[:step_idx + 1] if m == "depart"]

        if len(grad_pts) > 1:
            gx, gy = zip(*grad_pts)
            ax.plot(gx, gy, color='#2B5B84', linestyle='-', linewidth=2.0, zorder=4)
        if len(vor_pts) > 1:
            vx, vy = zip(*vor_pts)
            ax.plot(vx, vy, color='black', linestyle='--', linewidth=2.0, zorder=4)
        if len(dep_pts) > 0 and len(vor_pts) > 0:
            dx_p, dy_p = zip(*([vor_pts[-1]] + dep_pts))
            ax.plot(dx_p, dy_p, color='#D9534F', linestyle='--', linewidth=2.0, zorder=4)

        if frame_idx < frames_init + frames_move:
            d1, cp1, _ = readings[0]
            d2, cp2, _ = readings[1]

            if mode == "gradient":
                ax.set_title("Phase 1: Gradient Ascent (Moving Away from Closest Wall/Obstacle)", fontsize=11)
                ax.plot([robot_pos[0], cp1[0]], [robot_pos[1], cp1[1]], color='black', linewidth=1.2, zorder=5)
                vec = np.array(robot_pos) - np.array(cp1)
                if np.linalg.norm(vec) > 1e-6:
                    vec = (vec / np.linalg.norm(vec)) * 0.7
                    ax.annotate(
                        '', xy=(robot_pos[0] + vec[0], robot_pos[1] + vec[1]), xytext=robot_pos,
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.8), zorder=7
                    )
            elif mode == "voronoi":
                ax.set_title("Phase 2: Tracing Voronoi Edge (Equidistant to Obstacles/Walls)", fontsize=11)
                circle = patches.Circle(
                    robot_pos, radius=d1, fill=False, edgecolor='#666666', linestyle=':', linewidth=1.3, zorder=3
                )
                ax.add_patch(circle)
                ax.plot([robot_pos[0], cp1[0]], [robot_pos[1], cp1[1]], color='#666666', linestyle=':', linewidth=1.2)
                ax.plot([robot_pos[0], cp2[0]], [robot_pos[1], cp2[1]], color='#666666', linestyle=':', linewidth=1.2)
            else:
                ax.set_title("Phase 3: Departing Voronoi Roadmap to Goal", fontsize=11)

            # Render Robot
            ax.scatter(*robot_pos, c='#888888', edgecolors='black', s=85, linewidth=1.5, zorder=8)
        else:
            ax.set_title("Sensor-Based Voronoi Navigation Complete", fontsize=12)
            all_x, all_y = zip(*[pt for pt, _, _ in full_trajectory])
            ax.plot(all_x, all_y, color='red', linewidth=2.8, zorder=5)

    anim = FuncAnimation(fig, update, frames=total_frames, interval=90)
    anim.save(output_filename, writer=PillowWriter(fps=11))
    plt.close(fig)
    print(f"Saved '{output_filename}' successfully!")


# ==============================================================================
# 2. GRID-BASED VORONOI CONSTRUCTION GIF
# ==============================================================================
def generate_grid_based_gif(output_filename="voronoi_grid_based.gif"):
    print("Computing grid-based wave-front Voronoi diagram...")

    owner_grid = np.zeros((GRID_COLS, GRID_ROWS), dtype=int)
    is_obstacle = np.zeros((GRID_COLS, GRID_ROWS), dtype=bool)
    is_voronoi = np.zeros((GRID_COLS, GRID_ROWS), dtype=bool)

    queue = deque()

    # Seed wave-front from the 3 obstacles (matching slides 22-27)
    for obs_id, cell_list in enumerate(ALL_OBSTACLE_GROUPS, start=1):
        for (c, r) in cell_list:
            owner_grid[c, r] = obs_id
            is_obstacle[c, r] = True
            queue.append((c, r, obs_id, 0))

    snapshots = [(owner_grid.copy(), is_voronoi.copy())]
    current_wave = 0
    neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        c, r, obs_id, dist = queue.popleft()
        if dist > current_wave:
            snapshots.append((owner_grid.copy(), is_voronoi.copy()))
            current_wave = dist

        for dc, dr in neighbors_4:
            nc, nr = c + dc, r + dr
            if 0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS:
                if is_obstacle[nc, nr] or is_voronoi[nc, nr]:
                    continue
                if owner_grid[nc, nr] == 0:
                    owner_grid[nc, nr] = obs_id
                    queue.append((nc, nr, obs_id, dist + 1))
                elif owner_grid[nc, nr] != obs_id:
                    is_voronoi[nc, nr] = True

    snapshots.append((owner_grid.copy(), is_voronoi.copy()))

    # Path planning along the grid Voronoi skeleton
    voronoi_cells = set(zip(*np.where(is_voronoi)))

    def connect_to_skeleton(cell):
        q = deque([(cell, [cell])])
        visited = {cell}
        while q:
            curr, path = q.popleft()
            if curr in voronoi_cells:
                return path
            for dc in [-1, 0, 1]:
                for dr in [-1, 0, 1]:
                    nxt = (curr[0] + dc, curr[1] + dr)
                    if (0 <= nxt[0] < GRID_COLS and 0 <= nxt[1] < GRID_ROWS and
                            not is_obstacle[nxt] and nxt not in visited):
                        visited.add(nxt)
                        q.append((nxt, path + [nxt]))
        return [cell]

    start_bridge = connect_to_skeleton(START_CELL)
    goal_bridge = connect_to_skeleton(GOAL_CELL)
    v_start, v_goal = start_bridge[-1], goal_bridge[-1]

    q = deque([(v_start, [v_start])])
    visited_v = {v_start}
    skeleton_path = [v_start]
    while q:
        curr, path = q.popleft()
        if curr == v_goal:
            skeleton_path = path
            break
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                nxt = (curr[0] + dc, curr[1] + dr)
                if nxt in voronoi_cells and nxt not in visited_v:
                    visited_v.add(nxt)
                    q.append((nxt, path + [nxt]))

    full_grid_path = start_bridge + skeleton_path[1:] + list(reversed(goal_bridge))[1:]
    path_coords = [(c + 0.5, r + 0.5) for (c, r) in full_grid_path]

    wave_colors = {
        1: '#F47A7A',  # Coral red wave
        2: '#F6B27A',  # Soft orange wave
        3: '#E5CD7B'   # Ochre yellow wave
    }

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    frames_init = 10
    frames_waves = len(snapshots) * 3
    frames_path = 18
    total_frames = frames_init + frames_waves + frames_path

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(0, GRID_COLS)
        ax.set_ylim(0, GRID_ROWS)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        if frame_idx < frames_init:
            snap_owner, snap_vor = snapshots[0]
            ax.set_title("Grid-Based Voronoi: Initial Environment & Obstacles", fontsize=12)
        elif frame_idx < frames_init + frames_waves:
            snap_idx = min((frame_idx - frames_init) // 3, len(snapshots) - 1)
            snap_owner, snap_vor = snapshots[snap_idx]
            ax.set_title(f"Wave-Front Growth & Collision Detection (Step {snap_idx}/{len(snapshots)-1})", fontsize=11)
        else:
            snap_owner, snap_vor = snapshots[-1]
            ax.set_title("Grid Fully Divided into Voronoi Regions + Planned Path", fontsize=11)

        for c in range(GRID_COLS):
            for r in range(GRID_ROWS):
                if is_obstacle[c, r]:
                    cell_color = '#595959'
                elif snap_vor[c, r]:
                    cell_color = '#FF1A1A'
                elif snap_owner[c, r] > 0:
                    cell_color = wave_colors[snap_owner[c, r]]
                else:
                    cell_color = 'white'

                rect = patches.Rectangle(
                    (c, r), 1, 1, facecolor=cell_color, edgecolor='black', linewidth=1.0, zorder=1
                )
                ax.add_patch(rect)

        if frame_idx >= frames_init + frames_waves:
            progress = min(1.0, (frame_idx - (frames_init + frames_waves) + 1) / (frames_path * 0.7))
            num_pts = max(2, int(progress * len(path_coords)))
            px, py = zip(*path_coords[:num_pts])
            ax.plot(px, py, color='#0044CC', linewidth=3.2, zorder=4)

        ax.scatter(*START_POS, c='blue', edgecolors='white', s=100, linewidth=1.2, zorder=5)
        ax.text(START_POS[0], START_POS[1] + 0.35, 'start', fontsize=11, ha='center', va='bottom', fontweight='bold')
        ax.scatter(*GOAL_POS, c='blue', edgecolors='white', s=100, linewidth=1.2, zorder=5)
        ax.text(GOAL_POS[0], GOAL_POS[1] - 0.35, 'goal', fontsize=11, ha='center', va='top', fontweight='bold')

    anim = FuncAnimation(fig, update, frames=total_frames, interval=110)
    anim.save(output_filename, writer=PillowWriter(fps=9))
    plt.close(fig)
    print(f"Saved '{output_filename}' successfully!")


if __name__ == "__main__":
    generate_sensor_based_gif("voronoi_sensor_based.gif")
    generate_grid_based_gif("voronoi_grid_based.gif")
    print("All Voronoi animations generated successfully!")