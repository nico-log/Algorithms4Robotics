import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from shapely.geometry import Polygon, Point, LineString
import networkx as nx

def generate_cell_decomposition_gif(output_filename="cell_decomposition.gif"):
    print("Initializing environment and computing exact trapezoidal decomposition...")

    # 1. EXACT VERTEX COORDINATES (matching v1..v18 from Slide 32)
    V = {
        # Outer boundary vertices (v12 to v18)
        'v16': (0.4, 5.2),
        'v17': (1.3, 9.1),
        'v18': (4.2, 9.7),
        'v12': (9.2, 9.2),
        'v13': (10.9, 3.6),
        'v14': (8.4, 0.9),
        'v15': (2.2, 0.5),
        # C-shaped obstacle vertices (v5 to v11)
        'v11': (1.9, 7.6),
        'v5':  (7.5, 8.6),
        'v6':  (8.0, 7.9),
        'v7':  (2.8, 6.8),
        'v8':  (4.0, 2.3),
        'v9':  (8.5, 1.9),
        'v10': (3.1, 1.4),
        # Central obstacle vertices (v1 to v4)
        'v4':  (5.4, 3.9),
        'v1':  (5.9, 6.6),
        'v2':  (9.6, 4.3),
        'v3':  (6.8, 2.6)
    }

    outer_coords = [V['v16'], V['v17'], V['v18'], V['v12'], V['v13'], V['v14'], V['v15']]
    obs_c_coords = [V['v11'], V['v5'], V['v6'], V['v7'], V['v8'], V['v9'], V['v10']]
    obs_mid_coords = [V['v4'], V['v1'], V['v2'], V['v3']]

    outer_poly = Polygon(outer_coords)
    obs_c_poly = Polygon(obs_c_coords)
    obs_mid_poly = Polygon(obs_mid_coords)

    # Free space polygon
    free_space = outer_poly.difference(obs_c_poly.buffer(-1e-6)).difference(obs_mid_poly.buffer(-1e-6))

    # Helper to interpolate Y on segment (pA, pB) at a given X
    def interp_y(pA, pB, x):
        xA, yA = pA
        xB, yB = pB
        return yA + (yB - yA) * (x - xA) / (xB - xA)

    # 2. COMPUTE EXACT VERTICAL DECOMPOSITION SEGMENTS FROM CRITICAL VERTICES (Slide 31 & 34)
    vertical_segments = [
        # x = v17 (1.3): down to edge (v16, v15)
        ((V['v17'][0], interp_y(V['v16'], V['v15'], V['v17'][0])), V['v17']),
        # x = v11 (1.9): up to (v17, v18) and down to (v16, v15)
        (V['v11'], (V['v11'][0], interp_y(V['v17'], V['v18'], V['v11'][0]))),
        ((V['v11'][0], interp_y(V['v16'], V['v15'], V['v11'][0])), V['v11']),
        # x = v15 (2.2): up to edge (v11, v10)
        (V['v15'], (V['v15'][0], interp_y(V['v11'], V['v10'], V['v15'][0]))),
        # x = v10 (3.1): down to edge (v15, v14)
        ((V['v10'][0], interp_y(V['v15'], V['v14'], V['v10'][0])), V['v10']),
        # x = v8 (4.0): up to edge (v7, v6)
        (V['v8'], (V['v8'][0], interp_y(V['v7'], V['v6'], V['v8'][0]))),
        # x = v18 (4.2): down to edge (v11, v5)
        ((V['v18'][0], interp_y(V['v11'], V['v5'], V['v18'][0])), V['v18']),
        # x = v4 (5.4): up to edge (v7, v6) and down to edge (v8, v9)
        (V['v4'], (V['v4'][0], interp_y(V['v7'], V['v6'], V['v4'][0]))),
        ((V['v4'][0], interp_y(V['v8'], V['v9'], V['v4'][0])), V['v4']),
        # x = v1 (5.9): up to edge (v7, v6)
        (V['v1'], (V['v1'][0], interp_y(V['v7'], V['v6'], V['v1'][0]))),
        # x = v3 (6.8): down to edge (v8, v9)
        ((V['v3'][0], interp_y(V['v8'], V['v9'], V['v3'][0])), V['v3']),
        # x = v5 (7.5): up to edge (v18, v12)
        (V['v5'], (V['v5'][0], interp_y(V['v18'], V['v12'], V['v5'][0]))),
        # x = v6 (8.0): up to edge (v18, v12) and down to edge (v1, v2)
        (V['v6'], (V['v6'][0], interp_y(V['v18'], V['v12'], V['v6'][0]))),
        ((V['v6'][0], interp_y(V['v1'], V['v2'], V['v6'][0])), V['v6']),
        # x = v14 (8.4): up to edge (v8, v9)
        (V['v14'], (V['v14'][0], interp_y(V['v8'], V['v9'], V['v14'][0]))),
        # x = v9 (8.5): up to edge (v3, v2) and down to edge (v14, v13)
        (V['v9'], (V['v9'][0], interp_y(V['v3'], V['v2'], V['v9'][0]))),
        ((V['v9'][0], interp_y(V['v14'], V['v13'], V['v9'][0])), V['v9']),
        # x = v12 (9.2): down to edge (v1, v2)
        ((V['v12'][0], interp_y(V['v1'], V['v2'], V['v12'][0])), V['v12']),
        # x = v2 (9.6): up to edge (v12, v13) and down to edge (v14, v13)
        (V['v2'], (V['v2'][0], interp_y(V['v12'], V['v13'], V['v2'][0]))),
        ((V['v2'][0], interp_y(V['v14'], V['v13'], V['v2'][0])), V['v2'])
    ]

    # Representative label coordinates for cells c1..c19 (matching Slide 34-37)
    cell_label_pos = {
        'c1':  (0.85, 5.3),
        'c2':  (1.60, 5.0),
        'c3':  (2.05, 3.2),
        'c4':  (2.60, 2.0),
        'c5':  (3.05, 8.6),
        'c6':  (3.45, 5.7),
        'c7':  (4.70, 4.6),
        'c8':  (5.50, 1.3),
        'c9':  (5.65, 6.4),
        'c10': (5.85, 8.8),
        'c11': (6.10, 2.7),
        'c12': (6.95, 6.7),
        'c13': (7.60, 2.5),
        'c14': (7.75, 8.8),
        'c15': (8.45, 2.4),
        'c16': (8.60, 7.0),
        'c17': (9.40, 6.3),
        'c18': (9.05, 3.2),
        'c19': (10.1, 3.8)
    }

    # Start (in c1) and Goal (in c6) points as shown in Slide 36-37
    start_point = (1.0, 6.1)
    goal_point = (3.5, 5.2)

    # 3. BUILD EXACT ADJACENCY GRAPH FROM SLIDE 35-37
    adj_edges = [
        ('c1', 'c2'),
        ('c2', 'c5'), ('c5', 'c10'), ('c10', 'c14'), ('c14', 'c16'),
        ('c2', 'c3'), ('c3', 'c4'), ('c4', 'c8'), ('c8', 'c15'),
        ('c6', 'c7'),
        ('c7', 'c9'), ('c9', 'c12'), ('c12', 'c16'),
        ('c7', 'c11'), ('c11', 'c13'), ('c13', 'c15'),
        ('c16', 'c17'), ('c17', 'c19'),
        ('c15', 'c18'), ('c18', 'c19')
    ]

    adj_graph = nx.Graph()
    for i in range(1, 20):
        adj_graph.add_node(f'c{i}')
    adj_graph.add_edges_from(adj_edges)

    # Node layout for the right-side Adjacency Graph (matching Slide 35-37)
    graph_pos = {
        'c1':  (0.6, 4.8),
        'c2':  (2.0, 4.6),
        'c5':  (3.8, 7.2),
        'c10': (5.4, 8.2),
        'c14': (7.0, 8.1),
        'c6':  (3.3, 4.6),
        'c7':  (4.6, 4.3),
        'c9':  (5.5, 6.1),
        'c12': (7.2, 5.9),
        'c16': (8.5, 6.4),
        'c17': (9.7, 5.1),
        'c19': (10.8, 3.9),
        'c11': (5.6, 3.2),
        'c13': (7.2, 3.2),
        'c18': (9.6, 2.9),
        'c3':  (3.4, 2.1),
        'c4':  (5.0, 1.0),
        'c8':  (6.8, 0.9),
        'c15': (8.4, 2.0)
    }

    # Shortest path on adjacency graph: ['c1', 'c2', 'c5', 'c10', 'c14', 'c16', 'c12', 'c9', 'c7', 'c6']
    path_cells = nx.shortest_path(adj_graph, source='c1', target='c6')

    # Helper to compute the midpoint of a vertical segment at x_val
    def seg_midpoint(p_low, p_high):
        return ((p_low[0] + p_high[0]) / 2.0, (p_low[1] + p_high[1]) / 2.0)

    # Exact waypoints following Slide 37 red trajectory (connecting boundary midpoints & cell centroids)
    waypoints = [
        start_point,
        # Midpoint c1 -> c2 (at x = 1.3)
        (1.3, 5.9),
        # Centroid c2 -> Midpoint c2 -> c5 (at x = 1.9)
        (1.9, 8.4),
        # Centroid c5 -> Midpoint c5 -> c10 (at x = 4.2)
        (3.1, 8.7),
        (4.2, 8.95),
        # Centroid c10 -> Midpoint c10 -> c14 (at x = 7.5)
        (5.9, 8.95),
        (7.5, 8.95),
        # Centroid c14 -> Midpoint c14 -> c16 (at x = 8.0)
        (7.75, 8.75),
        (8.0, 8.6),
        # Centroid c16 -> Midpoint c16 -> c12 (at x = 8.0)
        (8.45, 7.1),
        (8.0, 6.7),
        # Centroid c12 -> Midpoint c12 -> c9 (at x = 5.9)
        (6.9, 6.65),
        (5.9, 6.95),
        # Centroid c9 -> Midpoint c9 -> c7 (at x = 5.4)
        (5.65, 6.0),
        (5.4, 5.55),
        # Centroid c7 -> Midpoint c7 -> c6 (at x = 4.0)
        (4.6, 4.8),
        (4.0, 4.4),
        goal_point
    ]

    # 4. ANIMATION SETUP (Left: Robot Workspace | Right: Adjacency Graph)
    fig, (ax_env, ax_graph) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=100)
    plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05, wspace=0.08)

    f_init = 12       # Phase 1: Only environment, obstacles, start, and goal
    f_lines = 28      # Phase 2: Drawing vertical trapezoidal decomposition lines + cell labels
    f_graph = 20      # Phase 3: Building the adjacency graph on the right
    f_path = 30       # Phase 4: Highlighting graph nodes and drawing the red path
    total_frames = f_init + f_lines + f_graph + f_path

    def update(frame_idx):
        ax_env.clear()
        ax_graph.clear()

        ax_env.set_xlim(0, 11.3)
        ax_env.set_ylim(0, 10.2)
        ax_env.set_aspect('equal')
        ax_env.axis('off')

        ax_graph.set_xlim(0, 11.6)
        ax_graph.set_ylim(0, 9.5)
        ax_graph.set_aspect('equal')
        ax_graph.axis('off')

        # --- LEFT PANEL: ENVIRONMENT & OBSTACLES ---
        outer_patch = patches.Polygon(
            outer_coords, closed=True, fill=False, edgecolor='black', linewidth=2.2, zorder=3
        )
        ax_env.add_patch(outer_patch)

        for obs_coords in [obs_c_coords, obs_mid_coords]:
            obs_patch = patches.Polygon(
                obs_coords, closed=True, facecolor='#A6A6A6', edgecolor='black', linewidth=2.0, zorder=3
            )
            ax_env.add_patch(obs_patch)

        # Phase 2+: Draw vertical trapezoidal decomposition segments
        if frame_idx >= f_init:
            line_progress = min(1.0, (frame_idx - f_init + 1) / (f_lines * 0.75))
            num_lines = int(line_progress * len(vertical_segments))

            for p1, p2 in vertical_segments[:num_lines]:
                ax_env.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=0.9, zorder=2)

            # Display cell labels c1..c19 once decomposition lines are complete
            if line_progress >= 1.0:
                for label, (lx, ly) in cell_label_pos.items():
                    ax_env.text(lx, ly, label, fontsize=9.5, ha='center', va='center', color='black', zorder=4)

        # Always draw Start and Goal points in the environment
        ax_env.scatter(*start_point, c='black', s=65, zorder=6)
        ax_env.text(start_point[0] + 0.05, start_point[1] + 0.35, 'start', fontsize=10, ha='center', va='bottom')

        ax_env.scatter(*goal_point, c='black', s=65, zorder=6)
        ax_env.text(goal_point[0] + 0.1, goal_point[1] - 0.35, 'goal', fontsize=10, ha='center', va='top')

        # --- RIGHT PANEL: ADJACENCY GRAPH ---
        if frame_idx >= f_init + f_lines:
            graph_progress = min(1.0, (frame_idx - (f_init + f_lines) + 1) / (f_graph * 0.8))
            num_g_edges = int(graph_progress * len(adj_edges))

            # Draw adjacency edges
            for u, v in adj_edges[:num_g_edges]:
                p1, p2 = graph_pos[u], graph_pos[v]
                ax_graph.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=0.9, zorder=1)

            # Determine how many path steps are active in Phase 4
            active_path_nodes = set()
            if frame_idx >= f_init + f_lines + f_graph:
                path_progress = min(1.0, (frame_idx - (f_init + f_lines + f_graph) + 1) / (f_path * 0.75))
                num_active_nodes = max(1, int(np.ceil(path_progress * len(path_cells))))
                active_path_nodes = set(path_cells[:num_active_nodes])

                # Draw red trajectory in the environment (left panel)
                num_wp = max(2, int(np.ceil(path_progress * len(waypoints))))
                wx, wy = zip(*waypoints[:num_wp])
                ax_env.plot(wx, wy, color='#E60026', linewidth=2.2, zorder=5)

            # Draw graph nodes (bold border for nodes on the shortest path as in Slide 37)
            for node, (nx_p, ny_p) in graph_pos.items():
                is_highlighted = node in active_path_nodes
                circle = patches.Circle(
                    (nx_p, ny_p), radius=0.45, facecolor='white',
                    edgecolor='black', linewidth=2.8 if is_highlighted else 0.9, zorder=3
                )
                ax_graph.add_patch(circle)
                ax_graph.text(nx_p, ny_p, node, fontsize=9.5, ha='center', va='center', zorder=4)

            # Draw 'start' and 'goal' labels above c1 and c6 in the graph
            ax_graph.text(graph_pos['c1'][0], graph_pos['c1'][1] + 0.65, 'start', fontsize=10.5, ha='center', va='bottom')
            ax_graph.text(graph_pos['c6'][0], graph_pos['c6'][1] + 0.65, 'goal', fontsize=10.5, ha='center', va='bottom')

        # Titles for each stage
        if frame_idx < f_init:
            fig.suptitle("Trapezoidal Cell Decomposition: Phase 1 - Initial Environment (Start & Goal)", fontsize=13, fontweight='bold')
        elif frame_idx < f_init + f_lines:
            fig.suptitle("Trapezoidal Cell Decomposition: Phase 2 - Vertical Line Sweep & Cell Partitioning", fontsize=13, fontweight='bold')
        elif frame_idx < f_init + f_lines + f_graph:
            fig.suptitle("Trapezoidal Cell Decomposition: Phase 3 - Adjacency Graph Construction", fontsize=13, fontweight='bold')
        else:
            fig.suptitle("Trapezoidal Cell Decomposition: Phase 4 - Graph Search & Path Planning (Slide 37)", fontsize=13, fontweight='bold')

    print("Rendering animation frames...")
    anim = FuncAnimation(fig, update, frames=total_frames, interval=110)
    anim.save(output_filename, writer=PillowWriter(fps=9))
    plt.close(fig)
    print(f"GIF successfully saved as '{output_filename}'!")

if __name__ == "__main__":
    generate_cell_decomposition_gif()