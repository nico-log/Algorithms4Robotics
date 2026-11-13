import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from shapely.geometry import Polygon, Point, LineString
import networkx as nx

def prob_methods(output_filename="prob_methods.gif"):
    # Set seed to ensure sampling reproducibility and guarantee a valid path
    np.random.seed(42)

    # 1. ENVIRONMENT DEFINITION (based on slides on pages 2 and 5)
    obstacles_coords = [
        # Top-left obstacle
        [(1.4, 8.2), (3.5, 8.4), (3.1, 6.2), (2.5, 5.7), (2.3, 6.8)],
        # Bottom-left obstacle (concave polygon)
        [(1.9, 4.6), (2.7, 3.3), (2.2, 2.5), (1.3, 2.9), (1.5, 1.9), (2.5, 1.2), (3.3, 2.7), (2.7, 3.6)],
        # Center obstacle
        [(4.4, 5.2), (5.0, 6.0), (6.0, 4.4), (5.4, 3.8), (4.3, 3.7)],
        # Top-right obstacle
        [(7.0, 7.8), (7.6, 8.7), (9.0, 7.2), (8.1, 5.5), (6.9, 6.4)],
        # Bottom-right obstacle
        [(7.3, 4.0), (8.7, 3.7), (9.0, 2.0), (8.4, 0.7), (7.2, 1.8)]
    ]
    
    obstacles = [Polygon(coords) for coords in obstacles_coords]

    # Start and Goal points
    start_point = (5.6, 9.0)
    goal_point = (4.4, 0.8)

    # 2. LEARNING PHASE: FREE CONFIGURATION SPACE SAMPLING (Q_free)
    num_samples = 42
    k_neighbors = 5
    max_edge_distance = 3.8

    sampled_points = []
    while len(sampled_points) < num_samples:
        x = np.random.uniform(0.4, 9.6)
        y = np.random.uniform(0.5, 9.5)
        candidate_point = Point(x, y)
        # Keep the point only if it lies in the collision-free space (with a safety margin)
        if all(not obstacle.buffer(0.15).contains(candidate_point) for obstacle in obstacles):
            sampled_points.append((x, y))

    def is_collision_free(point_a, point_b):
        segment = LineString([point_a, point_b])
        return all(not obstacle.buffer(0.05).intersects(segment) for obstacle in obstacles)

    # Graph construction (Roadmap M)
    roadmap_graph = nx.Graph()
    for node_id, coords in enumerate(sampled_points):
        roadmap_graph.add_node(node_id, pos=coords)

    roadmap_edges = []
    for i, point_a in enumerate(sampled_points):
        # Compute Euclidean distance to all other nodes to find the k-nearest neighbors (V_k)
        distances = [
            (j, np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))
            for j, point_b in enumerate(sampled_points) if i != j
        ]
        distances.sort(key=lambda item: item[1])
        
        for j, dist in distances[:k_neighbors]:
            if dist <= max_edge_distance and not roadmap_graph.has_edge(i, j):
                point_b = sampled_points[j]
                if is_collision_free(point_a, point_b):
                    roadmap_graph.add_edge(i, j, weight=dist)
                    roadmap_edges.append((point_a, point_b))

    # 3. QUERY PHASE: CONNECTING START/GOAL AND PATH SEARCH (A*)
    roadmap_graph.add_node('start', pos=start_point)
    roadmap_graph.add_node('goal', pos=goal_point)
    query_edges = []

    for label, query_point in [('start', start_point), ('goal', goal_point)]:
        distances = [
            (i, np.hypot(query_point[0] - point[0], query_point[1] - point[1]))
            for i, point in enumerate(sampled_points)
        ]
        distances.sort(key=lambda item: item[1])
        connected_count = 0
        for i, dist in distances:
            if connected_count >= 4:  # Connect to the k=4 closest collision-free nodes
                break
            if is_collision_free(query_point, sampled_points[i]):
                roadmap_graph.add_edge(label, i, weight=dist)
                query_edges.append((query_point, sampled_points[i]))
                connected_count += 1

    # Shortest path search using A* (Euclidean distance heuristic)
    node_positions = nx.get_node_attributes(roadmap_graph, 'pos')
    def euclidean_heuristic(node_u, node_v):
        pos_u, pos_v = node_positions[node_u], node_positions[node_v]
        return np.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1])

    shortest_path_nodes = nx.astar_path(
        roadmap_graph, 'start', 'goal', heuristic=euclidean_heuristic, weight='weight'
    )
    shortest_path_coords = [node_positions[node] for node in shortest_path_nodes]

    # 4. PLOT CONFIGURATION AND GIF GENERATION
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    
    # Frame allocation across the 4 phases
    frames_init = 12       # Phase 1: Environment, obstacles, start, and goal only
    frames_sample = 25     # Phase 2: Progressive sampling of nodes in Q_free
    frames_connect = 35    # Phase 3: Graph connections (Roadmap + Query edges)
    frames_path = 25       # Phase 4: Shortest path discovery and rendering
    total_frames = frames_init + frames_sample + frames_connect + frames_path

    def update_frame(frame_idx):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw obstacles (yellow polygons with black edges matching the slides)
        for coords in obstacles_coords:
            polygon_patch = patches.Polygon(
                coords, closed=True, facecolor='#F7C815', edgecolor='black', linewidth=1.3, zorder=2
            )
            ax.add_patch(polygon_patch)

        # Phase 3: Draw Roadmap edges and Start/Goal query connections
        if frame_idx >= frames_init + frames_sample:
            progress = min(1.0, (frame_idx - (frames_init + frames_sample) + 1) / (frames_connect * 0.8))
            num_edges_to_draw = int(progress * len(roadmap_edges))
            
            for point_a, point_b in roadmap_edges[:num_edges_to_draw]:
                ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], color='black', linewidth=1.0, zorder=3)
            
            # Draw red dashed query edges (accessibility / departability)
            if progress >= 1.0:
                for point_a, point_b in query_edges:
                    ax.plot(
                        [point_a[0], point_b[0]], [point_a[1], point_b[1]],
                        color='#D9534F', linestyle='--', linewidth=1.2, zorder=3
                    )

        # Phase 2: Draw sampled nodes (white circles with black borders)
        if frame_idx >= frames_init:
            progress = min(1.0, (frame_idx - frames_init + 1) / frames_sample)
            num_points_to_draw = int(progress * len(sampled_points))
            if num_points_to_draw > 0:
                points_array = np.array(sampled_points[:num_points_to_draw])
                ax.scatter(
                    points_array[:, 0], points_array[:, 1],
                    c='white', edgecolors='black', s=30, linewidth=1.2, zorder=4
                )

        # Phase 4: Highlight the shortest path in red
        if frame_idx >= frames_init + frames_sample + frames_connect:
            progress = min(
                1.0, (frame_idx - (frames_init + frames_sample + frames_connect) + 1) / (frames_path * 0.6)
            )
            num_segments = max(1, int(progress * (len(shortest_path_coords) - 1)))
            for i in range(num_segments):
                point_a, point_b = shortest_path_coords[i], shortest_path_coords[i + 1]
                ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], color='red', linewidth=3.0, zorder=5)

        # Always draw Start and Goal points (blue markers with text labels)
        ax.scatter(*start_point, c='blue', s=110, zorder=6)
        ax.text(start_point[0], start_point[1] + 0.35, 'start', fontsize=12, ha='center', va='bottom', color='black')

        ax.scatter(*goal_point, c='blue', s=110, zorder=6)
        ax.text(goal_point[0], goal_point[1] - 0.35, 'goal', fontsize=12, ha='center', va='top', color='black')

    print("Generating Probabilistic RoadMap (PRM) animation...")
    animation = FuncAnimation(fig, update_frame, frames=total_frames, interval=100)
    animation.save(output_filename, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"GIF successfully saved as '{output_filename}'!")

if __name__ == "__main__":
    prob_methods()