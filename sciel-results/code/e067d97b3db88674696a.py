import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    # Distance squared on triangular lattice: (a,b) represents a + b*omega, omega=exp(i*pi/3)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Hex distance from origin (axial coordinates)
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Generate all points in a box with hex distance <= 3
    all_points = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if hex_dist([a, b]) <= 3:
                all_points.append([a, b])

    # Inner: radius <= 2 (19 points)
    inner = [p for p in all_points if hex_dist(p) <= 2]
    # Outer: radius == 3 (18 points)
    outer = [p for p in all_points if hex_dist(p) == 3]

    assert len(inner) == 19, f"inner={len(inner)}"
    assert len(outer) == 18, f"outer={len(outer)}"

    # We need 32 total, so pick 13 from outer
    k_needed = 32 - len(inner)  # 13
    
    # Pre-compute edges within inner
    inner_edges = count_edges(inner)
    
    # Pre-compute adjacency: for each outer point, count edges to inner and to other outer points
    # outer_to_inner[i] = number of edges from outer[i] to inner points
    # outer_to_outer[i][j] = 1 if outer[i] and outer[j] are adjacent, else 0
    outer_to_inner = [0] * len(outer)
    outer_adj = [[0] * len(outer) for _ in range(len(outer))]
    
    for i, p in enumerate(outer):
        for q in inner:
            if dist_sq(p, q) == 1:
                outer_to_inner[i] += 1
        for j in range(i+1, len(outer)):
            if dist_sq(p, outer[j]) == 1:
                outer_adj[i][j] = 1
                outer_adj[j][i] = 1

    # Spiral ordering for outer points: sort by angle around origin
    # This helps prioritize combinations that follow the spiral pattern
    def angle_key(p):
        a, b = p
        # Convert to Cartesian for angle calculation
        # omega = exp(i*pi/3), so point a + b*omega has coords:
        # x = a + b*cos(pi/3) = a + b/2
        # y = b*sin(pi/3) = b*sqrt(3)/2
        x = a + b * 0.5
        y = b * (math.sqrt(3) / 2)
        return math.atan2(y, x)
    
    # Sort outer points by angle (spiral order)
    outer_sorted = sorted(outer, key=angle_key)
    # Create mapping from original outer indices to sorted indices
    outer_index_map = {id(p): i for i, p in enumerate(outer)}
    
    # Rebuild adjacency in sorted order
    outer_to_inner_sorted = [0] * len(outer_sorted)
    outer_adj_sorted = [[0] * len(outer_sorted) for _ in range(len(outer_sorted))]
    
    for i, p in enumerate(outer_sorted):
        orig_i = outer_index_map[id(p)]
        outer_to_inner_sorted[i] = outer_to_inner[orig_i]
        for j in range(i+1, len(outer_sorted)):
            orig_j = outer_index_map[id(outer_sorted[j])]
            outer_adj_sorted[i][j] = outer_adj[orig_i][orig_j]
            outer_adj_sorted[j][i] = outer_adj[orig_j][orig_i]

    # Exhaustive search over all subsets of 13 from the 18 outer points
    # C(18, 13) = 8568, which is small enough to enumerate fully
    best_set = None
    best_edges = -1
    
    # Pre-compute for each combination: sum of outer_to_inner + edges within combo
    # Use the sorted order for potential early termination, but still check all
    
    for combo_indices in combinations(range(len(outer_sorted)), k_needed):
        # Count edges within combo
        e_combo = 0
        for i_idx in range(len(combo_indices)):
            for j_idx in range(i_idx+1, len(combo_indices)):
                i = combo_indices[i_idx]
                j = combo_indices[j_idx]
                if outer_adj_sorted[i][j] == 1:
                    e_combo += 1
        
        # Count edges between combo and inner
        e_cross = sum(outer_to_inner_sorted[i] for i in combo_indices)
        
        total_edges = inner_edges + e_combo + e_cross
        
        if total_edges > best_edges:
            best_edges = total_edges
            best_set = inner + [outer_sorted[i] for i in combo_indices]
        
        # Check time deadline
        if time.time() > deadline:
            break

    # If we didn't find a solution due to timeout, use a greedy fallback
    if best_set is None:
        # Greedy: pick outer points one by one, each time choosing the one that
        # maximizes (edges to inner + edges to already-picked outer points)
        selected_outer = []
        remaining = list(range(len(outer_sorted)))
        
        for _ in range(k_needed):
            best_idx = -1
            best_score = -1
            for idx in remaining:
                # Score: edges to inner + edges to already selected outer
                score = outer_to_inner_sorted[idx]
                for sel in selected_outer:
                    if outer_adj_sorted[idx][sel] == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected_outer.append(best_idx)
            remaining.remove(best_idx)
        
        best_set = inner + [outer_sorted[i] for i in selected_outer]
        best_edges = count_edges(best_set)

    # Verify we have exactly 32 distinct points
    result = best_set
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64

    return result
