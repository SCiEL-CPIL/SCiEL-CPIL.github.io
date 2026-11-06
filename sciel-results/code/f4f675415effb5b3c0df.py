import time
import math

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    # Distance squared on triangular lattice: (a,b) represents a + b*omega, omega=exp(i*pi/3)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    # Hex distance from origin
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Generate all points in a hexagon of radius 3
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

    # Build index mapping
    point_to_idx = {tuple(p): i for i, p in enumerate(all_points)}
    n_pts = len(all_points)

    # Precompute adjacency using bitmasks for fast edge counting
    # adj_mask[i] is a bitmask where bit j is set if points i and j are adjacent
    adj_mask = [0] * n_pts
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            if dist_sq(all_points[i], all_points[j]) == 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    def count_edges_fast(indices):
        """Count edges given indices into all_points using bitmask AND."""
        # Build bitmask for the selected set
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        count = 0
        for idx in indices:
            # Count neighbors that are also in the selected set
            neighbors = adj_mask[idx] & mask
            count += bin(neighbors).count('1')
        return count // 2

    # Order outer points by angle to get perimeter order
    def angle(p):
        a, b = p
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    outer_sorted = sorted(outer, key=angle)
    n_out = len(outer_sorted)
    k_needed = 32 - len(inner)  # 13

    inner_indices = [point_to_idx[tuple(p)] for p in inner]

    # Evaluate all contiguous blocks of 13 (18 possible, wrap-around)
    best_set = None
    best_edges = -1

    for start in range(n_out):
        block = [outer_sorted[(start + i) % n_out] for i in range(k_needed)]
        block_indices = [point_to_idx[tuple(p)] for p in block]
        all_indices = inner_indices + block_indices
        edges = count_edges_fast(all_indices)
        if edges > best_edges:
            best_edges = edges
            best_set = inner + block

    # The theoretical maximum is 76 (Harborth's theorem for n=32)
    if best_edges < 76 and time.time() < deadline - 5.0:
        # Local search: try swapping outer points to improve
        selected_set = set(tuple(p) for p in best_set)
        unselected_outer = [p for p in outer if tuple(p) not in selected_set]

        improved = True
        current_set = [list(p) for p in selected_set]
        current_edges = best_edges

        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(current_set)):
                if tuple(current_set[i]) == (0, 0):
                    continue  # Don't swap the center
                for j in range(len(unselected_outer)):
                    new_set = current_set.copy()
                    new_set[i] = unselected_outer[j]
                    if len(set(tuple(p) for p in new_set)) < 32:
                        continue
                    new_indices = [point_to_idx[tuple(p)] for p in new_set]
                    edges = count_edges_fast(new_indices)
                    if edges > current_edges:
                        current_set = new_set
                        current_edges = edges
                        selected_set = set(tuple(p) for p in current_set)
                        unselected_outer = [p for p in outer if tuple(p) not in selected_set]
                        improved = True
                        break
                if improved:
                    break

        if current_edges > best_edges:
            best_edges = current_edges
            best_set = current_set

    # Final polish: try 2-swap moves if still below 76
    if best_edges < 76 and time.time() < deadline - 2.0:
        current_set = [list(p) for p in best_set]
        current_edges = best_edges
        selected_set = set(tuple(p) for p in current_set)
        unselected_outer = [p for p in outer if tuple(p) not in selected_set]

        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    for k in range(len(unselected_outer)):
                        for l in range(k+1, len(unselected_outer)):
                            new_set = current_set.copy()
                            new_set[i] = unselected_outer[k]
                            new_set[j] = unselected_outer[l]
                            if len(set(tuple(p) for p in new_set)) < 32:
                                continue
                            new_indices = [point_to_idx[tuple(p)] for p in new_set]
                            edges = count_edges_fast(new_indices)
                            if edges > current_edges:
                                current_set = new_set
                                current_edges = edges
                                selected_set = set(tuple(p) for p in current_set)
                                unselected_outer = [p for p in outer if tuple(p) not in selected_set]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if current_edges > best_edges:
            best_edges = current_edges
            best_set = current_set

    result = best_set
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64

    return result
