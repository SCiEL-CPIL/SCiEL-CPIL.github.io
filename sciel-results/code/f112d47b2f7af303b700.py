import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

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

    # Hex distance from origin
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Generate all points in a box
    # Radius 3 hexagon fits in |a|,|b|,|a+b| <= 3, so a,b in [-3,3] suffices
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
    # Start with inner (19 points), then add 13 outer points to maximize edges

    # To maximize edges among selected outer points, consecutive points along
    # the hex perimeter share an edge. So pick a contiguous block of 13.
    # Order outer points by angle to get perimeter order.
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    outer_sorted = sorted(outer, key=angle)

    # Evaluate all contiguous blocks of 13 (18 possible, wrap-around)
    best_set = None
    best_edges = -1
    n_out = len(outer_sorted)
    k_needed = 32 - len(inner)  # 13

    for start in range(n_out):
        block = [outer_sorted[(start + i) % n_out] for i in range(k_needed)]
        candidate = inner + block
        edges = count_edges(candidate)
        if edges > best_edges:
            best_edges = edges
            best_set = candidate

    # The theoretical maximum is 76.
    # If we didn't reach 76, we could try a local search, but the heuristic is known to work.
    # For robustness, if best_edges < 76, we perform a quick local search.
    if best_edges < 76 and time.time() < deadline - 10.0:
        # Simple local search: swap an outer point with an unselected outer point
        selected_set = set(tuple(p) for p in best_set)
        unselected_outer = [p for p in outer if tuple(p) not in selected_set]
        
        improved = True
        current_set = [list(p) for p in selected_set]
        current_edges = best_edges
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(current_set)):
                for j in range(len(unselected_outer)):
                    new_set = current_set.copy()
                    new_set[i] = unselected_outer[j]
                    # Check for duplicates
                    if len(set(tuple(p) for p in new_set)) < 32:
                        continue
                    edges = count_edges(new_set)
                    if edges > current_edges:
                        current_set = new_set
                        current_edges = edges
                        # Update selected/unselected
                        selected_set = set(tuple(p) for p in current_set)
                        unselected_outer = [p for p in outer if tuple(p) not in selected_set]
                        improved = True
                        break
                if improved:
                    break
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_set = current_set

    # Verify we have exactly 32 distinct points
    result = best_set
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64

    if time.time() > deadline:
        return result

    return result
