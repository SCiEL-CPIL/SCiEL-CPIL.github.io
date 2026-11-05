import time
import itertools

def construct():
    deadline = time.time() + 140

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    # Generate all points in the radius-3 hexagon (37 points)
    R = 3
    all_points = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if abs(a+b) <= R:
                all_points.append((a, b))

    n = len(all_points)  # 37
    assert n == 37

    # Build adjacency matrix and degrees
    adj = [[0]*n for _ in range(n)]
    deg = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if dist_sq(all_points[i], all_points[j]) == 1:
                adj[i][j] = 1
                adj[j][i] = 1
                deg[i] += 1
                deg[j] += 1

    total_edges = sum(deg) // 2

    # We need to remove 5 points from 37 to get 32.
    # Maximize edges remaining = total_edges - (edges lost)
    # Edges lost = edges within removed set + edges between removed and kept
    # = sum(deg[i] for i in removed) - edges_within_removed
    # So minimize: sum(deg[i] for i in removed) - edges_within_removed

    best_cost = float('inf')
    best_removed = None

    remove_count = 5
    for combo in itertools.combinations(range(n), remove_count):
        if time.time() > deadline - 5:
            break

        # Compute cost = sum(deg[i]) - edges_within_combo
        cost = sum(deg[i] for i in combo)
        # Count edges within combo
        m = len(combo)
        for x in range(m):
            for y in range(x+1, m):
                cost -= adj[combo[x]][combo[y]]

        if cost < best_cost:
            best_cost = cost
            best_removed = combo

    if best_removed is None:
        # Fallback: just take first 32
        return [list(p) for p in all_points[:32]]

    # Build result: all points except those in best_removed
    removed_set = set(best_removed)
    result = [list(p) for i, p in enumerate(all_points) if i not in removed_set]

    # Verify: exactly 32 distinct points, |a|,|b| <= 64, mutual distance >= 1
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
    # Distance check: all pairs at distance >= 1 (i.e., dist_sq >= 1)
    for i in range(32):
        for j in range(i+1, 32):
            d = dist_sq(result[i], result[j])
            assert d >= 1, f"Points {result[i]} and {result[j]} too close: {d}"

    return result
