import numpy as np
import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    # We want to maximize the number of points in [0, N)^2 such that no three
    # form an isosceles triangle.
    #
    # Key insight: if we place points on a line (y=0), the condition reduces to:
    # for any three points a, b, c on the line, |a-b| != |a-c| and |a-b| != |b-c|
    # and |a-c| != |b-c|. This is equivalent to all pairwise distances being distinct.
    # Such a set is called a "Golomb ruler" or "Sidon set" in the additive sense.
    #
    # However, we can do better by using 2D. The condition in 2D is:
    # For any three distinct points p, q, r, the triangle pqr is not isosceles.
    # That is, for any three points, no two of the three pairwise distances are equal.
    #
    # Strategy: Use a greedy approach with random restarts and local search.
    # 1. Start with a random set of points.
    # 2. Try to add more points greedily.
    # 3. Use local search to improve.
    #
    # A faster check: when adding a new point p to an existing set S, we need to check
    # that for every pair (a, b) in S, the distances d(p,a), d(p,b), d(a,b) are all distinct.
    # This is O(|S|^2) per candidate point.
    #
    # We can speed this up by precomputing all pairwise squared distances in S.
    # When adding p, we compute d(p,a) for all a in S. Then for each pair (a,b) in S,
    # we check if d(p,a) == d(p,b) or d(p,a) == d(a,b) or d(p,b) == d(a,b).
    #
    # Actually, a simpler check: for the new point p and existing set S:
    # - For each a in S, d(p,a) must not equal d(p,b) for any b != a in S.
    # - For each a in S, d(p,a) must not equal d(a,b) for any b != a in S.
    #
    # Let's implement a vectorized check for adding a point.

    def can_add(p, points, dist_matrix):
        """Check if point p can be added to the set of points."""
        if not points:
            return True
        # Compute distances from p to all existing points
        dx = points[:, 0] - p[0]
        dy = points[:, 1] - p[1]
        d2 = dx * dx + dy * dy  # squared distances
        
        # Check 1: all d2 values must be distinct (no two existing points at same distance from p)
        if len(np.unique(d2)) < len(d2):
            return False
        
        # Check 2: for each existing point a, d(p,a) != d(a,b) for any b != a
        # This means d2[i] != dist_matrix[i, j] for all j != i
        # We can check: for each i, d2[i] should not be in dist_matrix[i, :] except at j=i (which is 0)
        # Actually, dist_matrix[i,i] = 0, and d2[i] > 0, so we just need d2[i] not in dist_matrix[i, :]
        for i in range(len(points)):
            # Check if d2[i] appears in dist_matrix[i, :] (excluding the diagonal which is 0)
            if d2[i] in dist_matrix[i, :]:
                return False
        return True

    def compute_dist_matrix(points):
        """Compute pairwise squared distance matrix."""
        if len(points) == 0:
            return np.array([])
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        return np.sum(diff * diff, axis=2)

    def greedy_add(points, dist_matrix):
        """Try to add more points greedily."""
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            # Try adding points from a random subset of the grid
            candidates = np.random.choice(N * N, size=min(100, N * N), replace=False)
            for idx in candidates:
                x, y = divmod(idx, N)
                p = np.array([x, y])
                # Check if p is already in points
                if any(np.all(p == q) for q in points):
                    continue
                if can_add(p, points, dist_matrix):
                    points = np.vstack([points, p]) if len(points) > 0 else p.reshape(1, 2)
                    dist_matrix = compute_dist_matrix(points)
                    improved = True
                    break
        return points, dist_matrix

    def local_search(points, dist_matrix):
        """Try to improve by removing and re-adding points."""
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            # Try removing a random point and re-adding a different one
            for _ in range(50):
                if time.time() > deadline - 30:
                    break
                if len(points) < 2:
                    break
                idx = np.random.randint(len(points))
                old_point = points[idx]
                # Remove the point
                new_points = np.delete(points, idx, axis=0)
                new_dist = compute_dist_matrix(new_points)
                # Try to add a different point
                candidates = np.random.choice(N * N, size=min(50, N * N), replace=False)
                for cidx in candidates:
                    x, y = divmod(cidx, N)
                    p = np.array([x, y])
                    if any(np.all(p == q) for q in new_points):
                        continue
                    if can_add(p, new_points, new_dist):
                        points = np.vstack([new_points, p]) if len(new_points) > 0 else p.reshape(1, 2)
                        dist_matrix = compute_dist_matrix(points)
                        improved = True
                        break
                if improved:
                    break
        return points, dist_matrix

    best_points = None
    best_count = 0

    # Multiple random restarts
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        # Random initialization: pick a random subset of points
        # Start with a small random set
        k = random.randint(5, 15)
        candidates = np.random.choice(N * N, size=k, replace=False)
        points = []
        dist_matrix = np.array([])
        for idx in candidates:
            x, y = divmod(idx, N)
            p = np.array([x, y])
            if can_add(p, np.array(points) if points else np.empty((0, 2)), dist_matrix if len(dist_matrix) > 0 else np.array([])):
                points.append(p)
                if len(points) > 1:
                    dist_matrix = compute_dist_matrix(np.array(points))
        
        points = np.array(points) if points else np.empty((0, 2))
        if len(points) > 0:
            dist_matrix = compute_dist_matrix(points)
        
        # Greedy expansion
        points, dist_matrix = greedy_add(points, dist_matrix)
        
        # Local search
        points, dist_matrix = local_search(points, dist_matrix)
        
        if len(points) > best_count:
            best_count = len(points)
            best_points = points.copy()
            print(f"Restart {restart}: improved to {best_count} points")

    if best_points is not None and len(best_points) > 0:
        return [(int(x), int(y)) for x, y in best_points]
    return []
# EVOLVE-BLOCK-END
