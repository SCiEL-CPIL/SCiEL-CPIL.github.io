import random
import time
import numpy as np

N = 64
TOTAL_POINTS = N * N

def construct():
    # Precompute points and distances
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Distance matrix
    # Using a list of lists of ints is fast enough for small K
    dist_matrix = [[0] * num_points for _ in range(num_points)]
    for i in range(num_points):
        xi, yi = points[i]
        for j in range(i + 1, num_points):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx * dx + dy * dy
            dist_matrix[i][j] = d2
            dist_matrix[j][i] = d2

    def count_conflicts(S):
        """
        Returns the number of isosceles triangles in S.
        S is a set of indices.
        """
        if len(S) < 3:
            return 0
        count = 0
        S_list = list(S)
        for p in S_list:
            dists = {}
            for q in S_list:
                if p == q:
                    continue
                d2 = dist_matrix[p][q]
                dists[d2] = dists.get(d2, 0) + 1
            for c in dists.values():
                if c >= 2:
                    count += c * (c - 1) // 2
        return count

    def find_conflict_point(S):
        """
        Returns a point in S that is part of at least one isosceles triangle, or None if S is valid.
        """
        if len(S) < 3:
            return None
        
        S_list = list(S)
        # Shuffle to ensure randomness in which conflict point is picked
        random.shuffle(S_list)
        
        for p in S_list:
            # Check vertex at p
            dists_p = {}
            for q in S_list:
                if p == q:
                    continue
                d2 = dist_matrix[p][q]
                dists_p[d2] = dists_p.get(d2, 0) + 1
            
            is_conflict = False
            for c in dists_p.values():
                if c >= 2:
                    is_conflict = True
                    break
            
            if is_conflict:
                return p
            
            # Check vertex at q in S \ {p}
            # For each q, check if dist(p,q) is in dist(q, S \ {q,p})
            for q in S_list:
                if p == q:
                    continue
                d_pq = dist_matrix[p][q]
                
                # Check if d_pq exists in distances from q to others
                # We can do this quickly by checking if any r in S \ {p,q} has dist(q,r) == d_pq
                # Optimization: Precompute dist sets? No, K is small.
                # Just loop
                for r in S_list:
                    if r == p or r == q:
                        continue
                    if dist_matrix[q][r] == d_pq:
                        is_conflict = True
                        break
                if is_conflict:
                    return p # p is involved in a conflict with vertex q
            
        return None

    best_set = set()
    deadline = time.time() + 295.0
    
    # Start with a small size and increase
    current_size = 5
    
    while time.time() < deadline:
        if current_size > len(best_set):
            # Try to find a valid set of current_size
            success = False
            local_deadline = time.time() + 5.0 # Limit time per size attempt
            
            # Initialize with random points
            S = set(random.sample(range(num_points), current_size))
            
            # Min-Conflicts loop
            while time.time() < local_deadline:
                conflicts = count_conflicts(S)
                if conflicts == 0:
                    success = True
                    break
                
                # Find a point to move
                p = find_conflict_point(S)
                if p is None:
                    # Should not happen if conflicts > 0, but safety
                    break
                
                # Pick a random new point
                # Ensure it's not in S
                while True:
                    q = random.randint(0, num_points - 1)
                    if q not in S:
                        break
                
                S.remove(p)
                S.add(q)
            
            if success:
                best_set = S
                current_size += 1
            else:
                # Failed to find a set of this size, try again or break?
                # Try a few times before giving up on this size
                # If we've spent too much time, break
                if time.time() > deadline - 10:
                    break
                # Retry with a new random start
                continue

    return [points[i] for i in best_set]
