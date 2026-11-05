import time
import random

def construct():
    deadline = time.time() + 140
    
    # 1. Helper functions
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    def count_edges(points_set):
        """Count edges in a set of points. O(N^2) but N is small (32)."""
        pts = list(points_set)
        cnt = 0
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if is_adjacent(pts[i], pts[j]):
                    cnt += 1
        return cnt

    # 2. Generate Candidate Pool
    # We need a pool large enough to contain an optimal solution.
    # Hexagon of radius 4 has 61 points. Optimal solution for n=32 is compact.
    # Let's use R=4 to be safe and fast.
    R = 4
    pool = []
    pool_set = set()
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if max(abs(a), abs(b), abs(a+b)) <= R:
                p = (a, b)
                pool.append(p)
                pool_set.add(p)
    
    # 3. Pre-compute adjacency in the pool
    # adj[i] is a set of indices j such that pool[i] and pool[j] are adjacent
    n_pool = len(pool)
    adj = [set() for _ in range(n_pool)]
    pool_index_map = {p: i for i, p in enumerate(pool)}
    
    for i in range(n_pool):
        p = pool[i]
        # Check 6 neighbors directly for speed
        for da, db in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
            nb = (p[0]+da, p[1]+db)
            if nb in pool_index_map:
                j = pool_index_map[nb]
                adj[i].add(j)
                
    # 4. Initial Construction: Hex-2 Core + Greedy Shell-3
    # This provides a strong starting point, likely close to optimal.
    
    # Hex-2 points
    hex2 = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                hex2.add((a, b))
                
    # Shell-3 points
    shell3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) == 3:
                shell3.append((a, b))
                
    # Greedy selection of 13 points from shell3 to maximize edges
    current_set = set(hex2)
    selected_shell = set()
    
    for _ in range(13):
        best_point = None
        best_gain = -1
        
        remaining = [p for p in shell3 if p not in selected_shell]
        
        for p in remaining:
            gain = 0
            # Count neighbors in current_set (which includes hex2 + previously selected shell)
            for q in current_set:
                if is_adjacent(p, q):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_point = p
                
        if best_point:
            selected_shell.add(best_point)
            current_set.add(best_point)
            
    # Ensure we have exactly 32 points
    if len(current_set) < 32:
        # Fallback: fill with random points from pool if greedy failed (unlikely)
        for p in pool:
            if len(current_set) >= 32:
                break
            if p not in current_set:
                current_set.add(p)
                
    # 5. Hill Climbing with Pre-computed Adjacency
    # We represent the current set as a set of indices into the pool for fast lookup
    # and to use the pre-computed adj lists.
    
    # Map current points to indices
    current_indices = set()
    for p in current_set:
        if p in pool_index_map:
            current_indices.add(pool_index_map[p])
        else:
            # This shouldn't happen if R is large enough, but handle gracefully
            pass
            
    # If some points are outside pool (shouldn't happen with R=4 and hex2/shell3), 
    # we'd need to add them to pool. But hex2/shell3 are inside R=3, so R=4 is safe.
    
    def get_score_from_indices(indices_set):
        """Calculate score using pre-computed adj lists. O(E) where E is edges."""
        score = 0
        # Iterate over unique pairs? No, just iterate over set and adj, divide by 2?
        # Or just count edges.
        # Since adj is symmetric, we can just sum degrees and divide by 2.
        deg_sum = 0
        for i in indices_set:
            deg_sum += len(adj[i] & indices_set)
        return deg_sum // 2

    current_score = get_score_from_indices(current_indices)
    
    # Local Search: Try all swaps
    # We want to replace one point in current_indices with one point not in current_indices.
    # Score change = -deg(i) + deg(new_i) - (1 if i and new_i are adjacent else 0)
    # Wait, if i and new_i are adjacent, that edge is removed when i is removed, 
    # and added when new_i is added? 
    # Let's derive carefully:
    # S = Set. i in S. j not in S.
    # S' = S - {i} + {j}
    # Edges in S: E(S)
    # Edges in S': E(S')
    # E(S') = E(S) - deg_S(i) + deg_{S-{i}}(j)
    # deg_{S-{i}}(j) = deg_S(j) - (1 if i~j else 0)
    # So Delta = deg_{S-{i}}(j) - deg_S(i)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_swap_delta = 0
        best_i = None
        best_j = None
        
        # To speed this up, we can iterate over i in current_indices and j in pool_indices \ current_indices
        # Pool size is ~61. Current set size is 32.
        # Pairs to check: 32 * (61-32) = 32 * 29 = 928.
        # This is very fast.
        
        for i in current_indices:
            deg_i = len(adj[i] & current_indices)
            
            for j in range(n_pool):
                if j in current_indices:
                    continue
                
                # Calculate deg of j in S-{i}
                # deg_{S-{i}}(j) = (number of neighbors of j in S) - (1 if i is neighbor of j else 0)
                neighbors_j_in_S = 0
                # We can use the adj list of j
                for k in adj[j]:
                    if k in current_indices:
                        neighbors_j_in_S += 1
                        
                if i in adj[j]:
                    neighbors_j_in_S -= 1
                    
                delta = neighbors_j_in_S - deg_i
                
                if delta > best_swap_delta:
                    best_swap_delta = delta
                    best_i = i
                    best_j = j
                    
        if best_i is not None:
            current_indices.remove(best_i)
            current_indices.add(best_j)
            current_score += best_swap_delta
            improved = True
            if current_score >= 76:
                # We reached the theoretical maximum, stop early
                break

    # 6. Convert back to list of pairs
    final_points = [pool[i] for i in current_indices]
    
    # Final verification
    if len(final_points) != 32:
        # Fallback to hardcoded optimal if something went wrong (should not happen)
        # This is a known optimal configuration for n=32 (Hexagon radius 3 minus 5 corners, or similar)
        # Actually, let's just return what we have, it should be 32.
        pass
        
    return [[a, b] for a, b in final_points]
