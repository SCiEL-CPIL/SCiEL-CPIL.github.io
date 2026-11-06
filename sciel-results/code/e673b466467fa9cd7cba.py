import time
import random
import numpy as np

def _has_k222_count_np(A, n):
    """
    Counts the number of K2,2,2 subgraphs in the graph defined by adjacency matrix A.
    A is an n x n matrix (0/1).
    """
    count = 0
    # A K2,2,2 consists of two parts of size 2: {c,d} and {a,b} and {e,f}? No.
    # K2,2,2 has parts P1={u1,u2}, P2={v1,v2}, P3={w1,w2}.
    # Edges exist between all pairs of parts.
    # So for any pair in P1 (c,d), they must be connected to all 4 in P2 U P3.
    # Let's iterate over pairs (c,d) as one part.
    # Common neighbors N(c) & N(d).
    # We need to select 4 vertices from common neighbors: {a,b} and {e,f}.
    # Such that a,b are connected to e,f.
    # Actually, the condition is:
    # There exist 4 distinct vertices x,y,z,w in N(c) & N(d)
    # such that the subgraph induced by {x,y,z,w} contains a K2,2.
    # Wait, the edges between {x,y} and {z,w} must exist.
    # So x~z, x~w, y~z, y~w.
    # This is equivalent to:
    # For some partition of 4 common neighbors into two pairs {x,y} and {z,w},
    # A[x,z]=A[x,w]=A[y,z]=A[y,w]=1.
    
    # Efficient check:
    # For each pair (c,d):
    #   common = A[c] & A[d]
    #   Let M be the submatrix of A induced by vertices in common.
    #   We need to check if M contains a K2,2.
    #   A K2,2 in M exists if there are two vertices x,y in common such that
    #   |N_M(x) & N_M(y)| >= 2.
    
    # Let's implement this.
    for c in range(n):
        for d in range(c + 1, n):
            if A[c, d] == 0:
                continue
            common = A[c] & A[d]
            # Get indices of common neighbors
            common_idx = np.nonzero(common)[0]
            m = len(common_idx)
            if m < 4:
                continue
            
            # Extract submatrix
            # We need to check if there is a K2,2 in A[common_idx, common_idx]
            # K2,2 exists if there are x, y such that they share at least 2 common neighbors in common_idx.
            # Let's iterate over pairs in common_idx
            # To speed up, we can use matrix multiplication?
            # Or just loop since m is small (max 27).
            # m=27 -> 27*26/2 = 351 pairs.
            # For each pair, count common neighbors.
            
            # Optimization: Precompute bitsets for common neighbors?
            # Or use numpy.
            # Let's try a vectorized approach for the inner loop.
            # For each i in common_idx:
            #   neighbors_in_common = A[common_idx, common_idx][:, i]
            #   We need to find if any other j has dot product with neighbors_in_common >= 2.
            
            # Actually, simpler:
            # For each pair (i, j) in common_idx:
            #   shared = A[common_idx, i] & A[common_idx, j]
            #   if shared.sum() >= 2: return violation (or count)
            
            # Let's just do it simply first, optimize if needed.
            found = False
            for i in range(m):
                ci = common_idx[i]
                # Neighbors of ci within common_idx
                # A[common_idx, ci] is a boolean vector
                # We need to check if there is a pair (j, k) in common_idx
                # such that A[ci, cj]=1, A[ci, ck]=1, A[cj, ck]=1? No.
                
                # Let's stick to the pair (x,y) in common.
                # x, y are in common.
                # They must share >= 2 neighbors in common.
                # Let's iterate x, y.
                for j in range(i + 1, m):
                    cj = common_idx[j]
                    # Check if ci and cj share >= 2 neighbors in common
                    # Neighbors of ci in common: A[common_idx, ci]
                    # Neighbors of cj in common: A[common_idx, cj]
                    # Intersection:
                    # shared_count = np.sum(A[common_idx, ci] & A[common_idx, cj])
                    # Note: A[ci, cj] is 1, so they are neighbors.
                    # But we need neighbors *in common*.
                    # A[common_idx, ci] gives connections to all in common_idx.
                    # So shared_count is the number of vertices in common_idx connected to both ci and cj.
                    # If shared_count >= 2, we have a K2,2: {ci, cj} and {k, l}.
                    # Wait, the K2,2 parts are {ci, cj} and {k, l}.
                    # Edges: ci-k, ci-l, cj-k, cj-l.
                    # Yes, this is exactly it.
                    
                    # Optimization:
                    # Instead of summing every time, we can precompute?
                    # Or just do it.
                    # m is small.
                    # Let's use bit operations if possible?
                    # A is a numpy array.
                    # Let's convert A to a list of integers for bit operations?
                    # That might be faster.
                    
                    # For now, let's try the numpy way and see if it's fast enough.
                    # shared = A[common_idx, ci] & A[common_idx, cj]
                    # if shared.sum() >= 2:
                    #    count += 1
                    #    found = True
                    #    break
                    # if found: break
                    
                    # This is O(m^2) per pair (c,d).
                    # Total O(n^2 * m^2).
                    # n=29, m~14 avg. 29^2 * 14^2 ~ 160k ops. Very fast.
                    
                    shared = A[common_idx, ci] & A[common_idx, cj]
                    if shared.sum() >= 2:
                        count += 1
                        # We can break early if we just want validity, but we need count for SA.
                        # If we want to speed up SA, we can stop counting after a threshold?
                        # No, delta needs accurate count.
                        # But if count is huge, we can approximate?
                        # Let's stick to exact for now.
                        # Optimization: if count is already large, maybe we can stop?
                        # No, we need the difference.
                        
                        # Actually, we can break out of the inner loops if we just need to know IF it exists.
                        # But for SA cost, we need the number.
                        # However, if we find one, we know it's invalid.
                        # Maybe a separate function for validity check that breaks early?
                        # Yes.
                        
                        # For now, let's just count.
                        pass
    return count

def _is_valid_fast_np(A, n):
    """
    Checks if A contains NO K2,2,2.
    Breaks early on first violation.
    """
    for c in range(n):
        for d in range(c + 1, n):
            if A[c, d] == 0:
                continue
            common = A[c] & A[d]
            common_idx = np.nonzero(common)[0]
            m = len(common_idx)
            if m < 4:
                continue
            
            for i in range(m):
                ci = common_idx[i]
                for j in range(i + 1, m):
                    cj = common_idx[j]
                    shared = A[common_idx, ci] & A[common_idx, cj]
                    if shared.sum() >= 2:
                        return False
    return True

def _build_adj_np(colors, n):
    A = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                A[i, j] = 1
                A[j, i] = 1
            idx += 1
    return A

def _get_color_list_np(A, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(int(A[i, j]))
    return colors

def _find_violation_edges_np(A, n):
    """
    Returns a list of (i, j) edges that are part of at least one K2,2,2.
    """
    violating = set()
    for c in range(n):
        for d in range(c + 1, n):
            if A[c, d] == 0:
                continue
            common = A[c] & A[d]
            common_idx = np.nonzero(common)[0]
            m = len(common_idx)
            if m < 4:
                continue
            
            for i in range(m):
                ci = common_idx[i]
                for j in range(i + 1, m):
                    cj = common_idx[j]
                    shared = A[common_idx, ci] & A[common_idx, cj]
                    if shared.sum() >= 2:
                        # Violation found involving c, d, ci, cj
                        # The K2,2,2 is {c,d}, {ci,cj}, {k,l} where k,l are in shared.
                        # Edges in the subgraph:
                        # (c,d), (c,ci), (c,cj), (d,ci), (d,cj)
                        # (ci,cj), (ci,k), (ci,l), (cj,k), (cj,l)
                        # (k,l)
                        
                        violating.add((c, d))
                        violating.add((c, ci))
                        violating.add((c, cj))
                        violating.add((d, ci))
                        violating.add((d, cj))
                        violating.add((ci, cj))
                        
                        # Add edges to k, l
                        shared_idx = common_idx[shared.nonzero()[0]]
                        for k in shared_idx:
                            violating.add((ci, k))
                            violating.add((cj, k))
                            for l in shared_idx[shared_idx > k]:
                                violating.add((k, l))
                                violating.add((ci, l)) # Already added?
                                violating.add((cj, l))
                                # Also (k,l)
                                # And (c,k), (d,k)?
                                # c and d are connected to ALL common neighbors.
                                # So (c,k) and (d,k) are also edges in the K2,2,2.
                                violating.add((c, k))
                                violating.add((d, k))
                                violating.add((c, l))
                                violating.add((d, l))
                                
    return list(violating)

def _simulated_annealing_np(colors, n, time_limit=30, initial_temp=1.0, cooling_rate=0.9999):
    start_time = time.time()
    A = _build_adj_np(colors, n)
    current_cost = _has_k222_count_np(A, n)
    
    if current_cost == 0:
        return colors, True
    
    best_cost = current_cost
    best_colors = colors[:]
    temp = initial_temp
    
    # Precompute edge mapping
    edge_to_ij = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_ij.append((i, j))
            
    ij_to_idx = {}
    for idx, (i, j) in enumerate(edge_to_ij):
        ij_to_idx[(i, j)] = idx
        ij_to_idx[(j, i)] = idx
        
    violation_edges = _find_violation_edges_np(A, n)
    if not violation_edges:
        return colors, True
        
    step_count = 0
    while time.time() - start_time < time_limit:
        step_count += 1
        
        if violation_edges and random.random() < 0.7:
            vi, vj = random.choice(violation_edges)
            idx = ij_to_idx.get((vi, vj))
            if idx is None:
                idx = ij_to_idx.get((vj, vi))
            if idx is None:
                idx = random.randint(0, len(colors) - 1)
        else:
            idx = random.randint(0, len(colors) - 1)
            
        i, j = edge_to_ij[idx]
        
        old_val = A[i, j]
        A[i, j] = 1 - old_val
        A[j, i] = 1 - old_val
        colors[idx] = 1 - old_val
        
        new_cost = _has_k222_count_np(A, n)
        delta = new_cost - current_cost
        
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = colors[:]
                if best_cost == 0:
                    return best_colors, True
            if step_count % 30 == 0 or new_cost == 0:
                violation_edges = _find_violation_edges_np(A, n)
        else:
            A[i, j] = old_val
            A[j, i] = old_val
            colors[idx] = old_val
            
        temp *= cooling_rate
        if temp < 1e-10:
            temp = 1e-10
            
    return best_colors, best_cost == 0

def _paley(n, p):
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def construct():
    deadline = time.time() + 140
    best_n = 0
    best_colors = None
    
    target_n = 29
    p = 29
    
    # 1. Try Paley(29)
    colors = _paley(target_n, p)
    if colors:
        A = _build_adj_np(colors, target_n)
        if _is_valid_fast_np(A, target_n):
            return {'n': target_n, 'colors': colors}
            
    # 2. Try Simulated Annealing from Paley
    colors_copy = colors[:]
    _, success = _simulated_annealing_np(colors_copy, target_n, time_limit=50, initial_temp=1.5, cooling_rate=0.9999)
    if success:
        return {'n': target_n, 'colors': colors_copy}
        
    # 3. Random Restarts
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        colors = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        remaining = deadline - 10 - time.time()
        tl = max(3, min(8, remaining / max(1, 30 - attempt)))
        _, success = _simulated_annealing_np(colors, target_n, time_limit=tl, initial_temp=0.8, cooling_rate=0.9999)
        if success:
            return {'n': target_n, 'colors': colors}
            
    # Fallback
    for n in range(28, 10, -1):
        if time.time() > deadline - 5:
            break
        p_candidate = None
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                p_candidate = pp
                break
        if not p_candidate:
            continue
        colors = _paley(n, p_candidate)
        if colors:
            A = _build_adj_np(colors, n)
            if _is_valid_fast_np(A, n):
                return {'n': n, 'colors': colors}
                
    return {'n': 1, 'colors': [0]}
