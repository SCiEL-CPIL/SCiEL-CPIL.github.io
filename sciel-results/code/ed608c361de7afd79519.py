import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin
    
    # Fallback solution n=11 (from seed)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback[:]

    # Precompute edge endpoints for each n up to 26
    # We'll build these on the fly as needed to save memory, but precompute for speed
    edge_cache = {}
    
    def get_edges(n):
        if n not in edge_cache:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append((i, j))
            edge_cache[n] = edges
        return edge_cache[n]

    def get_cost(colors, n):
        """
        Calculate the total number of C4s across all 5 color classes.
        A C4 in color c is determined by a pair of vertices (u, w) having >= 2 common neighbors in color c.
        Number of C4s = sum_{u < w} binom(common_neighbors(u, w), 2)
        """
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return float('inf')
        
        # Build adjacency matrices for each color
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        
        total_cost = 0.0
        for m in mats:
            M = m @ m
            diag_M = np.diag(M)
            sum_M_sq = np.sum(M**2)
            sum_M = np.sum(M)
            
            sum_diag_sq = np.sum(diag_M**2)
            sum_diag = np.sum(diag_M)
            
            sum_off_diag_sq = (sum_M_sq - sum_diag_sq) / 2.0
            sum_off_diag = (sum_M - sum_diag) / 2.0
            
            cost_c = (sum_off_diag_sq - sum_off_diag) / 2.0
            total_cost += cost_c
            
        return total_cost

    # Strategy: Use Simulated Annealing with efficient delta updates
    # This is a structural change from the current hill climbing with random restarts
    # SA can escape local minima and is better suited for this type of problem
    
    def solve_n_sa(n, time_limit):
        """
        Solve for n using Simulated Annealing with delta energy updates
        """
        if time.time() > deadline - 5:
            return None, None
            
        edges = get_edges(n)
        num_edges = len(edges)
        
        # Precompute edge indices for each vertex
        # For each vertex v, list of (edge_idx, other_vertex)
        vert_edges = [[] for _ in range(n)]
        for e_idx, (u, v) in enumerate(edges):
            vert_edges[u].append((e_idx, v))
            vert_edges[v].append((e_idx, u))
        
        # Random initialization
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Build initial adjacency and compute cost
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        for e_idx, (u, v) in enumerate(edges):
            c = colors[e_idx]
            mats[c][u, v] = 1
            mats[c][v, u] = 1
            
        # Precompute M = mats[c] @ mats[c] for each color
        Ms = [m @ m for m in mats]
        
        # Compute initial cost
        def compute_total_cost(Ms):
            total = 0.0
            for M in Ms:
                diag_M = np.diag(M)
                sum_M_sq = np.sum(M**2)
                sum_M = np.sum(M)
                sum_diag_sq = np.sum(diag_M**2)
                sum_diag = np.sum(diag_M)
                sum_off_diag_sq = (sum_M_sq - sum_diag_sq) / 2.0
                sum_off_diag = (sum_M - sum_diag) / 2.0
                cost_c = (sum_off_diag_sq - sum_off_diag) / 2.0
                total += cost_c
            return total
        
        current_cost = compute_total_cost(Ms)
        
        # SA parameters
        T = 1.0  # Initial temperature
        T_min = 0.01
        cooling = 0.995
        
        best_cost = current_cost
        best_colors = colors[:]
        
        step = 0
        while T > T_min and time.time() < deadline - 5:
            step += 1
            
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            u, v = edges[e_idx]
            old_c = colors[e_idx]
            new_c = random.randint(0, 4)
            if old_c == new_c:
                continue
            
            # Compute delta cost for changing edge (u,v) from old_c to new_c
            # For color old_c: removing edge (u,v)
            # For color new_c: adding edge (u,v)
            
            # Delta for removing (u,v) from old_c
            # This affects M_old = mats[old_c] @ mats[old_c]
            # For each pair (a,b), M[a,b] changes if a or b is u or v
            # Specifically, M[u,w] decreases by 1 if w is a neighbor of v in old_c (and w != u)
            # M[v,w] decreases by 1 if w is a neighbor of u in old_c (and w != v)
            # M[u,v] decreases by deg_u + deg_v - 2 (but we don't care about diagonal for C4 count)
            
            # More precisely, for the cost function:
            # cost = sum_{a < b} C(M[a,b], 2) = 0.5 * sum_{a != b} (M[a,b]^2 - M[a,b])
            # When we change M[a,b] by delta, the cost change is:
            # 0.5 * ((M[a,b]+delta)^2 - (M[a,b]+delta) - (M[a,b]^2 - M[a,b]))
            # = 0.5 * (2*M[a,b]*delta + delta^2 - delta)
            # = M[a,b]*delta + 0.5*delta^2 - 0.5*delta
            
            # For removing edge (u,v) from color old_c:
            # For each w != u, v:
            #   M[u,w] decreases by 1 if mats[old_c][v,w] == 1
            #   M[v,w] decreases by 1 if mats[old_c][u,w] == 1
            # For w = u: M[u,u] changes, but diagonal doesn't affect cost
            # For w = v: M[v,v] changes, but diagonal doesn't affect cost
            # Also M[u,v] and M[v,u] change, but these are off-diagonal
            
            # Let's compute delta_cost for removing (u,v) from old_c
            delta_remove = 0.0
            M_old = Ms[old_c]
            
            # For each w, check if w is neighbor of v in old_c (affects M[u,w])
            # and if w is neighbor of u in old_c (affects M[v,w])
            for w in range(n):
                if w == u or w == v:
                    continue
                # M[u,w] changes by -1 if mats[old_c][v,w] == 1
                if mats[old_c][v, w] == 1:
                    # M[u,w] decreases by 1
                    delta = -1
                    old_M = M_old[u, w]
                    delta_remove += old_M * delta + 0.5 * delta * delta - 0.5 * delta
                # M[v,w] changes by -1 if mats[old_c][u,w] == 1
                if mats[old_c][u, w] == 1:
                    delta = -1
                    old_M = M_old[v, w]
                    delta_remove += old_M * delta + 0.5 * delta * delta - 0.5 * delta
            
            # Also M[u,v] and M[v,u] change
            # M[u,v] = sum_k mats[old_c][u,k] * mats[old_c][k,v]
            # When we remove edge (u,v), this affects the computation
            # Actually, M[u,v] = number of common neighbors of u and v in old_c
            # Removing edge (u,v) doesn't change the common neighbors, so M[u,v] stays the same?
            # No, wait. M = A @ A, so M[u,v] = sum_k A[u,k] A[k,v]
            # When we remove edge (u,v), A[u,v] and A[v,u] become 0
            # So M[u,v] loses the term where k=v: A[u,v]A[v,v] = 0 (no self-loops)
            # And M[v,u] loses the term where k=u: A[v,u]A[u,u] = 0
            # So M[u,v] and M[v,u] are unchanged? That's not right.
            # Actually, M[u,v] = sum_k A[u,k] A[k,v]. The term k=u is A[u,u]A[u,v] = 0.
            # The term k=v is A[u,v]A[v,v] = 0.
            # So removing edge (u,v) doesn't affect M[u,v] or M[v,u].
            # But it affects M[u,w] for w != u,v if v is a neighbor of w, etc.
            
            # For adding (u,v) to new_c:
            delta_add = 0.0
            M_new = Ms[new_c]
            
            for w in range(n):
                if w == u or w == v:
                    continue
                if mats[new_c][v, w] == 1:
                    delta = 1
                    old_M = M_new[u, w]
                    delta_add += old_M * delta + 0.5 * delta * delta - 0.5 * delta
                if mats[new_c][u, w] == 1:
                    delta = 1
                    old_M = M_new[v, w]
                    delta_add += old_M * delta + 0.5 * delta * delta - 0.5 * delta
            
            delta_cost = delta_add - delta_remove
            
            # Accept or reject
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept the move
                colors[e_idx] = new_c
                
                # Update matrices
                mats[old_c][u, v] = 0
                mats[old_c][v, u] = 0
                mats[new_c][u, v] = 1
                mats[new_c][v, u] = 1
                
                # Update Ms
                # This is expensive to do incrementally, so we recompute
                Ms[old_c] = mats[old_c] @ mats[old_c]
                Ms[new_c] = mats[new_c] @ mats[new_c]
                
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                
                if best_cost == 0:
                    return best_colors, best_n
            else:
                # Reject the move
                pass
            
            T *= cooling
        
        return best_colors, best_cost

    # Try to find solutions for larger n, starting from high to low
    # We prioritize finding a valid solution for a large n.
    # If we fail for 26, we try 25, etc.
    # But we must return the best one found.
    
    # To ensure we don't spend all time on n=26 and fail, we allocate time based on n.
    # n=26 is hard. n=12 is easy.
    # Strategy: Try n=12, 13, ... sequentially? 
    # If we find n=12, we have a score > 1.0.
    # If we have time, try n=13, etc.
    # This guarantees an improvement over the seed (which likely gets stuck at 11 or low 10s).
    
    for n in range(12, 27):
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this n
        # Larger n gets less time if we are running out of time
        # But we want to maximize n.
        # Let's give each n a fixed number of iterations or a time slice.
        # Time slice: 10 seconds for each n? 
        # If we find a valid one, we update best and continue to next n.
        
        time_limit = deadline - 5
        result_colors, result_cost = solve_n_sa(n, time_limit)
        
        if result_colors is not None and result_cost == 0:
            best_n = n
            best_colors = result_colors[:]
            continue
        elif result_colors is not None:
            # Even if not zero, if it's better than current best for this n, keep it
            # But we need a valid solution, so only update if cost is 0
            pass

    return {'n': best_n, 'colors': best_colors}
