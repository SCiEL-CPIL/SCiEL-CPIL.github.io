import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # We aim for n=26
    n = 26
    num_colors = 5
    
    def init_state():
        # col[i][j] is the color of edge (i,j)
        col = np.zeros((n, n), dtype=np.int8)
        # A[c] is the adjacency matrix for color c
        A = [np.zeros((n, n), dtype=np.int8) for _ in range(num_colors)]
        # K[c] is the matrix of common neighbor counts for color c
        # K[c][i][j] = number of common neighbors of i and j in color c
        K = [np.zeros((n, n), dtype=np.int32) for _ in range(num_colors)]
        
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, num_colors - 1)
                col[i, j] = c
                col[j, i] = c
                A[c][i, j] = 1
                A[c][j, i] = 1
        
        # Initialize K
        for c in range(num_colors):
            K[c] = A[c] @ A[c]
            # K[i][i] is the degree of i in color c, which is not a C4 count itself, 
            # but we only care about i != j for C4s.
            # However, the formula sum_{i<j} binom(K[i][j], 2) works.
            
        return col, A, K

    def calc_total_penalty(K):
        total = 0
        for c in range(num_colors):
            k_mat = K[c]
            # We need sum_{i<j} binom(k_mat[i][j], 2)
            # k_mat is symmetric.
            # binom(k, 2) = k*(k-1)//2
            # We can compute this vectorized.
            # Create a matrix of binom values
            vals = k_mat * (k_mat - 1) // 2
            # Sum upper triangle
            # np.triu_indices
            i_idx, j_idx = np.triu_indices(n, k=1)
            total += np.sum(vals[i_idx, j_idx])
        return total

    def get_penalty_delta(u, v, c_old, c_new, A, K):
        """
        Calculate change in penalty if edge (u,v) changes from c_old to c_new.
        Updates A and K are NOT performed here, only delta is calculated.
        """
        delta = 0
        
        # Helper to get penalty contribution of a pair (i,j) in color c
        # Contrib = binom(K[c][i][j], 2)
        
        # The change affects K[c_old] and K[c_new].
        # For c_old, we are removing edge (u,v).
        # This means:
        # K[c_old][u, x] decreases by A[c_old][v, x] for all x
        # K[c_old][v, x] decreases by A[c_old][u, x] for all x
        # Note: K is symmetric, so K[x, u] and K[x, v] also change.
        # The pairs (i,j) that change are those where i is u or v, or j is u or v.
        # Specifically, the entries in K[c_old] that change are:
        # (u, x) and (v, x) for all x != u, v.
        # Also (u, v) itself? 
        # K[c][u][v] is the number of common neighbors of u and v.
        # Does removing edge (u,v) change the common neighbors of u and v?
        # Common neighbors of u and v are w such that (u,w) and (v,w) are in color c.
        # The edge (u,v) itself is not a common neighbor.
        # So K[c][u][v] does NOT change when edge (u,v) is removed?
        # Wait. 
        # K[c][u][v] = sum_w A[c][u][w] * A[c][v][w].
        # Removing A[c][u][v] and A[c][v][u] does not affect the sum for w != u, v.
        # For w=u: A[c][u][u]=0.
        # For w=v: A[c][v][v]=0.
        # So yes, K[c][u][v] is unchanged.
        
        # So we only need to consider pairs (u, x) and (v, x) for x != u, v.
        
        # Let's iterate over x
        # For c_old (removal):
        # For each x != u, v:
        #   k_old_u_x = K[c_old][u][x]
        #   k_new_u_x = k_old_u_x - A[c_old][v][x]
        #   delta += binom(k_new_u_x, 2) - binom(k_old_u_x, 2)
        
        #   k_old_v_x = K[c_old][v][x]
        #   k_new_v_x = k_old_v_x - A[c_old][u][x]
        #   delta += binom(k_new_v_x, 2) - binom(k_old_v_x, 2)
        
        # For c_new (addition):
        # For each x != u, v:
        #   k_old_u_x = K[c_new][u][x]
        #   k_new_u_x = k_old_u_x + A[c_new][v][x]
        #   delta += binom(k_new_u_x, 2) - binom(k_old_u_x, 2)
        
        #   k_old_v_x = K[c_new][v][x]
        #   k_new_v_x = k_old_v_x + A[c_new][u][x]
        #   delta += binom(k_new_v_x, 2) - binom(k_old_v_x, 2)

        # Optimization: Vectorize the sum over x
        # We need sum_{x != u, v} [ binom(k - a, 2) - binom(k, 2) ]
        
        def calc_delta_for_color(c, sign):
            # sign = -1 for removal, +1 for addition
            Kc = K[c]
            Ac = A[c]
            
            # Vectors of current common neighbor counts for u and v
            # Exclude u and v themselves? 
            # The pairs are (u,x) and (v,x).
            # x ranges 0..n-1, x!=u, v.
            
            # Let's get the rows Kc[u] and Kc[v]
            row_u = Kc[u].copy()
            row_v = Kc[v].copy()
            
            # The changes to these rows
            # For (u,x), change is -Ac[v][x] (if sign=-1)
            # For (v,x), change is -Ac[u][x] (if sign=-1)
            
            # We need to sum binom(new, 2) - binom(old, 2) for all x != u, v.
            
            # Let's create masks for x != u, v
            mask = np.ones(n, dtype=bool)
            mask[u] = False
            mask[v] = False
            
            # Changes for row_u
            delta_row_u = sign * Ac[v]
            new_row_u = row_u + delta_row_u
            
            # Changes for row_v
            delta_row_v = sign * Ac[u]
            new_row_v = row_v + delta_row_v
            
            # Calculate penalty difference
            # binom(k, 2) = k*(k-1)//2
            # diff = (new*(new-1) - old*(old-1)) // 2
            
            # We only care about x where mask is True
            old_u = row_u[mask]
            new_u = new_row_u[mask]
            old_v = row_v[mask]
            new_v = new_row_v[mask]
            
            diff_u = (new_u * (new_u - 1) - old_u * (old_u - 1)) // 2
            diff_v = (new_v * (new_v - 1) - old_v * (old_v - 1)) // 2
            
            return np.sum(diff_u) + np.sum(diff_v)
        
        delta_c_old = calc_delta_for_color(c_old, -1)
        delta_c_new = calc_delta_for_color(c_new, 1)
        
        return delta_c_old + delta_c_new

    def apply_move(u, v, c_old, c_new, col, A, K):
        col[u, v] = c_new
        col[v, u] = c_new
        
        # Update A
        A[c_old][u, v] = 0
        A[c_old][v, u] = 0
        A[c_new][u, v] = 1
        A[c_new][v, u] = 1
        
        # Update K
        # K[c_old] changes:
        # K[u, x] -= A[c_old][v, x]
        # K[v, x] -= A[c_old][u, x]
        # And symmetric counterparts.
        
        # Let's do it with vector ops
        # For c_old
        Kc = K[c_old]
        Ac = A[c_old]
        
        # Update rows and columns for u and v
        # K[u, :] changes by -Ac[v, :]
        # K[:, u] changes by -Ac[:, v]
        # K[v, :] changes by -Ac[u, :]
        # K[:, v] changes by -Ac[:, u]
        
        # Be careful not to double count or miss symmetry.
        # K is symmetric.
        # K[i, j] is common neighbors of i, j.
        # If we remove edge (u, v), 
        # For any x, K[u, x] decreases by 1 if (v, x) was an edge.
        # This is exactly Ac[v, x].
        
        Kc[u, :] -= Ac[v, :]
        Kc[:, u] -= Ac[:, v]
        Kc[v, :] -= Ac[u, :]
        Kc[:, v] -= Ac[:, u]
        
        # Note: K[u, v] and K[v, u] are affected by these operations?
        # K[u, v] -= Ac[v, v] (which is 0) and K[u, v] -= Ac[v, v] (0).
        # Wait, K[u, :] -= Ac[v, :] sets K[u, v] -= Ac[v, v] = 0.
        # K[:, u] -= Ac[:, v] sets K[v, u] -= Ac[v, v] = 0.
        # So K[u, v] is unchanged, which is correct.
        
        # However, we modified K[u, :] and K[:, u].
        # We also modified K[v, :] and K[:, v].
        # Is there overlap?
        # K[u, v] is in K[u, :] and K[:, v].
        # In K[u, :] update: K[u, v] -= Ac[v, v] = 0.
        # In K[:, v] update: K[u, v] -= Ac[u, u] = 0.
        # So it's fine.
        
        # For c_new
        Kc = K[c_new]
        Ac = A[c_new]
        
        Kc[u, :] += Ac[v, :]
        Kc[:, u] += Ac[:, v]
        Kc[v, :] += Ac[u, :]
        Kc[:, v] += Ac[:, u]

    best_sol = None
    
    # Try multiple restarts
    for restart in range(10):
        if time.time() > deadline: break
        
        col, A, K = init_state()
        current_penalty = calc_total_penalty(K)
        
        if current_penalty == 0:
            best_sol = col
            break
            
        T = 1.0
        no_improve = 0
        best_local_penalty = current_penalty
        best_local_col = col.copy()
        
        # Precompute list of edges for random selection
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() < deadline:
            if current_penalty == 0:
                best_sol = col
                break
                
            # Pick random edge
            u, v = random.choice(edges)
            
            c_old = col[u, v]
            c_new = random.randint(0, num_colors - 1)
            if c_new == c_old: continue
            
            delta = get_penalty_delta(u, v, c_old, c_new, A, K)
            
            if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / T)):
                # Accept
                apply_move(u, v, c_old, c_new, col, A, K)
                
                current_penalty += delta
                no_improve = 0
                if current_penalty < best_local_penalty:
                    best_local_penalty = current_penalty
                    best_local_col = col.copy()
            else:
                no_improve += 1
                
            if no_improve > 2000:
                break
            T *= 0.9995
            if T < 0.1: T = 0.1
            
        if best_local_penalty == 0:
            best_sol = best_local_col
            break

    if best_sol is not None:
        res_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                res_colors.append(int(best_sol[i, j]))
        return {'n': n, 'colors': res_colors}
    
    # Fallback: n=25
    n = 25
    col, A, K = init_state()
    current_penalty = calc_total_penalty(K)
    T = 1.0
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for _ in range(100000):
        if time.time() > deadline: break
        if current_penalty == 0: break
        
        u, v = random.choice(edges)
        c_old = col[u, v]
        c_new = random.randint(0, num_colors - 1)
        if c_new == c_old: continue
        
        delta = get_penalty_delta(u, v, c_old, c_new, A, K)
        
        if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / T)):
            apply_move(u, v, c_old, c_new, col, A, K)
            current_penalty += delta
        T *= 0.999
        if T < 0.1: T = 0.1
        
    if current_penalty == 0:
        res_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                res_colors.append(int(col[i, j]))
        return {'n': n, 'colors': res_colors}
        
    # Final fallback: n=11
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
