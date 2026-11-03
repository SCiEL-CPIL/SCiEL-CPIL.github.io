"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def calc_cost(p):
        b_perm = b[:, p][:, p]
        return int(np.sum(a * b_perm))
    
    # Precompute interaction vectors for fast delta calculation
    # For each facility i and location u, precompute:
    #   row_vec[i, u] = a[i, :] (the row of A for facility i)
    #   col_vec[i, u] = a[:, i] (the column of A for facility i)
    # But these don't depend on u, so we just use a[i] and a[:, i]
    
    # Key insight: delta for swap (i,j) with u=p[i], v=p[j] can be written as:
    # delta = (a[i] - a[j]) @ (B_p[v] - B_p[u]) + (a[:,i] - a[:,j]) @ (B_p[:,v] - B_p[:,u])
    #         + correction terms for the 4 corner cells
    # We precompute diff_a_row[i,j] = a[i] - a[j] and diff_a_col[i,j] = a[:,i] - a[:,j]
    # But that's O(n^2 * n) memory. Instead, compute on the fly but use dot products.
    
    def get_delta_fast(i, j, p, B_p):
        u = p[i]
        v = p[j]
        
        # Precompute the difference vectors
        # da_row = a[i] - a[j]  (shape n)
        # da_col = a[:, i] - a[:, j]  (shape n)
        # db_row = B_p[v] - B_p[u]  (shape n)
        # db_col = B_p[:, v] - B_p[:, u]  (shape n)
        
        da_row = a[i] - a[j]
        da_col = a[:, i] - a[:, j]
        db_row = B_p[v] - B_p[u]
        db_col = B_p[:, v] - B_p[:, u]
        
        # Main term: da_row @ db_row + da_col @ db_col
        main_term = np.dot(da_row, db_row) + np.dot(da_col, db_col)
        
        # Corner corrections:
        # The main term includes k=i and k=j contributions that need adjustment.
        # For k=i: da_row[i]*db_row[i] + da_col[i]*db_col[i]
        #   = (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[j,i])*(b[u,v]-b[u,u])
        # For k=j: da_row[j]*db_row[j] + da_col[j]*db_col[j]
        #   = (a[i,j]-a[j,j])*(b[v,v]-b[u,v]) + (a[i,j]-a[j,j])*(b[v,v]-b[v,u])
        
        # Direct corner delta (new - old for the 4 corner terms):
        # a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u])
        
        b_uu = b[u, u]
        b_vv = b[v, v]
        b_uv = b[u, v]
        b_vu = b[v, u]
        
        # The main term's k=i contribution:
        # (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[j,i])*(b[u,v]-b[u,u])
        ki_contrib = (a[i, i] - a[j, i]) * (b_vu - b_uu) + (a[i, i] - a[j, i]) * (b_uv - b_uu)
        
        # The main term's k=j contribution:
        # (a[i,j]-a[j,j])*(b[v,v]-b[u,v]) + (a[i,j]-a[j,j])*(b[v,v]-b[v,u])
        kj_contrib = (a[i, j] - a[j, j]) * (b_vv - b_uv) + (a[i, j] - a[j, j]) * (b_vv - b_vu)
        
        # Direct corner delta:
        direct = (a[i, i] * (b_vv - b_uu) + a[j, j] * (b_uu - b_vv) + 
                  a[i, j] * (b_vu - b_uv) + a[j, i] * (b_uv - b_vu))
        
        delta = main_term - ki_contrib - kj_contrib + direct
        return int(delta)
    
    # Initialization Strategies
    
    def init_rank():
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p

    def init_rank_reversed():
        """Reverse rank matching - might find different local optima"""
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")[::-1]
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p

    def init_spectral():
        """Spectral initialization using eigendecomposition"""
        try:
            # Use the leading eigenvector of B to get location ordering
            # Then match facility "centrality" to location "centrality"
            # For distance matrices, small eigenvalues correspond to smooth modes
            eigvals, eigvecs = np.linalg.eigh(b.astype(np.float64))
            # Use the eigenvector with smallest absolute eigenvalue (most "central")
            # Or use a combination
            # Let's use the eigenvector corresponding to the smallest eigenvalue
            idx = np.argmin(np.abs(eigvals))
            loc_scores = eigvecs[:, idx]
            loc_order = np.argsort(loc_scores, kind="stable")
            
            a_row_sums = a.sum(axis=1)
            fac_order = np.argsort(a_row_sums, kind="stable")
            
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            return p
        except:
            return np.random.permutation(n).astype(np.int64)

    def init_random():
        return np.random.permutation(n).astype(np.int64)

    def init_greedy():
        """Greedy construction: place facilities one by one at best location"""
        p = np.full(n, -1, dtype=np.int64)
        remaining_locs = set(range(n))
        
        # Order facilities by flow importance
        a_row_sums = a.sum(axis=1)
        fac_order = np.argsort(-a_row_sums, kind="stable")
        
        for fac in fac_order:
            best_loc = None
            best_cost = np.inf
            for loc in remaining_locs:
                # Cost of placing fac at loc given current assignment
                # sum over already-placed facilities j: a[fac,j] * b[loc, p[j]] + a[j,fac] * b[p[j], loc]
                # + a[fac,fac] * b[loc,loc]
                cost = a[fac, fac] * b[loc, loc]
                for j in range(n):
                    if p[j] >= 0:
                        cost += a[fac, j] * b[loc, p[j]] + a[j, fac] * b[p[j], loc]
                if cost < best_cost:
                    best_cost = cost
                    best_loc = loc
            p[fac] = best_loc
            remaining_locs.remove(best_loc)
        
        return p

    # Local Search: Best Improvement with Tabu
    def local_search(p, current_cost, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0
        
        best_p = p.copy()
        best_cost = current_cost
        B_p = b[:, p].copy()
        
        tabu_tenure = 12
        tabu_list = {}
        iteration = 0
        no_improve = 0
        max_no_improve = 25
        
        while time.time() < max_time:
            best_delta = np.inf
            best_i = -1
            best_j = -1
            is_aspiration = False
            
            # Iterate over all pairs with early termination if possible
            for i in range(n):
                for j in range(i + 1, n):
                    # Tabu check
                    move_key = (i, j)
                    tabu_until = tabu_list.get(move_key, -1)
                    
                    if tabu_until > iteration:
                        # Check aspiration
                        delta = get_delta_fast(i, j, p, B_p)
                        new_cost = current_cost + delta
                        if new_cost < best_cost:
                            is_aspiration = True
                        else:
                            continue
                    else:
                        is_aspiration = False
                    
                    if not is_aspiration:
                        delta = get_delta_fast(i, j, p, B_p)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        is_aspiration = False
            
            if best_i == -1:
                # No improving move found
                # Pick a random non-tabu move
                valid_moves = []
                for i in range(n):
                    for j in range(i + 1, n):
                        move_key = (i, j)
                        if tabu_list.get(move_key, -1) <= iteration:
                            valid_moves.append((i, j))
                if not valid_moves:
                    valid_moves = [(i, j) for i in range(n) for j in range(i + 1, n)]
                
                best_i, best_j = random.choice(valid_moves)
                best_delta = get_delta_fast(best_i, best_j, p, B_p)
            
            # Perform move
            u_old = p[best_i]
            v_old = p[best_j]
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            
            # Update B_p efficiently
            # B_p[k, l] = b[p[k], p[l]]
            # After swap: B_p[:, best_i] should be b[:, v_old] (was u_old)
            #               B_p[:, best_j] should be b[:, u_old] (was v_old)
            #               B_p[u_old, :] should be b[u_old, p]
            #               B_p[v_old, :] should be b[v_old, p]
            # But we need to be careful about the intersections
            
            # Save old values
            B_p_old_ui = B_p[u_old, best_i]
            B_p_old_vj = B_p[v_old, best_j]
            B_p_old_uv = B_p[u_old, best_j]
            B_p_old_vu = B_p[v_old, best_i]
            
            # Update columns
            B_p[:, best_i] = b[:, v_old]
            B_p[:, best_j] = b[:, u_old]
            
            # Update rows
            B_p[u_old, :] = b[u_old, p]
            B_p[v_old, :] = b[v_old, p]
            
            # Fix the 4 corner cells (they were set correctly by row/col updates,
            # but let's verify and fix if needed)
            # B_p[u_old, best_i] should be b[u_old, p[best_i]] = b[u_old, v_old]
            # B_p[u_old, best_j] should be b[u_old, p[best_j]] = b[u_old, u_old]
            # B_p[v_old, best_i] should be b[v_old, p[best_i]] = b[v_old, v_old]
            # B_p[v_old, best_j] should be b[v_old, p[best_j]] = b[v_old, u_old]
            # These should be correct from the row/col updates, but let's be safe
            B_p[u_old, best_i] = b[u_old, v_old]
            B_p[u_old, best_j] = b[u_old, u_old]
            B_p[v_old, best_i] = b[v_old, v_old]
            B_p[v_old, best_j] = b[v_old, u_old]
            
            # Update Tabu
            tabu_list[(best_i, best_j)] = iteration + tabu_tenure
            
            # Update Best
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            iteration += 1
            
            # Stagnation handling
            if no_improve > max_no_improve:
                # Perturb with 3-5 random swaps
                num_perturb = random.randint(3, 5)
                for _ in range(num_perturb):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                # Recalculate cost and B_p
                current_cost = calc_cost(p)
                B_p = b[:, p].copy()
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve = 0
                tabu_list.clear()
        
        return best_p, best_cost

    # Multi-start Local Search
    best_overall_p = None
    best_overall_cost = np.inf
    
    # Try different initializations
    inits = [
        init_rank(),
        init_rank_reversed(),
        init_spectral(),
        init_random()
    ]
    
    # Add greedy if time permits
    if time.time() < deadline - 30.0:
        inits.append(init_greedy())
    
    for init_p in inits:
        if time.time() > deadline - 15.0:
            break
        
        init_cost = calc_cost(init_p)
        if init_cost < best_overall_cost:
            best_overall_cost = init_cost
            best_overall_p = init_p.copy()
        
        # Run local search
        ls_p, ls_cost = local_search(init_p, init_cost)
        
        if ls_cost < best_overall_cost:
            best_overall_cost = ls_cost
            best_overall_p = ls_p.copy()
    
    # Final polish: run a short local search on the best solution found
    if best_overall_p is not None and time.time() < deadline - 10.0:
        final_p, final_cost = local_search(best_overall_p.copy(), best_overall_cost, max_time=deadline - 5.0)
        if final_cost < best_overall_cost:
            best_overall_cost = final_cost
            best_overall_p = final_p
    
    return best_overall_p.tolist()

# EVOLVE-BLOCK-END
