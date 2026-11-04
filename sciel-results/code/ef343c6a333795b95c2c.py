"""QAP solver for tho150 using B_perm-accelerated local search with SA restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        return float(np.sum(a_f * b_f[np.ix_(p, p)]))

    def mds_embedding(D, dim=2):
        nn = D.shape[0]
        J = np.eye(nn) - np.ones((nn, nn)) / nn
        B = -0.5 * J @ (D ** 2) @ J
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals = vals[:dim]
        vecs = vecs[:, :dim]
        vals = np.maximum(vals, 0)
        coords = vecs * np.sqrt(vals)
        return coords

    def procrustes_alignment(X, Y):
        from scipy.optimize import linear_sum_assignment
        U, S, Vt = np.linalg.svd(Y.T @ X)
        Q = U @ Vt
        X_aligned = X @ Q
        X_norms = np.sum(X_aligned ** 2, axis=1)
        Y_norms = np.sum(Y ** 2, axis=1)
        Cross = X_aligned @ Y.T
        C = X_norms[:, None] + Y_norms[None, :] - 2 * Cross
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def build_B_perm(p):
        return b_f[np.ix_(p, p)]

    def swap_cost(a_i, a_j, B_perm, p, li, lj):
        """Compute delta cost for swapping facilities i and j.
        a_i = a_f[i,:], a_j = a_f[j,:]
        B_perm[k,l] = b[p[k], p[l]]
        After swap: B_perm'[i,l] = B_perm[j,l] for l!=j, B_perm'[i,j]=B_perm[j,i]
                    B_perm'[j,l] = B_perm[i,l] for l!=i, B_perm'[j,i]=B_perm[i,j]
        delta = sum_l a[i,l]*(B'[i,l]-B[i,l]) + sum_l a[j,l]*(B'[j,l]-B[j,l])
              + sum_l a[l,i]*(B'[l,i]-B[l,i]) + sum_l a[l,j]*(B'[l,j]-B[l,j])
        For l not in {i,j}:
          B'[i,l] = B[j,l], B'[j,l] = B[i,l], B'[l,i]=B[l,j], B'[l,j]=B[l,i]
        So delta = sum_{l!=i,j} [a[i,l](B[j,l]-B[i,l]) + a[j,l](B[i,l]-B[j,l])
                                 + a[l,i](B[l,j]-B[l,i]) + a[l,j](B[l,i]-B[l,j])]
                 + terms for l=i and l=j (which cancel in pairs)
        Actually let's just compute directly:
        delta = (a_i - a_j) @ (B_perm[j,:] - B_perm[i,:]) 
              + (a_f[:,i] - a_f[:,j]) @ (B_perm[:,j] - B_perm[:,i])
        """
        diff_row = a_f[i, :] - a_f[j, :]
        diff_col = a_f[:, i] - a_f[:, j]
        drow = B_perm[j, :] - B_perm[i, :]
        dcol = B_perm[:, j] - B_perm[:, i]
        return np.dot(diff_row, drow) + np.dot(diff_col, dcol)

    def perform_swap(p, B_perm, i, j):
        li, lj = p[i], p[j]
        p[i], p[j] = lj, li
        # Update B_perm: swap rows i,j and cols i,j
        B_perm[[i, j], :] = B_perm[[j, i], :]
        B_perm[:, [i, j]] = B_perm[:, [j, i]]

    def local_search(p, B_perm, current_cost, end_time, seed=42, max_no_improve=3):
        """2-swap local search with B_perm acceleration."""
        curr_p = p.copy()
        curr_B = B_perm.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        no_improve_count = 0
        
        while time.time() < end_time:
            improved = False
            rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > end_time:
                    break
                
                delta = swap_cost(i, j, curr_B, curr_p, curr_p[i], curr_p[j])
                
                if delta < -1e-9:
                    perform_swap(curr_p, curr_B, i, j)
                    curr_cost += delta
                    improved = True
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
            else:
                no_improve_count = 0
        
        return best_p, best_cost, build_B_perm(best_p)

    def simulated_annealing(p, B_perm, current_cost, end_time, seed=777):
        """SA with B_perm for faster moves."""
        curr_p = p.copy()
        curr_B = B_perm.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        T = 5000.0
        cooling = 0.9995
        
        while time.time() < end_time and T > 0.5:
            i, j = rng.choice(n, 2, replace=False)
            
            delta = swap_cost(i, j, curr_B, curr_p, curr_p[i], curr_p[j])
            
            if delta < 0 or rng.random_sample() < np.exp(-delta / T):
                perform_swap(curr_p, curr_B, i, j)
                curr_cost += delta
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling
        
        return best_p, best_cost, build_B_perm(best_p)

    # === Initialization ===
    best_p_init = None
    best_cost_init = np.inf

    # MDS + Procrustes
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0
    
    for dim in [2, 3, 5]:
        if time.time() > deadline - 60.0:
            break
        X = mds_embedding(a_sym, dim)
        Y = mds_embedding(b_sym, dim)
        p_try = procrustes_alignment(X, Y)
        cost_try = compute_cost(p_try)
        if cost_try < best_cost_init:
            best_cost_init = cost_try
            best_p_init = p_try

    # Rank matching
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)
    if cost_rank < best_cost_init:
        best_cost_init = cost_rank
        best_p_init = p_rank

    # Random
    rng_init = np.random.RandomState(42)
    p_rand = rng_init.permutation(n)
    cost_rand = compute_cost(p_rand)
    if cost_rand < best_cost_init:
        best_cost_init = cost_rand
        best_p_init = p_rand

    # === Phase 1: Local Search ===
    B_perm_init = build_B_perm(best_p_init)
    end_ls1 = deadline - 40.0
    best_p, best_cost, best_B = local_search(best_p_init, B_perm_init, best_cost_init, end_ls1, seed=12345, max_no_improve=5)

    # === Phase 2: SA with restarts ===
    remaining = deadline - time.time()
    if remaining > 15.0:
        # Multiple SA restarts from perturbed solutions
        n_restarts = min(5, int(remaining / 12.0))
        rng_perturb = np.random.RandomState(999)
        
        for r in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb best solution
            p_pert = best_p.copy()
            B_pert = best_B.copy()
            cost_pert = best_cost
            
            n_swaps = 3 + rng_perturb.randint(3)  # 3-5 swaps
            for _ in range(n_swaps):
                i, j = rng_perturb.choice(n, 2, replace=False)
                delta = swap_cost(i, j, B_pert, p_pert, p_pert[i], p_pert[j])
                perform_swap(p_pert, B_pert, i, j)
                cost_pert += delta
            
            # SA on perturbed solution
            end_sa = deadline - 5.0
            p_sa, cost_sa, B_sa = simulated_annealing(p_pert, B_pert, cost_pert, end_sa, seed=700 + r)
            
            if cost_sa < best_cost:
                best_p, best_cost, best_B = p_sa, cost_sa, B_sa
            
            # Quick local search polish on SA result
            if time.time() < deadline - 3.0:
                end_polish = deadline - 2.0
                p_pol, cost_pol, B_pol = local_search(p_sa, B_sa, cost_sa, end_polish, seed=800 + r, max_no_improve=2)
                if cost_pol < best_cost:
                    best_p, best_cost, best_B = p_pol, cost_pol, B_pol

    # === Final polish ===
    if time.time() < deadline - 1.0:
        end_final = deadline - 0.5
        p_final, cost_final, B_final = local_search(best_p, best_B, best_cost, end_final, seed=999, max_no_improve=10)
        if cost_final < best_cost:
            best_p, best_cost = p_final, cost_final

    return best_p.tolist()
# EVOLVE-BLOCK-END
