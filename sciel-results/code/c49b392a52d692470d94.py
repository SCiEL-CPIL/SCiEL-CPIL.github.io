"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin for 120s limit
    
    # 1. Initialization: Greedy Vector-Space Heuristic
    # We build the permutation p sequentially.
    # Let S be the set of assigned facilities, and T be the set of assigned locations.
    # The cost of the partial assignment is sum_{i in S, j in S} a[i,j] b[p[i], p[j]].
    # When we add a new facility i to location k (where i not in S, k not in T),
    # the marginal cost increase is:
    # delta = sum_{j in S} a[i,j] b[k, p[j]] + sum_{j in S} a[j,i] b[p[j], k] + a[i,i] b[k,k]
    # Note: a and b are usually symmetric in QAP, but we handle general case.
    # Let v_k = sum_{j in S} (a[i,j] + a[j,i]) b[k, p[j]] + a[i,i] b[k,k]
    # Actually, it's easier to maintain a vector C of size n representing the cost
    # of assigning the *next* facility to each available location.
    
    # However, the "next" facility changes. 
    # A common greedy approach is to pick the facility-location pair (i, k) with minimum marginal cost,
    # where marginal cost depends on current assignment.
    # To make this O(n^2) or O(n^3) total, we can precompute or update efficiently.
    
    # Alternative simple strong init: Spectral
    # But let's try the specific directive: "Linearized Greedy Construction"
    # We will iterate n times. At each step, we have unassigned facilities U and unassigned locations L.
    # We want to pick i in U and k in L minimizing the marginal cost.
    # Marginal cost of assigning i to k given current P (partial):
    # MC(i, k) = sum_{j in Assigned} (a[i,j] b[k, p[j]] + a[j,i] b[p[j], k]) + a[i,i] b[k,k]
    # This is O(n) to compute for a pair. O(n^2) to find best pair. O(n^3) total.
    # For n=150, n^3 = 3.3M operations, which is very fast.
    
    p = np.full(n, -1, dtype=np.int64)
    assigned_facilities = np.zeros(n, dtype=bool)
    assigned_locations = np.zeros(n, dtype=bool)
    
    # Precompute a and b rows? No, just use numpy slices.
    
    for step in range(n):
        if time.time() > deadline - 10: # Leave time for LS
            break
            
        best_mc = np.inf
        best_i = -1
        best_k = -1
        
        # Iterate over unassigned facilities
        unassigned_fac_idx = np.where(~assigned_facilities)[0]
        unassigned_loc_idx = np.where(~assigned_locations)[0]
        
        # To speed up, we can vectorize the MC calculation for all pairs (i, k)
        # MC matrix M where M[i, k] = marginal cost
        # M[i, k] = sum_{j in Assigned} (a[i,j] b[k, p[j]] + a[j,i] b[p[j], k]) + a[i,i] b[k,k]
        
        # Let's compute this matrix for all unassigned i and k.
        # This might be memory heavy if done naively, but n=150 is small.
        # Size 150x150.
        
        # We need to sum over assigned j.
        # Let A_sub = a[unassigned_fac_idx, :]  # shape (U, n)
        # Let B_sub = b[:, unassigned_loc_idx]  # shape (n, L) -> transpose to (L, n)
        
        # The term sum_{j in Assigned} a[i,j] b[k, p[j]]
        # This is like A_sub @ (B_sub.T * assigned_locations_mask) ... wait.
        # b[k, p[j]] depends on p[j].
        
        # Let's build a matrix D where D[k, j] = b[k, p[j]] if j is assigned, else 0.
        # D is n x n.
        # D[k, j] = b[k, p[j]] * assigned_facilities[j]
        
        # Then sum_{j} a[i,j] D[k,j] = (a[i, :] @ D[:, k])? No.
        # (A_sub @ D.T)[i, k] = sum_j a[i,j] D.T[j, k] = sum_j a[i,j] D[k, j] = sum_j a[i,j] b[k, p[j]]
        
        # Similarly for the second term: sum_{j} a[j,i] b[p[j], k]
        # This is sum_j a[j,i] b[p[j], k]
        # Let E[j, k] = b[p[j], k] * assigned_facilities[j]
        # Then (A_sub.T @ E)[k, i]? No.
        # We want result for i, k.
        # sum_j a[j,i] E[j, k] = (A_sub.T @ E)[i, k]? 
        # A_sub.T is (n, U). E is (n, L).
        # (A_sub.T @ E) is (n, L). Element [i, k] is sum_j A_sub.T[j, i] E[j, k] = sum_j a[i, j] E[j, k].
        # But we need a[j, i]. If a is symmetric, a[i,j]=a[j,i].
        # QAP matrices are symmetric. Let's assume symmetry for speed, but verify.
        # If not symmetric, we need a.T.
        
        # Let's assume symmetry for now as it's standard for QAPLIB.
        # If not, we can use a.T.
        
        # Build D: n x n
        # D[k, j] = b[k, p[j]] if assigned[j] else 0
        # We can do this by:
        # p_assigned = p[assigned_facilities]
        # D_temp = b[:, p_assigned]  # shape (n, A) where A is count of assigned
        # D_full = np.zeros((n, n))
        # D_full[:, assigned_facilities] = D_temp
        
        # This is O(n^2) per step. O(n^3) total. 150^3 = 3.3M. Very fast.
        
        A_count = np.sum(assigned_facilities)
        if A_count == 0:
            # First step, just pick min a[i,i] b[k,k] or similar
            # Actually, just pick the pair with min a[i,i] * b[k,k] + small penalty?
            # Or just random? No, let's use the general formula, it will be 0 + a[i,i]b[k,k].
            pass
        
        # Efficient construction of D and E
        # D[k, j] = b[k, p[j]] * assigned[j]
        # We can compute D = b @ (np.diag(p) ... no.
        
        # Let's just do it simply since n is small.
        # But inside the loop, we want to avoid Python loops.
        
        # Vectorized MC calculation:
        # 1. Get indices of assigned facilities
        assigned_idx = np.where(assigned_facilities)[0]
        
        if len(assigned_idx) > 0:
            # p_assigned = p[assigned_idx]
            # Term 1: sum_{j in assigned} a[i, j] b[k, p[j]]
            # Matrix T1[i, k] = sum_j a[i, j] b[k, p[j]]
            # Let M1 = b[:, p[assigned_idx]]  # shape (n, A)
            # Then T1 = a[unassigned_fac_idx, :] @ M1.T ?
            # a[i, j] is in a[i, :]. M1.T is (A, n).
            # (a[i, :] @ M1.T) is scalar for fixed i? No.
            # We want T1 for all i in unassigned and all k in unassigned.
            # T1[i, k] = sum_{j in assigned} a[i, j] b[k, p[j]]
            # This is (a[unassigned_fac_idx, assigned_idx] @ b[unassigned_loc_idx, p[assigned_idx]].T)
            # Let A_part = a[unassigned_fac_idx, assigned_idx]  # (U, A)
            # Let B_part = b[unassigned_loc_idx, p[assigned_idx]]  # (L, A)
            # T1 = A_part @ B_part.T  # (U, L)
            
            # Term 2: sum_{j in assigned} a[j, i] b[p[j], k]
            # Assuming symmetry a[j,i] = a[i,j]
            # T2[i, k] = sum_{j in assigned} a[i, j] b[p[j], k]
            # Let B_part2 = b[p[assigned_idx], unassigned_loc_idx]  # (A, L)
            # T2 = A_part @ B_part2  # (U, L)
            
            # Term 3: a[i, i] b[k, k]
            # This is diagonal outer product.
            # diag_a = a[unassigned_fac_idx, unassigned_fac_idx]  # (U,)
            # diag_b = b[unassigned_loc_idx, unassigned_loc_idx]  # (L,)
            # T3 = np.outer(diag_a, diag_b)
            
            A_part = a[unassigned_fac_idx, assigned_idx]
            B_part1 = b[unassigned_loc_idx, p[assigned_idx]]
            B_part2 = b[p[assigned_idx], unassigned_loc_idx]
            
            T1 = A_part @ B_part1.T
            T2 = A_part @ B_part2
            
            diag_a = np.diag(a[unassigned_fac_idx, :][:, unassigned_fac_idx])
            diag_b = np.diag(b[unassigned_loc_idx, :][:, unassigned_loc_idx])
            T3 = np.outer(diag_a, diag_b)
            
            MC = T1 + T2 + T3
            
            # Find min in MC
            min_idx = np.unravel_index(np.argmin(MC), MC.shape)
            best_i = unassigned_fac_idx[min_idx[0]]
            best_k = unassigned_loc_idx[min_idx[1]]
        else:
            # No assigned, just pick min a[i,i] b[k,k]
            # Or random
            # Let's pick the one with min flow sum and min dist sum
            a_sum = a.sum(axis=1)
            b_sum = b.sum(axis=1)
            best_i = unassigned_fac_idx[np.argmin(a_sum[unassigned_fac_idx])]
            best_k = unassigned_loc_idx[np.argmin(b_sum[unassigned_loc_idx])]
            
        p[best_i] = best_k
        assigned_facilities[best_i] = True
        assigned_locations[best_k] = True

    # If we broke out early, fill rest randomly
    if np.any(~assigned_facilities):
        rem_f = np.where(~assigned_facilities)[0]
        rem_l = np.where(~assigned_locations)[0]
        np.random.shuffle(rem_l)
        p[rem_f] = rem_l
        assigned_facilities[:] = True
        assigned_locations[:] = True

    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    best_cost = calc_cost(p)
    best_p = p.copy()
    
    # 2. Local Search: Vectorized 2-opt
    # We use a swap-based local search.
    # To speed it up, we vectorize the delta calculation for all swaps involving a fixed i?
    # Or just use a simple loop with numpy dot products.
    
    # Optimized 2-opt:
    # Precompute Bp = b[:, p]
    # Delta for swapping i and j:
    # delta = (a[i] - a[j]) dot (Bp[p[j]] - Bp[p[i]])
    
    tabu_size = 50
    tabu_list = {}
    current_step = 0
    no_improve_ls = 0
    max_no_improve_ls = 2000
    
    p = best_p.copy()
    current_cost = best_cost
    
    while time.time() < deadline:
        if no_improve_ls > max_no_improve_ls:
            # Perturb
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_ls = 0
            tabu_list.clear()
            continue
            
        Bp = b[:, p]
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Iterate i
        for i in range(n):
            if time.time() > deadline:
                break
                
            u = p[i]
            ai = a[i]
            
            # Vectorize j loop?
            # delta_j = (ai - a[j]) dot (Bp[p[j]] - Bp[u])
            # This is hard to fully vectorize because Bp[p[j]] changes with j.
            # But we can precompute Bp[u]
            Bp_u = Bp[u]
            
            # For each j > i:
            # delta = (ai - a[j]) dot (Bp[p[j]] - Bp_u)
            # = (ai - a[j]) dot Bp[p[j]] - (ai - a[j]) dot Bp_u
            
            # Let's just loop j, it's fast enough for n=150 if optimized in C/numpy?
            # Python loop is slow. 150*150/2 = 11k iterations per pass.
            # 11k * 100 passes = 1M iterations. Might be too slow in Python.
            
            # Let's try to vectorize the inner loop.
            # We need to compute dot product for each j.
            # Bp is n x n. p is n.
            # Bp[p[j]] is row p[j] of Bp.
            # Let's get all Bp[p] -> Bp_perm = b[p][:, p] ? No.
            # Bp is b[:, p]. So Bp[:, j] is b[:, p[j]].
            # We want b[p[j], :]? No, Bp[u] is b[:, u].
            # Wait, Bp = b[:, p]. So Bp[:, j] = b[:, p[j]].
            # The term is (ai - a[j]) dot (b[:, p[j]] - b[:, u])
            
            # Let's construct a matrix of deltas for all j at once?
            # delta_j = (ai - a[j]) dot b[:, p[j]] - (ai - a[j]) dot b[:, u]
            
            # Part 2: (ai - a[j]) dot b[:, u]
            # b[:, u] is a vector of size n.
            # (ai - a[j]) is a vector of size n.
            # Dot product is scalar.
            # We can compute this for all j:
            # diff_a = ai - a  # shape (n,)
            # dot_bu = diff_a @ Bp[:, u]  # shape (n,) -> this gives (ai - a[j]) dot b[:, u] for all j
            
            # Part 1: (ai - a[j]) dot b[:, p[j]]
            # This is element-wise product sum?
            # sum_k (ai[k] - a[j][k]) * b[k, p[j]]
            # = ai[k] b[k, p[j]] - a[j][k] b[k, p[j]]
            # = (ai dot b[:, p[j]]) - (a[j] dot b[:, p[j]])
            
            # Let M1 = a @ b.T  # M1[i, l] = sum_k a[i, k] b[l, k] = a[i] dot b[l, :]
            # Then ai dot b[:, p[j]] = M1[i, p[j]]
            # And a[j] dot b[:, p[j]] = M1[j, p[j]]
            
            # So Part 1 for all j:
            # M1_i_pj = M1[i, p]  # shape (n,) -> M1[i, p[j]] for all j
            # M1_j_pj = np.diag(M1)  # M1[j, j] ... wait. M1[j, p[j]] is not diag unless p[j]=j.
            # M1_j_pj = M1[np.arange(n), p]  # shape (n,)
            
            # So Part 1 = M1[i, p] - M1[np.arange(n), p]
            
            # Total delta for all j:
            # delta_all = (M1[i, p] - M1[np.arange(n), p]) - (diff_a @ Bp[:, u])
            
            diff_a = ai - a
            dot_bu = diff_a @ Bp[:, u]
            
            M1 = a @ b.T # Precompute outside loop!
            
            # Precompute M1 outside the while loop
            # M1 = a @ b.T
            
            # Inside i loop:
            # part1 = M1[i, p] - M1[np.arange(n), p]
            # delta_all = part1 - dot_bu
            
            # We need to exclude i itself and check tabu.
            # delta_all[i] is for swapping i with i (0).
            
            # Let's implement this vectorization.
            
            pass # Placeholder for logic, need to restructure to precompute M1

    # Let's restart the local search section with precomputation
    
    p = best_p.copy()
    current_cost = best_cost
    M1 = a @ b.T
    arange_n = np.arange(n)
    M1_diag_p = M1[arange_n, p] # This changes with p, so cannot precompute fully
    
    # Actually, M1[arange_n, p] is M1[j, p[j]].
    # When p changes, this vector changes.
    # But we can update it.
    # Or just recompute it? O(n).
    
    tabu_list = {}
    current_step = 0
    no_improve_ls = 0
    
    while time.time() < deadline:
        if no_improve_ls > 2000:
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_ls = 0
            tabu_list.clear()
            continue
            
        Bp = b[:, p]
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # M1_j_pj = M1[arange_n, p]
        M1_j_pj = M1[arange_n, p]
        
        for i in range(n):
            if time.time() > deadline:
                break
                
            u = p[i]
            ai = a[i]
            
            diff_a = ai - a
            dot_bu = diff_a @ Bp[:, u]
            
            part1 = M1[i, p] - M1_j_pj
            delta_all = part1 - dot_bu
            
            # Exclude i
            delta_all[i] = np.inf
            
            # Apply tabu
            # Tabu check is hard to vectorize easily with dict.
            # Skip tabu for now? Or use a simple array for tabu?
            # Tabu array of size n x n? 150x150 = 22500. Fine.
            # tabu_until[i][j]
            
            # Let's just pick min delta
            min_j = np.argmin(delta_all)
            
            # Check if tabu
            # We'll skip tabu for simplicity and rely on perturbation
            # If we want tabu, we need to update it.
            
            if delta_all[min_j] < best_delta:
                best_delta = delta_all[min_j]
                best_i = i
                best_j = min_j
                
        if best_i == -1 or best_delta >= 0:
            no_improve_ls += 1
            # Perturb if stuck
            if no_improve_ls > 50:
                for _ in range(5):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                current_cost = calc_cost(p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve_ls = 0
            continue
            
        # Apply swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_ls = 0
        else:
            no_improve_ls += 1

    return best_p.tolist()
# EVOLVE-BLOCK-END
