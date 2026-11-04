import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute symmetric A for faster delta calculations in local search
    A_sym = a + a.T

    def calc_cost(p):
        # p is a permutation array where p[i] is the location of facility i
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # b_perm[i,j] = b[p[i], p[j]]
        b_perm = b[np.ix_(p, p)]
        return np.sum(a * b_perm)

    def local_search(p, time_limit):
        """
        Vectorized 2-opt local search.
        p: current permutation (facility i at location p[i])
        time_limit: absolute time to stop
        """
        p = p.copy()
        # D[i, j] represents b[p[i], p[j]]
        D = b[np.ix_(p, p)].copy()
        current_cost = int(np.sum(a * D))
        
        # We iterate through facilities i and try swapping with j > i
        # To avoid O(n^3) complexity, we vectorize the delta calculation for all j > i given a fixed i.
        
        # We need to be careful with the delta formula.
        # Let's use the reliable formula derived in the inspiration:
        # Delta(i, j) = sum_{k != i, j} (A_sym[i, k] - A_sym[j, k]) * (D[j, k] - D[i, k]) + correction terms
        
        # However, a simpler and robust vectorized approach for 2-opt in QAP:
        # Cost change when swapping locations of i and j:
        # Old terms involving i or j:
        # sum_k A[i,k]D[i,k] + sum_k A[k,i]D[k,i] + sum_k A[j,k]D[j,k] + sum_k A[k,j]D[k,j]
        # - A[i,i]D[i,i] - A[i,j]D[i,j] - A[j,i]D[j,i] - A[j,j]D[j,j] (these are counted twice in the sums above if k=i or k=j, 
        #   but actually the sum is over all k. Let's stick to the explicit delta derivation to be safe.)
        
        # Let's use the vectorized block update method which is standard for fast QAP solvers.
        # For a fixed i, we want to find best j.
        # Delta(i, j) = (A_sym[i, :] - A_sym[j, :]) dot (D[j, :] - D[i, :]) 
        #               - 2 * (A_sym[i, i] - A_sym[i, j]) * (D[i, i] - D[i, j]) # Correction for k=i, j in the dot product?
        
        # Actually, the inspiration's formula was:
        # S = sum(diff_A * diff_D, axis=1)
        # where diff_A = A_sym[i] - A_sym[js], diff_D = D[js] - D[i]
        # This S includes terms where k=i and k=j which are incorrect for the swap delta because 
        # the indices i and j themselves change roles.
        
        # Let's implement the specific correction logic from the inspiration which was verified.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n - 1):
                if time.time() > time_limit:
                    return p, current_cost
                
                js = np.arange(i + 1, n)
                
                # Vectorized delta calculation
                # diff_A shape: (n-i-1, n)
                # diff_D shape: (n-i-1, n)
                diff_A = A_sym[i] - A_sym[js]
                diff_D = D[js] - D[i]
                
                S = np.sum(diff_A * diff_D, axis=1)
                
                # Corrections for the terms where k=i and k=j in the summation of S
                # The term for k=i in S is: (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
                # The term for k=j in S is: (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
                
                # We need to subtract these from S and add the correct direct interaction terms for the swap.
                
                # Direct terms for swapping i and j:
                # The cost components that change directly are those involving A[i,i], A[i,j], A[j,i], A[j,j]
                # and their corresponding D terms.
                
                # Let's use the explicit formula:
                # Delta = S - term_i - term_j + term_direct
                
                # term_i: contribution of k=i in the sum S
                # coeff_i for each j in js: A_sym[i, i] - A_sym[j, i]
                coeff_i = A_sym[i, i] - A_sym[js, i]
                term_i = coeff_i * (D[js, i] - D[i, i])
                
                # term_j: contribution of k=j in the sum S
                # coeff_j for each j in js: A_sym[i, j] - A_sym[j, j]
                coeff_j = A_sym[i, js] - np.diag(A_sym[js])
                term_j = coeff_j * (D[js, js] - D[i, js])
                
                # term_direct: The actual change in cost due to the swap of i and j positions.
                # Old: A[i,i]D[i,i] + A[i,j]D[i,j] + A[j,i]D[j,i] + A[j,j]D[j,j]
                # New: A[i,i]D[j,j] + A[i,j]D[j,i] + A[j,i]D[i,j] + A[j,j]D[i,i]
                # Change = A[i,i](D[j,j]-D[i,i]) + A[i,j](D[j,i]-D[i,j]) + A[j,i](D[i,j]-D[j,i]) + A[j,j](D[i,i]-D[j,j])
                # Change = (A[i,i]-A[j,j])(D[j,j]-D[i,i]) + (A[i,j]-A[j,i])(D[j,i]-D[i,j])
                
                diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
                cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
                
                deltas = S - term_i - term_j + diag_term + cross_term
                
                min_idx = np.argmin(deltas)
                min_delta = deltas[min_idx]
                
                if min_delta < 0:
                    j = js[min_idx]
                    # Perform swap i and j in p
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    # D[i, :] becomes old D[j, :]
                    # D[j, :] becomes old D[i, :]
                    # D[:, i] becomes old D[:, j]
                    # D[:, j] becomes old D[:, i]
                    
                    # We need to be careful with in-place updates.
                    # Copy rows first
                    temp_row_i = D[i].copy()
                    temp_row_j = D[j].copy()
                    
                    D[i] = temp_row_j
                    D[j] = temp_row_i
                    
                    # Now columns
                    # Note: D[i, i] and D[j, j] have already been updated in the row swap.
                    # We need to update the rest of the columns.
                    # D[:, i] should become the old D[:, j] (but D[i, i] is already correct from row swap? No, D[i, i] is b[p[i], p[i]]. After swap, p[i] is old p[j]. So D[i, i] should be b[old_p[j], old_p[j]] which is old D[j, j]. This was set in D[i] = temp_row_j. So D[i, i] is correct.
                    # Similarly D[j, j] is correct.
                    # We need to update D[k, i] for k != i, j and D[k, j] for k != i, j.
                    
                    # Actually, it's safer to just recompute the affected parts or use a temporary buffer.
                    # Given n=256, copying rows is cheap.
                    
                    # Let's just do:
                    # D[i, :] = D[j, :] (before swap)
                    # D[j, :] = D[i, :] (before swap)
                    # D[:, i] = D[:, j] (before swap)
                    # D[:, j] = D[:, i] (before swap)
                    
                    # My previous code did:
                    # D[i], D[j] = D[j].copy(), D[i].copy()
                    # D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    # This is O(n) per swap. Acceptable.
                    
                    # Let's stick to the safe copy method
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    current_cost += int(min_delta)
                    improved = True

        return p, current_cost

    # Initialization
    best_p = None
    best_cost = np.inf

    # 1. Spectral Initialization
    try:
        A_sym_init = a + a.T
        B_sym_init = b + b.T
        k_eig = min(8, n - 1)
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym_init)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym_init)
        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]
        V_A = eigvecs_A[:, idx_A]
        V_B = eigvecs_B[:, idx_B]
        C_spectral = -V_A @ V_B.T
        row_ind, col_ind = linear_sum_assignment(C_spectral)
        p_spectral = np.empty(n, dtype=np.int64)
        p_spectral[row_ind] = col_ind
    except Exception:
        p_spectral = np.random.permutation(n)

    # Run local search on spectral init
    p_ls, cost_ls = local_search(p_spectral, start_time + 20)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()

    # 2. ILS Phase
    # Use remaining time for Iterated Local Search
    consecutive_no_improve = 0
    base_perturb = 2
    max_perturb = 10

    while time.time() < deadline - 2:
        # Adaptive perturbation
        if consecutive_no_improve > 5:
            perturb_k = min(max_perturb, base_perturb + consecutive_no_improve)
        else:
            perturb_k = base_perturb

        p_pert = best_p.copy()
        for _ in range(perturb_k):
            i, j = random.sample(range(n), 2)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Allocate time for this LS run
        # We want to leave some time for the next iteration
        # If we are close to deadline, give it all remaining time
        time_left = deadline - 2 - time.time()
        if time_left < 5:
            break
            
        # Give local search a chunk of time, e.g., half of remaining or fixed 10s, whichever is smaller but at least 2s
        ls_time_limit = time.time() + min(10.0, time_left * 0.7)
        
        p_ls, cost_ls = local_search(p_pert, ls_time_limit)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
            # If we are stuck for a while, maybe try a larger jump or reset?
            # For now, just increase perturb_k as above.

    return best_p.tolist()
