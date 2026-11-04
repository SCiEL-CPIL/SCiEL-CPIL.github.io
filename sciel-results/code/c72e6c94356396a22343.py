import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Check if B is symmetric
    is_sym = np.allclose(b, b.T)

    # --- Phase 0: Initialization ---
    # Use spectral embedding for a strong initial guess, with fallback
    try:
        A_sym_init = a + a.T
        B_sym_init = b + b.T

        k_eig = min(8, n - 1)
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym_init)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym_init)

        # Take eigenvectors with largest absolute eigenvalues
        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]

        V_A = eigvecs_A[:, idx_A]  # (n, k_eig)
        V_B = eigvecs_B[:, idx_B]  # (n, k_eig)

        # Build assignment cost: align spectral signatures
        C_spectral = -V_A @ V_B.T  # (n, n)

        row_ind, col_ind = linear_sum_assignment(C_spectral)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind

    except Exception:
        # Fallback to rank-based initialization
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order

    # --- Cost Calculation Helper ---
    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # --- Phase 1: Frank-Wolfe on Birkhoff Polytope ---
    # This is the "structural change": optimizing over the continuous relaxation
    # using the linearization of the quadratic objective.
    fw_deadline = start_time + 45.0

    while time.time() < fw_deadline:
        # Construct linear cost matrix C for the Frank-Wolfe step
        # The gradient of the objective f(X) = tr(A X B X^T) at X is roughly 2 * (A X B + A^T X B^T)
        # For the linear assignment subproblem, we minimize <C, X>
        B_p = b[p]  # B_p[i, k] = B[p[i], k]
        
        # Gradient direction for the QAP linearization
        # If A and B are symmetric, the gradient is 2 * A_sym @ B_p
        # More generally, the linear term to minimize is related to the gradient
        if is_sym:
            A_sym = a + a.T
            C = A_sym @ B_p
        else:
            # General case gradient: (A @ B_p + A.T @ B_p.T) ? 
            # Actually, let's use the standard Frank-Wolfe update for QAP:
            # Minimize <A, X B X^T>. Gradient G = A B X^T + A^T X B^T
            # Here X is a permutation matrix P. G = A B P^T + A^T P B^T
            # We want to find a permutation P_new that minimizes <G, P_new>
            # G[i, j] = sum_k A[i, k] B[k, p[j]] + sum_k A[k, i] B[p[k], j]
            # This is exactly: (A @ B_p) + (A.T @ B_p.T)
            C = a @ B_p + a.T @ B_p.T

        # Solve linear assignment to get the next vertex
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind

        new_cost = calc_cost(p_new)

        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improvement from global linearization. 
            # We can try a few local swaps to escape, or just break to local search.
            # Let's try a few random swaps to see if we can break the plateau
            escaped = False
            for _ in range(10):
                i, j = np.random.choice(n, 2, replace=False)
                # Quick delta check
                u, v = p[i], p[j]
                # Approximate delta
                # Instead of exact delta, just try and check cost
                p_tmp = p.copy()
                p_tmp[i], p_tmp[j] = p_tmp[j], p_tmp[i]
                c_tmp = calc_cost(p_tmp)
                if c_tmp < current_cost:
                    p = p_tmp
                    current_cost = c_tmp
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    escaped = True
                    break
            if not escaped:
                # Stuck, move to local search
                break

    # --- Phase 2: Iterated Local Search (ILS) ---
    # Refinement machinery from the inspiration: Fast vectorized local search + perturbations
    p = best_p.copy()
    current_cost = best_cost

    # Precompute A_sym for local search delta
    A_sym = a + a.T

    def local_search(p, current_cost, time_limit):
        """
        Optimized 2-opt local search for QAP using vectorized delta calculation.
        """
        n = len(p)
        p = p.copy()
        # D[i, j] = b[p[i], p[j]]
        D = b[np.ix_(p, p)].copy()
        current_cost = int(np.sum(a * D))
        
        while time.time() < time_limit:
            improved = False
            # Iterate i from 0 to n-2
            for i in range(n - 1):
                if time.time() > time_limit:
                    return p, current_cost
                
                js = np.arange(i + 1, n)
                
                # Vectorized delta calculation
                # diff_A = A_sym[i] - A_sym[js]
                # diff_D = D[js] - D[i]
                # S = sum(diff_A * diff_D, axis=1)
                
                diff_A = A_sym[i] - A_sym[js]
                diff_D = D[js] - D[i]
                
                S = np.sum(diff_A * diff_D, axis=1)
                
                # Corrections for diagonal and cross terms
                # term_i (k=i part)
                coeff_i = A_sym[i, i] - A_sym[js, i]
                term_i = coeff_i * (D[js, i] - D[i, i])
                
                # term_j (k=j part)
                coeff_j = A_sym[i, js] - np.diag(A_sym[js])
                term_j = coeff_j * (D[js, js] - D[i, js])
                
                # Direct terms
                diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
                cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
                
                deltas = S - term_i - term_j + diag_term + cross_term
                
                min_idx = np.argmin(deltas)
                min_delta = deltas[min_idx]
                
                if min_delta < 0:
                    j = js[min_idx]
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    current_cost += int(min_delta)
                    improved = True
            
            if not improved:
                break
        
        return p, current_cost

    def perturb(p, k):
        """Randomly swap k pairs of positions."""
        p = p.copy()
        n = len(p)
        for _ in range(k):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        return p

    # ILS Loop
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() < deadline - 5.0:
        # Adaptive perturbation
        if consecutive_no_improve > 5:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = perturb(best_p, perturb_k)
        
        # Run local search on the perturbed solution
        # Allocate time for this LS run: give it a chunk of remaining time
        elapsed = time.time() - start_time
        remaining = deadline - 5.0 - elapsed
        if remaining < 2.0:
            break
            
        # Give LS a fraction of the remaining time, but at least 1s
        ls_time = max(1.0, remaining * 0.8)
        ls_end_time = time.time() + ls_time
        
        p_ls, cost_ls = local_search(p_pert, calc_cost(p_pert), ls_end_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    return best_p.tolist()
