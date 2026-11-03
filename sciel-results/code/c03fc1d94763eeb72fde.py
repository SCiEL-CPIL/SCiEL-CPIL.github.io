"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # 1. Initial Solution: Rank Matching
    # Assign facilities with high flow sums to locations with low distance sums
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    # Calculate initial cost
    def calculate_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    best_p = p.copy()
    best_cost = calculate_cost(best_p)

    # Precompute symmetric parts for speed if matrices are symmetric (QAPLIB standard)
    # Check symmetry using exact integer comparison
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    if a_sym and b_sym:
        # Use efficient delta calculation
        # Precompute diagonal terms
        a_diag = np.diag(a)
        b_diag = np.diag(b)
        
        no_improvement_count = 0
        max_no_improvement = 5000
        
        while time.time() - start_time < time_limit:
            improved = False
            no_improvement_count = 0
            
            # Iterate through all pairs
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                u = p[i]
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit:
                        break
                    
                    v = p[j]
                    
                    # Calculate Delta Cost
                    # Delta = (A_ii - A_jj)(B_vv - B_uu) + 2 * sum_{m != i,j} (A_im - A_jm) * (B_v,p[m] - B_u,p[m])
                    
                    # Diagonal part
                    delta = (a_diag[i] - a_diag[j]) * (b_diag[v] - b_diag[u])
                    
                    # Off-diagonal part
                    # We need sum over m != i, j of (A[i,m] - A[j,m]) * (B[v, p[m]] - B[u, p[m]])
                    # Let diff_A = A[i, :] - A[j, :]
                    # Let diff_B_loc = B[v, p] - B[u, p]  (vector over locations)
                    # We need dot product of diff_A and diff_B_loc, excluding i and j.
                    
                    # Vectorized approach:
                    # diff_A = a[i, :] - a[j, :]
                    # diff_B = b[v, p] - b[u, p]
                    # dot = np.dot(diff_A, diff_B)
                    # Subtract contributions from m=i and m=j
                    # m=i term: diff_A[i] * diff_B[i]
                    # m=j term: diff_A[j] * diff_B[j]
                    
                    diff_A = a[i, :] - a[j, :]
                    diff_B = b[v, p] - b[u, p]
                    
                    total_dot = np.dot(diff_A, diff_B)
                    
                    # Exclude i and j
                    # p[i] is u, p[j] is v
                    # diff_B[i] is B[v, u] - B[u, u]
                    # diff_B[j] is B[v, v] - B[u, v]
                    
                    exclude_i = diff_A[i] * diff_B[i]
                    exclude_j = diff_A[j] * diff_B[j]
                    
                    off_diag_sum = total_dot - exclude_i - exclude_j
                    delta += 2 * off_diag_sum
                    
                    if delta < 0:
                        # Perform swap
                        p[i], p[j] = v, u
                        best_cost += int(delta)
                        best_p = p.copy()
                        improved = True
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
            if not improved:
                break

    else:
        # Fallback for non-symmetric: return initial solution
        pass

    # Final safety check: ensure best_p is a valid permutation
    # If for any reason it's corrupted, fall back to the initial valid solution
    # (though swaps should preserve validity)
    # We check if sorted(best_p) is 0..n-1
    # This is O(n log n), cheap for n=256
    if sorted(best_p.tolist()) != list(range(n)):
        # Fallback to identity permutation or initial best_p if it was valid
        # Since best_p was initialized from a valid construction, it should be valid.
        # If it's invalid here, something went wrong with the swap logic.
        # Return a known valid permutation: the initial one (re-constructed)
        # Or simply identity, which is always valid.
        # Let's return the initial best_p if it is valid, else identity.
        if sorted(best_p.tolist()) != list(range(n)):
             # Re-initialize a valid permutation
             valid_p = np.arange(n, dtype=np.int64)
             return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
