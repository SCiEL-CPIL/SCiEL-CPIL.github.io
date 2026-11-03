"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    # Safety margin: 150s limit, return by 145s
    time_limit = 145.0 

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    # Precompute row sums for potential fallback or initialization
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)

    # 1. Linear Assignment Initialization (Marginal Cost Heuristic)
    # The QAP objective is sum_{i,j} a[i,j] b[p[i], p[j]].
    # We can approximate this by a linear assignment problem.
    # One common heuristic is to define a cost matrix C[i, j] = sum_k a[i,k] b[j,k].
    # This represents the "average" or "total" interaction cost if facility i is at location j,
    # assuming the rest of the facilities are distributed uniformly or according to the structure of b.
    # Specifically, if we assume the assignment of other facilities is fixed or random,
    # the marginal contribution of placing i at j is roughly related to the sum of flows from i 
    # weighted by the distances from j to the locations of those facilities.
    # A tighter linearization often used:
    # C[i, j] = sum_k a[i, k] * b[j, k]
    # This is O(n^3) to compute, but n=150, so 150^3 = 3.375M operations. Very fast.
    
    # Compute cost matrix C
    # C[i, j] = dot(a[i, :], b[j, :])
    # C = a @ b.T
    C = a @ b.T
    
    # Solve the Linear Assignment Problem (Hungarian Algorithm)
    # We want to minimize the sum of C[i, p[i]]
    row_ind, col_ind = linear_sum_assignment(C)
    
    # row_ind is the facility index, col_ind is the location index
    # p[facility] = location
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    # Calculate initial cost
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # We can compute this as sum(a * b[np.ix_(p, p)])
    B_p = b[np.ix_(p, p)]
    cost = int(np.sum(a * B_p))
    
    best_cost = cost
    best_p = p.copy()

    # 2. Deterministic 2-Opt Local Search with Vectorization
    # We perform full scans of all pairs (i, j) to find the best improvement.
    # To make this fast enough, we vectorize the delta calculation for a fixed i against all j.
    
    # Delta for swapping i and j:
    # Let u = p[i], v = p[j].
    # New cost - Old cost = 
    #   (sum_k a[i,k] b[v, p'[k]] + sum_k a[j,k] b[u, p'[k]] + sum_{k!=i,j} a[k,m] b[p[k], p'[m]] terms)
    #   - (sum_k a[i,k] b[u, p[k]] + sum_k a[j,k] b[v, p[k]] + sum_{k!=i,j} a[k,m] b[p[k], p[m]] terms)
    
    # Let's derive the delta more carefully.
    # Cost(p) = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Let S = sum_{k,l} a[k,l] b[p[k], p[l]]
    # Swap i and j. Let p' be the new permutation.
    # p'[i] = v, p'[j] = u, p'[k] = p[k] for k!=i,j.
    
    # Terms involving i or j in the sum:
    # 1. i,i: a[i,i] b[v,v] (was a[i,i] b[u,u])
    # 2. j,j: a[j,j] b[u,u] (was a[j,j] b[v,v])
    # 3. i,j: a[i,j] b[v,u] (was a[i,j] b[u,v]) -> Same since b is symmetric
    # 4. j,i: a[j,i] b[u,v] (was a[j,i] b[v,u]) -> Same since a,b symmetric
    # 5. i,k (k!=i,j): a[i,k] b[v, p[k]] (was a[i,k] b[u, p[k]])
    # 6. j,k (k!=i,j): a[j,k] b[u, p[k]] (was a[j,k] b[v, p[k]])
    # 7. k,i (k!=i,j): a[k,i] b[p[k], v] (was a[k,i] b[p[k], u])
    # 8. k,j (k!=i,j): a[k,j] b[p[k], u] (was a[k,j] b[p[k], v])
    
    # Delta = a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) 
    #        + sum_{k!=i,j} [ a[i,k](b[v, p[k]] - b[u, p[k]]) 
    #                        + a[j,k](b[u, p[k]] - b[v, p[k]])
    #                        + a[k,i](b[p[k], v] - b[p[k], u])
    #                        + a[k,j](b[p[k], u] - b[p[k], v]) ]
    
    # Since a and b are symmetric:
    # a[k,i] = a[i,k], a[k,j] = a[j,k]
    # b[p[k], v] = b[v, p[k]], b[p[k], u] = b[u, p[k]]
    
    # Group terms for k:
    # Term_k = a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[j,k] (b[u, p[k]] - b[v, p[k]])
    #        + a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[j,k] (b[u, p[k]] - b[v, p[k]])
    #        = 2 * [ a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[j,k] (b[u, p[k]] - b[v, p[k]]) ]
    #        = 2 * (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    
    # So Delta = (a[i,i] - a[j,j]) (b[v,v] - b[u,u]) + 2 * sum_{k!=i,j} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
    
    # We can compute the sum over k!=i,j efficiently.
    # Let diff_a = a[i, :] - a[j, :]
    # Let diff_b = b[v, :] - b[u, :]
    # We need sum_{k!=i,j} diff_a[k] * diff_b[p[k]]
    
    # Vectorized approach for a fixed i:
    # We want to find j that minimizes Delta(i, j).
    # Delta(i, j) = (a[i,i] - a[j,j]) (b[p[i], p[i]] - b[p[j], p[j]]) + 2 * (sum_{k} (a[i,k] - a[j,k]) (b[p[i], p[k]] - b[p[j], p[k]]) - term_i - term_j)
    
    # This is still complex to fully vectorize over j for a fixed i because p[j] changes the indexing in diff_b.
    # However, we can iterate j in a loop for each i, but that's O(n^2) loops with O(n) work inside -> O(n^3) total.
    # 150^3 = 3.375M. In Python, 3.3M iterations is ~1-3 seconds. 
    # If we do this for, say, 10 full scans, it's 10-30 seconds. This is feasible.
    
    # Let's optimize the inner loop calculation.
    # For a fixed i, u = p[i].
    # We iterate j from 0 to n-1 (j != i).
    # v = p[j].
    # We need:
    # S_j = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[u, p[k]] - b[v, p[k]])
    #     = sum_{k != i, j} a[i,k] (b[u, p[k]] - b[v, p[k]]) - sum_{k != i, j} a[j,k] (b[u, p[k]] - b[v, p[k]])
    #     = sum_{k != i, j} a[i,k] b[u, p[k]] - sum_{k != i, j} a[i,k] b[v, p[k]] 
    #       - sum_{k != i, j} a[j,k] b[u, p[k]] + sum_{k != i, j} a[j,k] b[v, p[k]]
    
    # Let's precompute for the current p:
    # B_p = b[np.ix_(p, p)] is the permuted distance matrix.
    # B_p[i, k] = b[p[i], p[k]]
    
    # Let's recompute B_p after each swap.
    # B_p = b[np.ix_(p, p)]
    # This is O(n^2) = 22,500 operations. Very fast.
    
    # Now, for a fixed i, u = p[i]:
    # Row u of B_p is B_p[i, :] = [b[p[i], p[0]], ..., b[p[i], p[n-1]]]
    # Let row_u = B_p[i, :]
    
    # For a candidate j, v = p[j]:
    # Row v of B_p is B_p[j, :]
    # Let row_v = B_p[j, :]
    
    # We need S_j = sum_{k != i, j} (a[i,k] - a[j,k]) * (row_u[k] - row_v[k])
    # Note: row_u[k] is b[p[i], p[k]]. row_v[k] is b[p[j], p[k]].
    
    # Let's expand S_j:
    # S_j = sum_{k != i, j} a[i,k] row_u[k] - sum_{k != i, j} a[i,k] row_v[k] 
    #       - sum_{k != i, j} a[j,k] row_u[k] + sum_{k != i, j} a[j,k] row_v[k]
    
    # Let's define:
    # A_i = a[i, :]
    # A_j = a[j, :]
    # R_u = row_u
    # R_v = row_v
    
    # Sum1 = dot(A_i, R_u) - A_i[i]*R_u[i] - A_i[j]*R_u[j]  (remove k=i, j)
    # Sum2 = dot(A_i, R_v) - A_i[i]*R_v[i] - A_i[j]*R_v[j]
    # Sum3 = dot(A_j, R_u) - A_j[i]*R_u[i] - A_j[j]*R_u[j]
    # Sum4 = dot(A_j, R_v) - A_j[i]*R_v[i] - A_j[j]*R_v[j]
    
    # S_j = Sum1 - Sum2 - Sum3 + Sum4
    
    # Delta(i, j) = (a[i,i] - a[j,j]) * (b[u,u] - b[v,v]) + 2 * S_j
    # Note: b[u,u] = B_p[i,i], b[v,v] = B_p[j,j]
    
    # We can precompute dot products for all i, j?
    # dot(A_i, R_u) where R_u = B_p[i, :]. So dot(a[i, :], B_p[i, :])
    # This is the diagonal of (a * B_p)? No.
    # Let M = a * B_p.
    # dot(A_i, R_u) = sum_k a[i,k] B_p[i,k] = M[i, :].sum()
    # Let S_i = M[i, :].sum()
    
    # dot(A_i, R_v) = sum_k a[i,k] B_p[j,k]. This is not simply S_i.
    # This term depends on j.
    
    # Let's look at the complexity again.
    # If we loop i (n times) and inside loop j (n times), and calculate S_j in O(1) using precomputed sums?
    # S_j involves dot(A_i, R_v) which is dot(a[i, :], B_p[j, :]).
    # Let D = a @ B_p.
    # D[i, j] = sum_k a[i,k] B_p[j,k] = sum_k a[i,k] b[p[j], p[k]]
    # So dot(A_i, R_v) = D[i, j].
    
    # Similarly:
    # Sum1 = S_i - a[i,i] B_p[i,i] - a[i,j] B_p[i,j]
    # Sum2 = D[i, j] - a[i,i] B_p[j,i] - a[i,j] B_p[j,j]
    #        Note B_p[j,i] = B_p[i,j]
    # Sum3 = D[j, i] - a[j,i] B_p[i,j] - a[j,j] B_p[i,i]
    #        Note D[j, i] = sum_k a[j,k] B_p[i,k] = sum_k a[j,k] b[p[i], p[k]]
    # Sum4 = S_j - a[j,i] B_p[j,i] - a[j,j] B_p[j,j]
    
    # So S_j = (S_i - a[i,i]B_ii - a[i,j]B_ij) 
    #          - (D_ij - a[i,i]B_ij - a[i,j]B_jj)
    #          - (D_ji - a[i,j]B_ij - a[j,j]B_ii)
    #          + (S_j - a[i,j]B_ij - a[j,j]B_jj)
    
    # Let's simplify S_j:
    # S_j = S_i - S_j  (Wait, S_j appears on both sides? No, S_j is the sum we want to calculate. Let's call the precomputed row sum R_i)
    # Let R_i = sum_k a[i,k] B_p[i,k] = S_i.
    # Let R_j = sum_k a[j,k] B_p[j,k] = S_j.
    
    # S_j (the interaction sum) = 
    #   (R_i - a_ii B_ii - a_ij B_ij) 
    #   - (D_ij - a_ii B_ij - a_ij B_jj)
    #   - (D_ji - a_ij B_ij - a_jj B_ii)
    #   + (R_j - a_ij B_ij - a_jj B_jj)
    
    # Group terms:
    # R_i + R_j
    # - a_ii B_ii + a_jj B_ii = (a_jj - a_ii) B_ii
    # - a_ij B_ij + a_ij B_jj + a_ij B_ij - a_ij B_ij - a_jj B_jj = a_ij B_jj - a_jj B_jj - a_ij B_ij + a_ij B_ij?
    # Let's be precise.
    # Terms with B_ii: -a_ii + a_jj
    # Terms with B_jj: +a_ij - a_jj
    # Terms with B_ij: -a_ij + a_ii + a_ij - a_ij = a_ii - a_ij
    # Terms with D: -D_ij - D_ji
    
    # So Interaction_Sum = R_i + R_j + (a_jj - a_ii) B_ii + (a_ij - a_jj) B_jj + (a_ii - a_ij) B_ij - D_ij - D_ji
    
    # Total Delta = (a_ii - a_jj) B_ii + (a_jj - a_ii) B_jj + 2 * Interaction_Sum
    # Wait, the diagonal part in Delta was (a_ii - a_jj) (B_ii - B_jj) = (a_ii - a_jj) B_ii - (a_ii - a_jj) B_jj = (a_ii - a_jj) B_ii + (a_jj - a_ii) B_jj.
    # This matches the first two terms of Delta.
    
    # So Delta(i, j) = (a_ii - a_jj) (B_ii - B_jj) + 2 * [ R_i + R_j + (a_jj - a_ii) B_ii + (a_ij - a_jj) B_jj + (a_ii - a_ij) B_ij - D_ij - D_ji ]
    
    # Let's simplify the bracket:
    # Bracket = R_i + R_j - D_ij - D_ji + (a_jj - a_ii) B_ii + (a_ii - a_jj) B_ii  (Cancel? No)
    # Wait, (a_jj - a_ii) B_ii from interaction + (a_ii - a_jj) B_ii from diagonal?
    # Diagonal part: (a_ii - a_jj) B_ii
    # Interaction part has: (a_jj - a_ii) B_ii
    # Sum: 0.
    # Similarly for B_jj:
    # Diagonal: (a_jj - a_ii) B_jj
    # Interaction: (a_ij - a_jj) B_jj
    # Sum: (a_jj - a_ii + a_ij - a_jj) B_jj = (a_ij - a_ii) B_jj
    
    # So Delta(i, j) = 2 * [ R_i + R_j - D_ij - D_ji ] + (a_ij - a_ii) B_jj + (a_ii - a_ij) B_ij
    #                 = 2 * (R_i + R_j - D_ij - D_ji) + (a_ii - a_ij) (B_ij - B_jj)
    
    # Let's verify this formula.
    # Delta = 2 * (R_i + R_j - D_ij - D_ji) + (a_ii - a_ij) * (B_ij - B_jj)
    
    # Precomputations for current p:
    # B_p = b[np.ix_(p, p)]
    # D = a @ B_p
    # R = (a * B_p).sum(axis=1)
    
    # Now, for each i, we can compute Delta(i, j) for all j != i in O(n) vectorized operations.
    # For fixed i:
    # R_i is scalar.
    # B_ii = B_p[i, i]
    # a_ii = a[i, i]
    # a_i = a[i, :]
    # B_i = B_p[i, :]
    # D_i = D[i, :]
    # D_col_i = D[:, i]
    
    # We want to find j that minimizes Delta(i, j).
    # Delta(i, j) = 2 * (R_i + R[j] - D[i, j] - D[j, i]) + (a[i, i] - a[i, j]) * (B_p[i, j] - B_p[j, j])
    
    # Vectorize over j:
    # term1 = 2 * (R_i + R - D[i, :] - D[:, i])
    # term2 = (a[i, i] - a[i, :]) * (B_p[i, :] - np.diag(B_p))
    # Delta_vec = term1 + term2
    
    # Set Delta_vec[i] = inf (don't swap with self)
    # Find j_min = argmin(Delta_vec)
    # If Delta_vec[j_min] < 0, swap i and j_min.
    
    # Update p, B_p, D, R.
    # Swapping i and j in p changes B_p by swapping rows/cols i and j.
    # B_p_new = B_p with rows/cols i, j swapped.
    # We can just recompute B_p = b[np.ix_(p, p)] which is O(n^2).
    # D = a @ B_p is O(n^3) = 3.3M ops.
    # R = (a * B_p).sum(axis=1) is O(n^2).
    
    # Total cost per swap: O(n^3) for D update.
    # If we do 100 swaps, 330M ops. In NumPy, 330M float ops is ~1-3 seconds.
    # This allows for many full scans.
    
    # Let's implement the loop.
    
    def compute_state(p):
        B_p = b[np.ix_(p, p)]
        D = a @ B_p
        R = (a * B_p).sum(axis=1)
        return B_p, D, R

    B_p, D, R = compute_state(p)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Full scan: try to find the best swap (i, j) globally.
        # To do this, we can iterate i, compute Delta(i, :) for all j, and track the global min.
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            # Vectorized calculation of Delta(i, j) for all j
            # R_i = R[i]
            # D_i_row = D[i, :]
            # D_i_col = D[:, i]
            # a_i = a[i, :]
            # B_i_row = B_p[i, :]
            # diag_B = np.diag(B_p)
            
            term1 = 2.0 * (R[i] + R - D[i, :] - D[:, i])
            term2 = (a[i, i] - a[i, :]) * (B_p[i, :] - np.diag(B_p))
            
            delta_vec = term1 + term2
            
            # Exclude i
            delta_vec[i] = np.inf
            
            min_delta_in_row = np.min(delta_vec)
            if min_delta_in_row < best_delta:
                best_delta = min_delta_in_row
                best_i = i
                best_j = np.argmin(delta_vec)
        
        if best_delta < 0:
            # Apply swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            cost += int(best_delta)
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
            
            # Update state
            B_p, D, R = compute_state(p)
            improved = True
        else:
            # No improvement found in full scan
            improved = False

    return best_p.tolist()

# EVOLVE-BLOCK-END
