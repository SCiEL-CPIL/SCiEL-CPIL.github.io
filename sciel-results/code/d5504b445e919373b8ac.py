"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _compute_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    # p[i] = location of facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can compute this as: for each i, sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b[p[i], :]) but that's not quite right
    # Actually: sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Let's use: for each pair (i,j), a[i,j] * b[p[i], p[j]]
    # This is equivalent to: trace(A * B_perm) where B_perm[i,j] = b[p[i], p[j]]
    # Or: sum over all i,j of a[i,j] * b[p[i], p[j]]
    
    # Efficient computation:
    # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # For fixed i, sum_j a[i,j] * b[p[i], p[j]] = (a[i,:] * b[p[i], :]).sum()
    # But we can vectorize:
    # Create B_p where B_p[i, j] = b[p[i], p[j]]
    B_p = b[np.ix_(p, p)]
    return int(np.sum(a * B_p))


def _solve_assignment(cost_matrix):
    """Solve the linear assignment problem."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(len(cost_matrix), dtype=np.int64)
    p[row_ind] = col_ind
    return p


def solve(a, b):
    n = len(a)
    
    # Deadline guard: 150s wall-clock, use 145s as safety margin
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial solution: rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute some useful quantities
    # a_row_sums[i] = sum_j a[i,j]
    a_row_sums = a.sum(axis=1)
    
    # Iterative refinement using linearized assignment
    # At each step, we fix the current permutation and compute a linear cost
    # for reassigning facilities, then solve the assignment problem.
    
    # The linearized cost for assigning facility i to location j, given current
    # permutation p, is:
    # C[i,j] = sum_k a[i,k] * b[j, p[k]]  (interaction with all other facilities)
    #        + sum_k a[k,i] * b[p[k], j]  (interaction where i is the second index)
    # Note: the term a[i,i]*b[j,j] is included in both, but that's fine.
    # Actually, let's be more careful.
    # Total cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # For a fixed assignment of all facilities except i, the cost contribution
    # involving facility i at location j is:
    #   sum_{k != i} a[i,k] * b[j, p[k]] + sum_{k != i} a[k,i] * b[p[k], j] + a[i,i]*b[j,j]
    # But in the linear assignment, we treat each facility independently, so:
    # C[i,j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
    # The a[i,i]*b[j,j] term appears in both sums when k=i, so it's counted twice.
    # We should subtract it once: C[i,j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j] - a[i,i]*b[j,j]
    
    # For efficiency, precompute:
    # B_p = b[np.ix_(p, p)]  # B_p[i,k] = b[p[i], p[k]]
    # Then sum_k a[i,k] * b[j, p[k]] = sum_k a[i,k] * B_p[i,k]... no wait.
    # b[j, p[k]] is not directly B_p.
    # Let's define: for each k, p[k] is the location of facility k.
    # b[j, p[k]] = b[j, p[k]]
    # So sum_k a[i,k] * b[j, p[k]] = (a[i,:] @ b[:, p])  ... no.
    # Actually: sum_k a[i,k] * b[j, p[k]]
    # Let's create a matrix M1 where M1[i,j] = sum_k a[i,k] * b[j, p[k]]
    # M1[i,j] = sum_k a[i,k] * b[j, p[k]]
    # = (a * b[:, p][:, None]).sum(axis=2)  ... this is getting complicated.
    
    # Alternative approach: use the fact that
    # sum_k a[i,k] * b[j, p[k]] = sum_k a[i,k] * b[j, p[k]]
    # Let B_perm = b[np.ix_(p, p)] so B_perm[k, m] = b[p[k], p[m]]
    # That's not what we need. We need b[j, p[k]] for arbitrary j.
    
    # Let's think differently. Define:
    # S1[i,j] = sum_k a[i,k] * b[j, p[k]]
    # S2[i,j] = sum_k a[k,i] * b[p[k], j]
    
    # S1[i,j] = sum_k a[i,k] * b[j, p[k]]
    # For fixed j, b[j, p[k]] for all k is a vector: b[j, p]
    # So S1[i,j] = a[i,:] @ b[j, p]  (dot product of a[i,:] and b[j, p])
    # So S1 = a @ b[:, p]  ... wait, dimensions.
    # a is (n,n), b[:, p] is (n,n) where column k of b[:, p] is b[:, p[k]]
    # So (a @ b[:, p])[i,j] = sum_k a[i,k] * b[:, p][k,j] = sum_k a[i,k] * b[p[k], j]
    # That's S2, not S1!
    
    # Let me redo:
    # S1[i,j] = sum_k a[i,k] * b[j, p[k]]
    # b[j, p[k]] for all k: this is b[j, p] which is a 1D array of length n.
    # So S1[i,j] = dot(a[i,:], b[j, p])
    # S1 = a @ b[:, p].T  ... let's check.
    # (a @ b[:, p].T)[i,j] = sum_k a[i,k] * b[:, p].T[k,j] = sum_k a[i,k] * b[:, p][j,k] = sum_k a[i,k] * b[j, p[k]]
    # Yes! S1 = a @ b[:, p].T
    
    # S2[i,j] = sum_k a[k,i] * b[p[k], j]
    # b[p[k], j] for all k: this is b[p, j] which is a 1D array of length n.
    # S2[i,j] = dot(a[:,i], b[p, j])
    # S2 = a.T @ b[p, :].T  ... let's check.
    # (a.T @ b[p, :].T)[i,j] = sum_k a.T[i,k] * b[p, :].T[k,j] = sum_k a[k,i] * b[p, :][j,k] = sum_k a[k,i] * b[j, p[k]]
    # That's not right either. Let me be more careful.
    
    # b[p, :] is a matrix where row k is b[p[k], :], so b[p, :][k, j] = b[p[k], j]
    # S2[i,j] = sum_k a[k,i] * b[p[k], j] = sum_k a[k,i] * b[p, :][k, j]
    # = (a.T @ b[p, :])[i, j]  ... check:
    # (a.T @ b[p, :])[i,j] = sum_k a.T[i,k] * b[p, :][k,j] = sum_k a[k,i] * b[p[k], j]
    # Yes! S2 = a.T @ b[p, :]
    
    # And the diagonal correction: a[i,i]*b[j,j]
    # C[i,j] = S1[i,j] + S2[i,j] - a[i,i]*b[j,j]
    
    # Let's implement this iteratively.
    
    while time.time() < deadline:
        # Compute linearized cost matrix
        # S1[i,j] = sum_k a[i,k] * b[j, p[k]] = (a @ b[:, p].T)[i,j]
        # S2[i,j] = sum_k a[k,i] * b[p[k], j] = (a.T @ b[p, :])[i,j]
        
        b_col_p = b[:, p]  # shape (n, n), b_col_p[j, k] = b[j, p[k]]
        b_row_p = b[p, :]  # shape (n, n), b_row_p[k, j] = b[p[k], j]
        
        S1 = a @ b_col_p.T  # S1[i,j] = sum_k a[i,k] * b_col_p.T[k,j] = sum_k a[i,k] * b_col_p[j,k] = sum_k a[i,k] * b[j, p[k]]
        S2 = a.T @ b_row_p  # S2[i,j] = sum_k a.T[i,k] * b_row_p[k,j] = sum_k a[k,i] * b[p[k], j]
        
        # Diagonal correction
        diag_corr = np.outer(np.diag(a), np.diag(b))  # diag_corr[i,j] = a[i,i] * b[j,j]
        
        C = S1 + S2 - diag_corr
        
        # Solve the linear assignment problem
        new_p = _solve_assignment(C)
        
        new_cost = _compute_cost(a, b, new_p)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = new_p.copy()
        
        if new_cost <= current_cost:
            current_cost = new_cost
            p = new_p
        else:
            # If the new solution is worse, we might still want to update p to new_p
            # to break symmetry, or we can stop. Let's update p to new_p to continue refining.
            p = new_p
            current_cost = new_cost
        
        # Check for convergence
        if new_cost == current_cost and np.array_equal(new_p, p):
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
