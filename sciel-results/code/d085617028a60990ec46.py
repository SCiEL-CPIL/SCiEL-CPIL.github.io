"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # 1. Initialization: Spectral Rank Matching
    # This provides a decent starting point better than random.
    try:
        A_sym = (a + a.T) / 2.0
        B_sym = (b + b.T) / 2.0
        A_float = A_sym.astype(np.float64)
        B_float = B_sym.astype(np.float64)
        
        # Get top eigenvectors
        eigvals_A, eigvecs_A = np.linalg.eigh(A_float)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_float)
        
        idx_A = np.argsort(eigvals_A)[::-1]
        idx_B = np.argsort(eigvals_B)[::-1]
        
        v_A = eigvecs_A[:, idx_A[0]]
        v_B = eigvecs_B[:, idx_B[0]]
        
        # Rank matching
        fac_sorted = np.argsort(v_A)
        loc_sorted = np.argsort(v_B)
        
        p = np.empty(n, dtype=np.int64)
        p[fac_sorted] = loc_sorted
    except Exception:
        p = np.arange(n)
        np.random.shuffle(p)

    def compute_cost(p):
        # Exact integer cost
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized: 
        # b_perm[i, j] = b[p[i], p[j]]
        # cost = sum(a * b_perm)
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()

    # Helper to compute the "linearized" cost matrix for the LAP solver.
    # We approximate the QAP objective by fixing the positions of all facilities
    # except the ones we are re-assigning, or by using a surrogate cost matrix.
    # A common heuristic for QAP is to use the "assignment problem" relaxation 
    # where we minimize sum_i sum_j a[i,j] b[x_i, x_j].
    # If we fix the permutation p, the cost is C(p).
    # To find a better p, we can look at the marginal cost of swapping.
    # However, a more powerful approach is to solve a sequence of LAPs 
    # where the cost matrix C[i, j] represents the "penalty" of placing 
    # facility i at location j, given the current assignment of others.
    
    # Let's define a surrogate cost matrix M such that minimizing 
    # sum_i M[i, p[i]] approximates minimizing the QAP cost.
    # One such approximation is based on the "row sums" of the interaction.
    # If we move facility i to location j, the change in cost depends on 
    # its interactions with all other facilities k at their current locations p[k].
    # Delta(i, j) = sum_k a[i, k] (b[j, p[k]] - b[p[i], p[k]]) + terms involving i's old pos.
    # This is complex to compute for all i,j pairs efficiently in a loop.
    
    # Alternative Strategy: Iterated Local Search with 2-Swap moves
    # but accelerated by pre-computing interaction vectors.
    # Let's stick to a robust 2-swap local search but optimize the delta calculation 
    # using NumPy vectorization for batches of swaps, as the previous program did,
    # but add a "kick" (random perturbation) to escape local minima, and 
    # use a smarter initialization.
    
    # Actually, let's try a different structural change: 
    # Use the "LAP relaxation" directly.
    # We construct a cost matrix C where C[i, j] is the estimated cost of 
    # assigning facility i to location j, assuming all other facilities 
    # remain at their current locations.
    # C[i, j] = sum_{k != i} a[i, k] b[j, p[k]]
    # If we minimize sum_i C[i, p[i]], we are minimizing a linear approximation 
    # of the QAP cost around the current solution.
    # This is a "Linearization" approach.
    
    def get_lap_cost_matrix(p):
        # C[i, j] = sum_{k != i} a[i, k] b[j, p[k]]
        # Let B_p be the matrix where B_p[j, k] = b[j, p[k]].
        # Then C[i, :] = a[i, :] @ B_p - a[i, i] b[p[i], p[i]] ? No.
        # The term k=i is excluded.
        # sum_{k != i} a[i, k] b[j, p[k]] = sum_{k} a[i, k] b[j, p[k]] - a[i, i] b[j, p[i]]
        # Let's compute S = a @ B_p, where B_p = b[:, p] (rows of b indexed by p).
        # S[i, j] = sum_k a[i, k] b[j, p[k]]
        # We want C[i, j] = S[i, j] - a[i, i] b[j, p[i]]
        # Note that b[j, p[i]] is the j-th element of column p[i] of b.
        
        # B_p shape (n, n). B_p[j, k] = b[j, p[k]].
        B_p = b[:, p] 
        # S = a @ B_p. Shape (n, n).
        S = a @ B_p
        
        # Correction term:
        # diag_a = np.diag(a)
        # For each i, we subtract a[i, i] * b[:, p[i]] from row i of S.
        # C[i, j] = S[i, j] - a[i, i] * b[j, p[i]]
        
        # b[:, p[i]] is a column vector.
        # We can construct a matrix D where D[i, j] = a[i, i] * b[j, p[i]].
        # D = np.diag(a) @ b[:, p] ? No.
        # D[i, j] = a[i, i] * b[j, p[i]]
        # Let col_indices = p.
        # D = np.diag(a) * b[:, p]  -> This broadcasts diag(a) (n,) with b[:, p] (n, n).
        # b[:, p] has shape (n, n). Its j-th row is b[j, p].
        # We want to multiply row i of this matrix by a[i, i].
        # So D = a.diag[:, np.newaxis] * b[:, p]
        
        D = a.diagonal()[:, np.newaxis] * b[:, p]
        C = S - D
        return C

    # Iterated Local Search
    # 1. Compute LAP cost matrix
    # 2. Solve LAP to get new permutation p_new
    # 3. If p_new is better, accept.
    # 4. If not, or after some iterations, perturb p (random swap) and repeat.
    
    # To avoid getting stuck in the same solution immediately, we can add 
    # noise to the cost matrix or use a "restart" strategy.
    
    max_restarts = 50
    restarts = 0
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() - start_time < time_limit:
        # 1. Linearize and Solve LAP
        C = get_lap_cost_matrix(p)
        
        # Add small random noise to break ties and explore
        # The noise should be small enough not to destroy the structure
        # but large enough to prevent cycling.
        # Scale noise relative to the magnitude of C
        if np.max(np.abs(C)) > 0:
            noise_scale = np.max(np.abs(C)) * 0.01
        else:
            noise_scale = 1.0
        
        C_noisy = C + np.random.normal(0, noise_scale, (n, n))
        
        row_ind, col_ind = linear_sum_assignment(C_noisy)
        p_new = col_ind.astype(np.int64)
        
        cost_new = compute_cost(p_new)
        
        if cost_new < current_cost:
            p = p_new
            current_cost = cost_new
            no_improve_count = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve_count += 1
            # If we've tried many times without improvement, perturb the current best
            if no_improve_count >= max_no_improve:
                # Perturb best_p by random swaps
                p = best_p.copy()
                current_cost = best_cost
                num_swaps = np.random.randint(1, 5)
                for _ in range(num_swaps):
                    i, j = np.random.choice(n, 2, replace=False)
                    p[i], p[j] = p[j], p[i]
                current_cost = compute_cost(p)
                no_improve_count = 0
                restarts += 1
                if restarts > max_restarts:
                    # Try larger perturbations
                    max_no_improve = max(5, max_no_improve // 2)
        
        # Check time
        if time.time() - start_time > time_limit - 5.0:
            break

    # Final local search polish on best_p using 2-swaps
    # This is fast and can squeeze out last bits of improvement
    p = best_p.copy()
    current_cost = best_cost
    
    # Precompute a and b as float for faster delta calculations if needed, 
    # but we need int for cost.
    # Let's do a simple best-improvement 2-swap pass.
    # To keep it fast, we limit the number of passes.
    
    for _ in range(5):
        if time.time() - start_time > time_limit - 2.0:
            break
        
        improved = False
        # Sample a subset of pairs to check
        # Full scan is n^2/2 = ~11k pairs. 
        # Delta calculation is O(n). 11k * 150 = 1.6M ops. 
        # In Python loop, this might be ~1s.
        # Let's do a random sample of 2000 pairs per pass.
        
        pairs = np.random.choice(n*(n-1)//2, size=min(2000, n*(n-1)//2), replace=False)
        # Convert linear index to (i, j)
        # This is tricky. Let's just generate random i, j.
        i_idx = np.random.randint(0, n, size=2000)
        j_idx = np.random.randint(0, n, size=2000)
        mask = i_idx == j_idx
        i_idx[mask] = (i_idx[mask] + 1) % n
        
        for k in range(2000):
            i, j = i_idx[k], j_idx[k]
            if i == j:
                continue
            
            u, v = p[i], p[j]
            
            # Delta calculation
            # diff_a = a[:, i] - a[:, j]
            # diff_b = b[v, p] - b[u, p]
            # total_dot = np.dot(diff_a, diff_b)
            # term_i = (a[i, i] - a[i, j]) * (b[v, u] - b[u, u])
            # term_j = (a[j, i] - a[j, j]) * (b[v, v] - b[u, v])
            # delta = total_dot - term_i - term_j
            
            diff_a = a[:, i] - a[:, j]
            diff_b = b[v, p] - b[u, p]
            total_dot = np.dot(diff_a, diff_b)
            term_i = (a[i, i] - a[i, j]) * (b[v, u] - b[u, u])
            term_j = (a[j, i] - a[j, j]) * (b[v, v] - b[u, v])
            delta = total_dot - term_i - term_j
            
            if delta < 0:
                p[i], p[j] = v, u
                current_cost += int(delta)
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        if not improved:
            break

    return best_p.tolist()

# EVOLVE-BLOCK-END
