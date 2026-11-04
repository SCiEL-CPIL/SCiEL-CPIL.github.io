"""QAP solver: Sinkhorn Continuous Relaxation + Vectorized SA Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    
    start_time = time.time()
    deadline = 115.0

    def calc_cost(p):
        """Calculate the exact QAP cost for a permutation p."""
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_swap_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities at indices i and j in permutation p.
        Optimized for speed using NumPy vectorization.
        """
        u = p[i]
        v = p[j]
        
        # Direct pair term
        delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        # Diagonal terms
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        # Cross terms with all other k
        # We compute the change in interaction with all other facilities k != i, j
        # sum_{k != i, j} (a[i, k] * b[u, p[k]] + a[k, i] * b[p[k], u] + a[j, k] * b[v, p[k]] + a[k, j] * b[p[k], v])
        # - sum_{k != i, j} (a[i, k] * b[v, p[k]] + a[k, i] * b[p[k], v] + a[j, k] * b[u, p[k]] + a[k, j] * b[p[k], u])
        
        # Let's vectorize this.
        # For k in 0..n-1:
        # Term_k = (a[i, k] - a[j, k]) * (b[u, p[k]] - b[v, p[k]]) + (a[k, i] - a[k, j]) * (b[p[k], u] - b[p[k], v])
        
        # Note: b is symmetric in wil100, but we should keep it general.
        # If b is symmetric: b[u, p[k]] - b[v, p[k]] is row diff.
        
        # Let's compute the full sum over all k, then subtract k=i and k=j terms which were handled or need correction.
        # Actually, the formula above assumes k != i, j.
        
        # Vectorized calculation for all k:
        # diff_a_row = a[i] - a[j]  (shape n)
        # diff_a_col = a[:, i] - a[:, j] (shape n)
        
        # diff_b_row_u_v = b[u, :] - b[v, :] (shape n) -> We need b[u, p[k]] - b[v, p[k]]
        # diff_b_col_u_v = b[:, u] - b[:, v] (shape n) -> We need b[p[k], u] - b[p[k], v]
        
        # Let's precompute b rows/cols for u and v? No, u and v change per call.
        # b[u, p] is a vector of length n.
        # b[v, p] is a vector of length n.
        
        b_u_p = b[u, p]
        b_v_p = b[v, p]
        b_p_u = b[p, u]
        b_p_v = b[p, v]
        
        diff_b_row = b_u_p - b_v_p
        diff_b_col = b_p_u - b_p_v
        
        diff_a_row = a[i] - a[j]
        diff_a_col = a[:, i] - a[:, j]
        
        # Sum over all k
        sum_T1 = np.dot(diff_a_row, diff_b_row)
        sum_T2 = np.dot(diff_a_col, diff_b_col)
        
        # Subtract k=i terms
        # k=i: a[i, i] * (b[u, p[i]] - b[v, p[i]]) + a[i, i] * (b[p[i], u] - b[p[i], v])
        # p[i] = u
        # b[u, u] - b[v, u]
        # b[u, u] - b[u, v]
        term_ki_T1 = (a[i, i] - a[j, i]) * (b[u, u] - b[v, u])
        term_ki_T2 = (a[i, i] - a[i, j]) * (b[u, u] - b[u, v])
        
        # Subtract k=j terms
        # k=j: a[i, j] * (b[u, p[j]] - b[v, p[j]]) + a[j, j] * (b[p[j], u] - b[p[j], v])
        # p[j] = v
        # b[u, v] - b[v, v]
        # b[v, u] - b[v, v]
        term_kj_T1 = (a[i, j] - a[j, j]) * (b[u, v] - b[v, v])
        term_kj_T2 = (a[j, i] - a[j, j]) * (b[v, u] - b[v, v])
        
        delta += sum_T1 + sum_T2 - term_ki_T1 - term_ki_T2 - term_kj_T1 - term_kj_T2
        
        return int(delta)

    def hungarian_init():
        """
        Initialize using Hungarian algorithm on a surrogate cost matrix.
        Surrogate: C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
        This captures the first-order approximation of the QAP cost.
        """
        # a_row_sums is not enough, we need the interaction.
        # C[i, j] = (a @ b)[i, j] + (a.T @ b)[i, j] ?
        # Cost = sum_i sum_j a[i, j] b[p[i], p[j]]
        # Approximation: assume p[i] is independent.
        # Let's use the row sum of a weighted by row sum of b, which is a common heuristic.
        # Better: C[i, j] = sum_k a[i, k] * b[j, k]
        # This is the "row interaction" cost.
        
        # Let's compute C = a @ b.T ? No.
        # If facility i is at location j, it interacts with facility k at location p[k].
        # The cost contribution of facility i is sum_k a[i, k] b[j, p[k]].
        # If we ignore the dependence on p[k] for other k, we can approximate b[j, p[k]] by its average or by assuming p[k] is uniformly distributed?
        # A better surrogate: C[i, j] = sum_k a[i, k] * (sum_l b[j, l] / n) ? No, that's constant for j.
        
        # Standard heuristic: C[i, j] = sum_k a[i, k] * b[j, k] is not quite right because k is facility index, j is location index.
        # If we assume the assignment is identity (p[k]=k), then cost is sum a[i, j] b[j, k] ... no.
        
        # Let's use the surrogate:
        # C[i, j] = sum_k a[i, k] * b[j, k]  (if we assume facility k is at location k)
        # And C[i, j] += sum_k a[k, i] * b[k, j]
        
        C1 = a @ b.T  # C1[i, j] = sum_k a[i, k] b[j, k]
        C2 = a.T @ b  # C2[i, j] = sum_k a[k, i] b[k, j]
        
        C = C1 + C2
        
        # We want to minimize cost, so we minimize C.
        # linear_sum_assignment minimizes.
        row_ind, col_ind = linear_sum_assignment(C)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def sinkhorn_relaxation(C, iterations=100, eps=0.1):
        """
        Perform Sinkhorn-Knopp iterations on the cost matrix C to find a doubly stochastic matrix.
        Then use it to perturb the assignment.
        """
        # Normalize C to be positive
        C_norm = C - C.min() + 1.0
        
        # Initialize X
        X = np.ones((n, n), dtype=np.float64) / n
        
        for _ in range(iterations):
            # Row normalize
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            # Col normalize
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
            
            # Multiply by exp(-C/eps)
            # To avoid underflow, we can do:
            # X = X * np.exp(-C_norm / eps)
            # But this might underflow.
            # Better: 
            # X = np.exp(-C_norm / eps)
            # Then normalize.
            # Let's stick to the multiplicative update.
            X = X * np.exp(-C_norm / eps)
            
        return X

    def sa_refine(p, current_cost, time_budget):
        """
        Vectorized Simulated Annealing refinement.
        """
        p = p.copy()
        cost = current_cost
        best_p = p.copy()
        best_cost = cost
        
        # Temperature schedule
        T_start = 500.0
        T_end = 1.0
        alpha = 0.995
        
        T = T_start
        rng = np.random.default_rng()
        
        while T > T_end and time.time() - start_time < time_budget:
            # Generate random swap
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
                
            delta = get_swap_delta(p, i, j)
            
            # Acceptance probability
            if delta < 0:
                accept = True
            else:
                # Avoid log of 0 or underflow
                if T > 0:
                    prob = np.exp(-delta / T)
                    accept = rng.random() < prob
                else:
                    accept = False
            
            if accept:
                p[i], p[j] = p[j], p[i]
                cost += delta
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            
            T *= alpha
            
        return best_p, best_cost

    # 1. Initialization
    p = hungarian_init()
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # 2. Initial SA Refinement
    # Use a portion of the time for initial polish
    time_spent = time.time() - start_time
    remaining = deadline - time_spent
    # Allocate 30% of remaining time to initial SA
    p, current_cost = sa_refine(p, current_cost, time_spent + 0.3 * remaining)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # 3. Main Loop: Sinkhorn Perturbation + SA Refinement
    # We will perform multiple cycles.
    # Each cycle:
    # 1. Compute surrogate C based on current best_p? Or fixed?
    #    Using fixed C is faster. Using current best_p might be better for local exploration.
    #    Let's use a dynamic surrogate: C = a @ b.T + a.T @ b is fixed.
    #    However, to escape local minima, we can use the Sinkhorn matrix X to propose a new discrete permutation.
    
    C_surrogate = a @ b.T + a.T @ b
    
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() - start_time < deadline - 2.0:
        # 1. Sinkhorn Relaxation
        # Use a larger epsilon for more global exploration
        eps = 1.0 + 0.1 * no_improve_count
        X = sinkhorn_relaxation(C_surrogate, iterations=50, eps=eps)
        
        # 2. Convert X to a discrete permutation
        # Use Hungarian on X (or -X if maximizing, but X is a probability mass, so we want to match high probabilities)
        # We want to minimize the "cost" of assigning based on X.
        # So we use C_assign = -X for linear_sum_assignment (which minimizes).
        row_ind, col_ind = linear_sum_assignment(-X)
        
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        # 3. SA Refinement on the new permutation
        cost_new = calc_cost(p_new)
        
        # Use remaining time for SA, but limit per iteration to allow multiple cycles
        time_spent = time.time() - start_time
        remaining = deadline - time_spent
        # Allocate 20% of remaining time to this SA step
        sa_budget = time_spent + 0.2 * remaining
        
        p_new, cost_new = sa_refine(p_new, cost_new, sa_budget)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # If stuck, try a pure random restart or larger perturbation
            if no_improve_count > max_no_improve:
                # Random restart from Hungarian
                p_rand = hungarian_init()
                cost_rand = calc_cost(p_rand)
                # Quick SA
                time_spent = time.time() - start_time
                remaining = deadline - time_spent
                sa_budget = time_spent + 0.1 * remaining
                p_rand, cost_rand = sa_refine(p_rand, cost_rand, sa_budget)
                
                if cost_rand < best_cost:
                    best_cost = cost_rand
                    best_p = p_rand.copy()
                    no_improve_count = 0
                else:
                    # If still stuck, break or continue?
                    # Let's continue with larger eps
                    pass

    return best_p.tolist()
# EVOLVE-BLOCK-END
