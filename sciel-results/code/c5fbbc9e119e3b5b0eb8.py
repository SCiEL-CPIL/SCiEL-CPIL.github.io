"""QAP tai256c solver: iterated local search with vectorized 2-opt, Hungarian re-optimization, and adaptive diversification."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def or_opt_delta(p, i, j, k):
        """
        Calculate delta for moving segment [i, j] to position after k.
        i, j, k are indices in p.
        Assumes i < j.
        If k < i, segment moves left. If k > j, segment moves right.
        If k is inside [i, j], it's a rotation, which is less common in simple Or-opt, 
        but we handle it by treating it as a move to a new position.
        For simplicity and robustness, we restrict k to be outside [i, j] initially 
        or handle it carefully. 
        Standard Or-opt moves a block to a new position.
        Let's implement a simpler version: Move item at i to position j (insertion).
        This is effectively Or-opt of length 1.
        """
        # We will implement a more general Or-opt: move block [i, j] to after k.
        # To keep it efficient and correct, let's stick to length 1 (swap is length 1 exchange, 
        # insertion is length 1 move) and length 2/3 blocks.
        # Actually, let's implement a "shift" of a block [i, j] to position k (0-indexed, meaning insert before k or at end).
        # Let's use the definition: Move facility at index i to index j.
        # This is a 1-element Or-opt.
        
        # Let's implement a more powerful move: Swap two elements is already done.
        # Let's implement "Move i to j" (insertion).
        # Delta for moving p[i] to position j (shifting elements in between).
        
        # This is complex to vectorize quickly for all i,j.
        # Let's implement a simple 1-element move delta.
        # Move p[i] to p[j] (j > i). Elements i+1..j shift left.
        
        # This is O(n) to compute if done naively.
        # Given the time limit, maybe we should stick to swaps but change the diversification?
        # The directive says "different representation" or "different optimization method".
        # Simulated Annealing is a different method.
        # Or-opt is a different neighborhood.
        
        # Let's implement a fast 1-element move delta.
        # It is expensive to calculate exactly for all pairs in O(n^2) time per pass if we do it naively.
        # However, we can try it in the local search if we have time, or just use it in the kick.
        
        # Let's use a simpler structural change: 
        # 1. Simulated Annealing for diversification.
        # 2. 3-opt or Or-opt in local search? 
        # 3-opt is hard to vectorize.
        
        # Let's try a different local search neighborhood: 
        # Instead of just swaps, we can do "reversals" (inverting a segment).
        # Reversal delta calculation is also complex.
        
        # Let's focus on the "Different Optimization Method" part: Simulated Annealing.
        # And keep the swap local search.
        
        pass # Placeholder for future Or-opt implementation if needed

    def local_search(p, cost, max_passes=20, exact=False):
        """Steepest Descent Local Search with vectorized delta calculation for speed."""
        improved = True
        passes = 0
        p = p.copy()
        
        # Precompute symmetric A for vectorized delta calculation
        A_sym = a + a.T
        D = b[np.ix_(p, p)].copy()
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            for i in range(n - 1):
                if time.time() > deadline:
                    return p, cost
                    
                if exact:
                    # Exact O(n) delta calculation for final polish
                    best_delta = 0
                    best_j = -1
                    for j in range(i + 1, n):
                        delta = fast_swap_delta(p, i, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                    min_delta = best_delta
                    min_idx = None
                    if best_j >= 0:
                        j = best_j
                    else:
                        continue
                else:
                    js = np.arange(i + 1, n)
                    
                    # Vectorized delta calculation
                    diff_A = A_sym[i] - A_sym[js]
                    diff_D = D[js] - D[i]
                    
                    S = np.sum(diff_A * diff_D, axis=1)
                    
                    # Correction terms
                    coeff_i = A_sym[i, i] - A_sym[js, i]
                    term_i = coeff_i * (D[js, i] - D[i, i])
                    
                    coeff_j = A_sym[i, js] - np.diag(A_sym[js])
                    term_j = coeff_j * (D[js, js] - D[i, js])
                    
                    diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
                    cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
                    
                    deltas = S - term_i - term_j + diag_term + cross_term
                    
                    min_idx = np.argmin(deltas)
                    min_delta = deltas[min_idx]
                    
                    if min_delta < 0:
                        j = js[min_idx]
                    else:
                        continue
                
                if min_delta < 0:
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    cost += min_delta
                    improved = True
        
        return p, cost

    def sa_kick(p, cost, steps=50, T0=100.0, T1=1.0):
        """
        Simulated Annealing kick to diversify.
        Returns the best solution found during the SA process.
        """
        best_p = p.copy()
        best_cost = cost
        current_p = p.copy()
        current_cost = cost
        
        # Precompute for fast delta calculation during SA
        # We will use fast_swap_delta for simplicity in SA steps
        # as vectorization overhead might be high for single steps
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Temperature cooling
            T = T0 * (T1 / T0) ** (step / steps)
            
            # Pick random pair
            i, j = np.random.choice(n, 2, replace=False)
            
            # Compute delta
            delta = fast_swap_delta(current_p, i, j)
            
            # Acceptance probability
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-9)):
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
            
        return best_p, best_cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    # Removed tabu_kick, replaced by sa_kick

    # Initial solution: rank matching based on row sums
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with periodic Hungarian re-optimization and tabu kicks
    iteration = 0
    consecutive_no_improve = 0
    base_perturb = 3

    while time.time() < deadline - 10:  # Leave 10s buffer
        iteration += 1

        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb

        # Perturb: Use Simulated Annealing kick for structural diversification
        p = best_p.copy()
        p, cost = sa_kick(p, compute_cost(p), steps=20, T0=50.0, T1=1.0)

        # Local search from perturbed
        p, cost = local_search(p, cost)

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    # Final high-precision polish: use exact delta calculation with more passes
    if time.time() < deadline - 5:
        final_p = best_p.copy()
        final_cost = best_cost
        # First do a vectorized pass to get close
        final_p, final_cost = local_search(final_p, final_cost, max_passes=10)
        if final_cost < best_cost:
            best_p = final_p.copy()
            best_cost = final_cost
        
        # Then do exact polish passes
        if time.time() < deadline - 2:
            exact_p = best_p.copy()
            exact_cost = best_cost
            exact_p, exact_cost = local_search(exact_p, exact_cost, max_passes=5, exact=True)
            if exact_cost < best_cost:
                best_p = exact_p
                best_cost = exact_cost
        
        # One more vectorized pass to catch any remaining improvements
        if time.time() < deadline - 1:
            final_p = best_p.copy()
            final_cost = best_cost
            final_p, final_cost = local_search(final_p, final_cost, max_passes=3)
            if final_cost < best_cost:
                best_p = final_p
                best_cost = final_cost

    return best_p.tolist()
