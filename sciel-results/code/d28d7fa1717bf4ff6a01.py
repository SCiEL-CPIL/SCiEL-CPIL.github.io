"""QAP solver: bipartite-matching initialization + corrected fast local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Structural change: initialize via Hungarian on C = a @ b (linear relaxation).
    Then iterate: local search -> Hungarian re-linearization -> local search.
    """
    n = len(a)
    deadline = time.time() + 112.0  # safety margin before 120s limit

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def _update_b_p(p):
        """Precompute b_p[k,l] = b[p[k],l] and b_pT[k,l] = b[l,p[k]]."""
        return b[p, :].copy(), b[:, p].copy()

    def swap_delta_fast(i, j, p, b_p, b_pT):
        """Compute delta for swapping positions i and j. O(n) time."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0

        # Diagonal/cross terms
        delta = 0.0
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[i, j] * (b[v, u] - b[u, v])
        delta += a[j, i] * (b[u, v] - b[v, u])
        delta += a[j, j] * (b[u, u] - b[v, v])

        # Row i, k not in {i,j}: sum a[i,k]*(b[v,p[k]] - b[u,p[k]])
        # b[v,p[k]] = b_pT[k,v], b[u,p[k]] = b_pT[k,u]
        # Full dot: a[i,:] @ (b_pT[:,v] - b_pT[:,u])
        # Subtract k=i: a[i,i]*(b[v,u]-b[u,u])  [since b_pT[i,v]=b[v,p[i]]=b[v,u], b_pT[i,u]=b[u,u]]
        # Subtract k=j: a[i,j]*(b[v,u]-b[u,u])  [since b_pT[j,v]=b[v,p[j]]=b[v,u], b_pT[j,u]=b[u,u]]
        ri = b_pT[:, v] - b_pT[:, u]
        delta += np.dot(a[i], ri) - a[i, i] * (b[v, u] - b[u, u]) - a[i, j] * (b[v, u] - b[u, u])

        # Row j, k not in {i,j}: sum a[j,k]*(b[u,p[k]] - b[v,p[k]])
        rj = b_pT[:, u] - b_pT[:, v]
        delta += np.dot(a[j], rj) - a[j, i] * (b[u, u] - b[v, u]) - a[j, j] * (b[u, u] - b[v, v])

        # Col i, k not in {i,j}: sum a[k,i]*(b[p[k],v] - b[p[k],u])
        # b[p[k],v] = b_p[k,v], b[p[k],u] = b_p[k,u]
        # Full dot: a[:,i] @ (b_p[:,v] - b_p[:,u])
        # Subtract k=i: a[i,i]*(b[v,v]-b[u,u])  [b_p[i,v]=b[p[i],v]=b[u,v], b_p[i,u]=b[u,u]]
        # Wait: b_p[i,v] = b[p[i], v] = b[u, v], b_p[i,u] = b[p[i], u] = b[u, u]
        # Subtract k=j: a[j,i]*(b[u,v]-b[u,u])  [b_p[j,v]=b[p[j],v]=b[v,v], b_p[j,u]=b[p[j],u]=b[v,u]]
        # Wait: b_p[j,v] = b[p[j], v] = b[v, v], b_p[j,u] = b[p[j], u] = b[v, u]
        ci = b_p[:, v] - b_p[:, u]
        delta += np.dot(a[:, i], ci) - a[i, i] * (b[u, v] - b[u, u]) - a[j, i] * (b[v, v] - b[v, u])

        # Col j, k not in {i,j}: sum a[k,j]*(b[p[k],u] - b[p[k],v])
        cj = b_p[:, u] - b_p[:, v]
        delta += np.dot(a[:, j], cj) - a[i, j] * (b[u, u] - b[u, v]) - a[j, j] * (b[v, u] - b[v, v])

        return delta

    def local_search_2swap(p, cost, b_p, b_pT, max_time, num_trials=20000):
        """Hill-climbing 2-swap local search."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j, p, b_p, b_pT)
                if d < -0.5:
                    u, v = p[i], p[j]
                    p[i], p[j] = v, u
                    cost += int(d)
                    b_p[[i, j], :] = b_p[[j, i], :]
                    b_pT[:, [i, j]] = b_pT[:, [j, i]]
                    improved = True
        return p, cost, b_p, b_pT

    def hungarian_init():
        """Initialize via Hungarian on C = a @ b (linear relaxation of QAP)."""
        # C[i,j] = sum_k a[i,k]*b[k,j] = (a @ b)[i,j]
        # Minimizing sum_i C[i,p[i]] approximates the QAP
        C = a.astype(np.float64) @ b.astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def hungarian_refine(p):
        """Build linear cost matrix from current p, solve assignment, return new p."""
        # C_new[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
        #           = (a @ b[:,p])[i,j] + (a.T @ b[p,:])[i,j]
        b_col_p = b[:, p]   # shape (n,n): b_col_p[j,k] = b[j,p[k]]
        b_row_p = b[p, :]   # shape (n,n): b_row_p[k,j] = b[p[k],j]
        C = a.astype(np.float64) @ b_col_p.astype(np.float64).T + a.T.astype(np.float64) @ b_row_p.astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new

    # --- Phase 1: Hungarian-based initialization ---
    best_p = hungarian_init()
    best_cost = compute_cost(best_p)
    b_p, b_pT = _update_b_p(best_p)

    # --- Phase 2: Initial local search ---
    best_p, best_cost, b_p, b_pT = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 50.0)

    # --- Phase 3: Hungarian refinement + local search ---
    if time.time() < deadline - 35.0:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            b_p, b_pT = _update_b_p(best_p)
            best_p, best_cost, b_p, b_pT = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 25.0)

    # --- Phase 4: Second Hungarian refinement ---
    if time.time() < deadline - 20.0:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            b_p, b_pT = _update_b_p(best_p)
            best_p, best_cost, b_p, b_pT = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 10.0)

    # --- Phase 5: Final local search ---
    if time.time() < deadline - 2.0:
        best_p, best_cost, b_p, b_pT = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 1.0)

    return best_p.tolist()
# EVOLVE-BLOCK-END
