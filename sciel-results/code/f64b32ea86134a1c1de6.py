"""QAP tai256c solver: iterated local search with Hungarian re-optimization."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120s

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        # Delta for swapping facilities u and v (which are at locations pu and pv)
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # After swap: p[u]=pv, p[v]=pu
        # Change = sum_i a[i,u]*(b[p[i],pv]-b[p[i],pu]) + sum_j a[u,j]*(b[pv,p[j]]-b[pu,p[j]])
        #         + sum_i a[i,v]*(b[p[i],pu]-b[p[i],pv]) + sum_j a[v,j]*(b[pu,p[j]]-b[pv,p[j]])
        #         + a[u,v]*(b[pv,pu]-b[pu,pv]) + a[v,u]*(b[pu,pv]-b[pv,pu])
        #         - a[u,u]*(b[pu,pu]-b[pu,pu]) ... (self terms cancel)
        #         - a[v,v]*(b[pv,pv]-b[pv,pv]) ... (self terms cancel)
        #
        # More carefully:
        # Terms involving u in row: sum_j a[u,j] * (b[pv, p[j]] - b[pu, p[j]])
        #   But p[u]=pv now, so j=u term: a[u,u]*(b[pv,pv]-b[pu,pu])... wait p[u] was pu, now pv
        #   Actually p[j] for j=u is now pv (was pu). So:
        #   sum_j a[u,j] * b[pv, p[j]] - sum_j a[u,j] * b[pu, p[j]]
        #   where in the first sum, when j=u, p[j]=pv; in second sum when j=u, p[j]=pu
        #
        # Let me use the standard formula:
        # delta = sum_k != u,v [a[u,k]*(b[pv,p[k]]-b[pu,p[k]]) + a[k,u]*(b[p[k],pv]-b[p[k],pu])]
        #         + a[u,v]*(b[pv,pu]-b[pu,pv]) + a[v,u]*(b[pu,pv]-b[pv,pu])
        #         + a[v,v]*(b[pu,pu]-b[pv,pv]) ... no wait
        #
        # Standard 2-swap delta for QAP:
        # Let x = pu, y = pv (locations being swapped)
        # delta = sum_k!=u,v [ a[u,k]*(b[y,p[k]]-b[x,p[k]]) + a[k,u]*(b[p[k],y]-b[p[k],x])
        #         + a[v,k]*(b[x,p[k]]-b[y,p[k]]) + a[k,v]*(b[p[k],x]-b[p[k],y]) ]
        #         + a[u,v]*(b[y,x]-b[x,y]) + a[v,u]*(b[x,y]-b[y,x])
        #         + a[u,u]*(b[y,y]-b[x,x]) ... NO, u stays at same facility, location changes
        #
        # Actually the correct formula:
        # Before: facility u at x, facility v at y
        # After: facility u at y, facility v at x
        #
        # Delta = [sum_k a[u,k]*b[y,p[k]] + sum_k a[k,u]*b[p[k],y]] 
        #       - [sum_k a[u,k]*b[x,p[k]] + sum_k a[k,u]*b[p[k],x]]
        #       + [sum_k a[v,k]*b[x,p[k]] + sum_k a[k,v]*b[p[k],x]]
        #       - [sum_k a[v,k]*b[y,p[k]] + sum_k a[k,v]*b[p[k],y]]
        #       + a[u,v]*(b[y,x]-b[x,y]) + a[v,u]*(b[x,y]-b[y,x])
        #       - a[u,v]*(b[x,y]-b[y,x]) ... wait I'm double counting
        #
        # Let me be very careful. The total cost is sum_{i,j} a[i,j]*b[p[i],p[j]].
        # Only terms where i=u or j=u or i=v or j=v change.
        # 
        # Terms with i=u: sum_j a[u,j]*b[y,p[j]] (after) vs sum_j a[u,j]*b[x,p[j]] (before)
        #   But p[u] changes from x to y, so when j=u: b[y,y] vs b[x,x]
        #   When j=v: b[y,x] vs b[x,y]
        #   When j!=u,v: b[y,p[j]] vs b[x,p[j]]
        #
        # Terms with j=u (i!=u): sum_{i!=u} a[i,u]*b[p[i],y] (after) vs sum_{i!=u} a[i,u]*b[p[i],x] (before)
        #   When i=v: a[v,u]*b[x,y] vs a[v,u]*b[y,x]  (since p[v] changes from y to x)
        #   When i!=u,v: a[i,u]*b[p[i],y] vs a[i,u]*b[p[i],x]
        #
        # Terms with i=v: sum_j a[v,j]*b[x,p[j]] (after) vs sum_j a[v,j]*b[y,p[j]] (before)
        #   When j=u: a[v,u]*b[x,y] vs a[v,u]*b[y,x]
        #   When j=v: a[v,v]*b[x,x] vs a[v,v]*b[y,y]
        #   When j!=u,v: a[v,j]*b[x,p[j]] vs a[v,j]*b[y,p[j]]
        #
        # Terms with j=v (i!=v): sum_{i!=v} a[i,v]*b[p[i],x] (after) vs sum_{i!=v} a[i,v]*b[p[i],y] (before)
        #   When i=u: a[u,v]*b[y,x] vs a[u,v]*b[x,y]
        #   When i!=u,v: a[i,v]*b[p[i],x] vs a[i,v]*b[p[i],y]
        #
        # Now combine, being careful not to double-count:
        # 
        # For k not in {u,v}:
        #   a[u,k]*(b[y,p[k]]-b[x,p[k]]) + a[k,u]*(b[p[k],y]-b[p[k],x])
        #   + a[v,k]*(b[x,p[k]]-b[y,p[k]]) + a[k,v]*(b[p[k],x]-b[p[k],y])
        # = (a[u,k]-a[v,k])*(b[y,p[k]]-b[x,p[k]]) + (a[k,u]-a[k,v])*(b[p[k],y]-b[p[k],x])
        #
        # For the u-v interaction:
        #   From i=u,j=v: a[u,v]*(b[y,x]-b[x,y])
        #   From j=u,i=v: a[v,u]*(b[x,y]-b[y,x])
        #   From i=v,j=u: a[v,u]*(b[x,y]-b[y,x])
        #   From j=v,i=u: a[u,v]*(b[y,x]-b[x,y])
        #   Total: 2*a[u,v]*(b[y,x]-b[x,y]) + 2*a[v,u]*(b[x,y]-b[y,x])
        #         = 2*(a[u,v]-a[v,u])*(b[y,x]-b[x,y])
        #
        # For self terms:
        #   From i=u,j=u: a[u,u]*(b[y,y]-b[x,x])
        #   From i=v,j=v: a[v,v]*(b[x,x]-b[y,y])
        #   Total: (a[u,u]-a[v,v])*(b[y,y]-b[x,x])
        #
        # So: delta = sum_{k!=u,v} (a[u,k]-a[v,k])*(b[y,p[k]]-b[x,p[k]])
        #              + sum_{k!=u,v} (a[k,u]-a[k,v])*(b[p[k],y]-b[p[k],x])
        #              + 2*(a[u,v]-a[v,u])*(b[y,x]-b[x,y])
        #              + (a[u,u]-a[v,v])*(b[y,y]-b[x,x])

        x, y = pu, pv
        # Compute efficiently using precomputed differences
        # sum_{k!=u,v} (a[u,k]-a[v,k])*(b[y,p[k]]-b[x,p[k]])
        diff_a_row = a[u, :] - a[v, :]  # shape (n,)
        diff_b_col = b[y, p] - b[x, p]  # shape (n,)
        s1 = np.dot(diff_a_row, diff_b_col)
        # Subtract k=u and k=v terms
        s1 -= diff_a_row[u] * diff_b_col[u]  # k=u: b[y,p[u]]=b[y,x], b[x,p[u]]=b[x,x]
        s1 -= diff_a_row[v] * diff_b_col[v]  # k=v: b[y,p[v]]=b[y,y], b[x,p[v]]=b[x,y]
        # Actually diff_b_col[u] = b[y,p[u]] - b[x,p[u]] = b[y,x] - b[x,x]
        # diff_b_col[v] = b[y,p[v]] - b[x,p[v]] = b[y,y] - b[x,y]
        # diff_a_row[u] = a[u,u] - a[v,u]
        # diff_a_row[v] = a[u,v] - a[v,v]
        # So we subtracted (a[u,u]-a[v,u])*(b[y,x]-b[x,x]) and (a[u,v]-a[v,v])*(b[y,y]-b[x,y])

        # sum_{k!=u,v} (a[k,u]-a[k,v])*(b[p[k],y]-b[p[k],x])
        diff_a_col = a[:, u] - a[:, v]  # shape (n,)
        diff_b_row = b[p, y] - b[p, x]  # shape (n,)
        s2 = np.dot(diff_a_col, diff_b_row)
        # Subtract k=u and k=v terms
        s2 -= diff_a_col[u] * diff_b_row[u]  # k=u: a[u,u]-a[u,v], b[pu,y]-b[pu,x]=b[x,y]-b[x,x]
        s2 -= diff_a_col[v] * diff_b_row[v]  # k=v: a[v,u]-a[v,v], b[pv,y]-b[pv,x]=b[y,y]-b[y,x]

        # Cross term: 2*(a[u,v]-a[v,u])*(b[y,x]-b[x,y])
        cross = 2.0 * (a[u, v] - a[v, u]) * (b[y, x] - b[x, y])

        # Self term: (a[u,u]-a[v,v])*(b[y,y]-b[x,x])
        self_term = (a[u, u] - a[v, v]) * (b[y, y] - b[x, x])

        return s1 + s2 + cross + self_term

    def local_search(p, cost, max_passes=50):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            # Check time less frequently to reduce overhead
            for i in range(n):
                if passes % 5 == 0 and time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        if time.time() > deadline:
            return p, cost
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        # Build cost matrix: C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # This is the marginal cost of assigning facility i to location j
        # given the current assignment of other facilities.
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # We can compute this as: for each j, sum over k of a[i,k]*b[j,p[k]]
        # = a @ b[:, p]  ... wait, let me think more carefully.
        #
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # If we fix all assignments except facility i, then the cost contribution
        # involving facility i is:
        #   sum_j a[i,j] * b[p[i], p[j]] + sum_j a[j,i] * b[p[j], p[i]]
        # = sum_j a[i,j] * b[p[i], p[j]] + sum_j a[j,i] * b[p[j], p[i]]
        #
        # For facility i assigned to location j (instead of p[i]):
        #   marginal = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        #            - (terms where k=i should be excluded since p[i] changes)
        #
        # Actually for the Hungarian approach, we compute:
        # C[i, j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        # This is an approximation that ignores the self-interaction correction.
        # But it's good enough for a perturbation/re-optimization step.

        # C[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
        # First term: (a @ b[:, p])[i, j] ... no
        # sum_k a[i,k]*b[j,p[k]] = row i of (a @ b[:, p])? 
        # b[:, p] has columns rearranged: column k of b[:, p] is b[:, p[k]]
        # So (a @ b[:, p])[i, j] = sum_k a[i,k] * b[:,p][k,j] = sum_k a[i,k]*b[p[k], j]
        # That's the second term! sum_k a[i,k]*b[p[k], j]
        #
        # For the first term: sum_k a[i,k]*b[j, p[k]]
        # b[j, p[k]] is row j of b, column p[k]
        # = sum_k a[i,k] * b[j, p[k]]
        # Let B_p = b[np.ix_(np.arange(n), p)]  # B_p[j, k] = b[j, p[k]]
        # Then (B_p @ a.T)[j, i] = sum_k B_p[j,k] * a.T[k,i] = sum_k b[j,p[k]] * a[i,k]
        # So first term = (B_p @ a.T).T[i, j] = (a @ B_p.T)[i, j]? 
        # Let's just compute it directly with matrix multiply.

        # First term: M1[i,j] = sum_k a[i,k] * b[j, p[k]]
        # b[np.ix_(np.arange(n), p)] gives matrix where [j,k] = b[j, p[k]]
        # M1 = (a @ b[np.ix_(np.arange(n), p)].T) ... 
        # a @ X.T where X[j,k]=b[j,p[k]]: (a @ X.T)[i,j] = sum_k a[i,k]*X.T[k,j] = sum_k a[i,k]*X[j,k] = sum_k a[i,k]*b[j,p[k]] ✓
        
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        # Second term: M2[i,j] = sum_k a[k,i] * b[p[k], j]
        # b[np.ix_(p, np.arange(n))][k,j] = b[p[k], j]
        # M2 = (b[np.ix_(p, np.arange(n))] @ a)[i, j]? 
        # (Bp2 @ a)[i,j] where Bp2[k,j]=b[p[k],j]: sum_k Bp2[k,j]*a[j,i]... no
        # Let's think: M2[i,j] = sum_k a[k,i] * b[p[k], j]
        # = sum_k (a.T)[i,k] * b[p[k], j]
        # Bp2[k,j] = b[p[k], j]
        # (a.T @ Bp2)[i,j] = sum_k a.T[i,k] * Bp2[k,j] = sum_k a[k,i]*b[p[k],j] ✓
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    # Also try a few random initial solutions and keep the best
    for _ in range(3):
        p_rand = np.random.permutation(n).astype(np.int64)
        c_rand = compute_cost(p_rand)
        if c_rand < compute_cost(p):
            p = p_rand

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with periodic Hungarian re-optimization
    iteration = 0
    while time.time() < deadline:
        iteration += 1

        # Perturb: random swaps with adaptive intensity
        p = best_p.copy()
        n_swaps = np.random.randint(3, 10)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))

        # Every few iterations, try Hungarian re-optimization
        if iteration % 4 == 0 and time.time() < deadline - 30:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()
