"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(a, b):
    n = len(a)
    # deadline guard: keep ~10s safety margin under the 150s kill
    deadline = time.time() + 138.0

    # Precompute row sums of b (distance to each location from all other locations)
    # For a swap of facilities i and j (currently at locations u=p[i], v=p[j]):
    # delta = sum_k a[i,k]*(b[u,p[k]] - b[v,p[k]]) + sum_k a[j,k]*(b[v,p[k]] - b[u,p[k]])
    #        + (a[i,j]+a[j,i])*(b[u,v] - b[v,u])  -- but b is symmetric so b[u,v]==b[v,u], last term 0
    # More carefully:
    # The cost contribution involving facility i: sum_{k != i} a[i,k]*b[p[i],p[k]]
    # After swapping i<->j: facility i goes to v, facility j goes to u.
    # New cost for i: sum_{k != i,j} a[i,k]*b[v,p[k]] + a[i,j]*b[v,u]
    # Old cost for i: sum_{k != i,j} a[i,k]*b[u,p[k]] + a[i,j]*b[u,v]
    # Similarly for j.
    # Since b is symmetric, b[v,u]==b[u,v], so the cross terms cancel.
    # delta = sum_{k != i,j} a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_{k != i,j} a[j,k]*(b[u,p[k]] - b[v,p[k]])
    #       = sum_{k != i,j} (a[i,k] - a[j,k])*(b[v,p[k]] - b[u,p[k]])

    # We'll maintain an array 'cost_at[i]' = sum_{k != i} a[i,k]*b[p[i],p[k]]
    # and total_cost = sum_i cost_at[i] / 2  (since each pair counted twice)
    # Actually total = sum_{i,j} a[i,j]*b[p[i],p[j]]. Let's just track total directly.

    # Better approach: precompute for each location l, the vector d_l[k] = b[l, p[k]] for all k.
    # That's expensive. Instead, maintain an array 'loc_cost[l]' = sum_k b[l, p[k]] * something... 

    # Let's use the standard incremental swap delta:
    # delta(i,j) where i is at u=p[i], j is at v=p[j]:
    # delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])

    # We can precompute: for each pair of locations (u,v), the vector diff[u,v][k] = b[v,p[k]] - b[u,p[k]]
    # But that's O(n^3) memory. Instead, compute on the fly but cache per (u,v) row difference.

    # Alternative: precompute row differences. For each location l, store b[l, :] as a row.
    # Then b[v, p[k]] - b[u, p[k]] = (b[v] - b[u])[p[k]]
    # So delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v] - b[u])[p[k]]

    # Let's precompute row_diff[u,v] = b[v] - b[u] for all pairs? That's n^2 vectors of length n = n^3. Too much for n=100? 100^3 = 1e6 floats, that's 8MB. Actually fine.
    # But we can avoid storing all: just compute (b[v] - b[u]) on the fly, it's O(n) per swap check.

    # For efficiency in local search, we'll compute delta for a given (i,j) in O(n):
    #   diff_row = b[p[j]] - b[p[i]]  # O(n)
    #   delta = np.dot(a[i] - a[j], diff_row[p])  -- but need to exclude k=i and k=j
    #   Actually a[i,i]=a[j,j]=0 typically, and the terms for k=i,j: 
    #   k=i: (a[i,i]-a[j,i])*(b[v,p[i]]-b[u,p[i]]) = (0-a[j,i])*(b[v,u]-b[u,u]) 
    #   k=j: (a[i,j]-a[j,j])*(b[v,p[j]]-b[u,p[j]]) = (a[i,j]-0)*(b[v,v]-b[u,v])
    #   These are small corrections. Let's just include them or handle carefully.
    #   Simpler: compute full dot product and subtract the i and j terms.

    # Let's use a cleaner formulation. Total cost C = sum_{i,j} a[i,j] b[p[i],p[j]]
    # Swap i and j: new assignment p' where p'[i]=p[j], p'[j]=p[i], p'[k]=p[k] for k!=i,j
    # C' - C = sum_{k!=i,j} [a[i,k]b[p[j],p[k]] + a[j,k]b[p[i],p[k]] - a[i,k]b[p[i],p[k]] - a[j,k]b[p[j],p[k]]]
    #         + a[i,j]b[p[j],p[i]] + a[j,i]b[p[i],p[j]] - a[i,j]b[p[i],p[j]] - a[j,i]b[p[j],p[i]]
    # Since b symmetric: b[p[j],p[i]]=b[p[i],p[j]], so last group = a[i,j]b[u,v]+a[j,i]b[u,v]-a[i,j]b[u,v]-a[j,i]b[u,v]=0
    # C'-C = sum_{k!=i,j} (a[i,k]-a[j,k])(b[p[j],p[k]]-b[p[i],p[k]])
    #      = sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[p[j]]-b[p[i]])[p[k]]

    # Implementation:
    # Precompute nothing extra; use numpy operations.

    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    def compute_total(p):
        # C = sum_{i,j} a[i,j] b[p[i],p[j]]
        # Use matrix product: for each i, sum_j a[i,j]*b[p[i],p[j]]
        # Vectorized: 
        bp = b[np.arange(n), p]  # shape (n, n): bp[i,k] = b[p[i], p[k]]
        return int(np.sum(a * bp))

    best_p = p.copy()
    best_cost = compute_total(p)

    # Iterated Local Search with 2-opt swaps
    # For each iteration, do a full pass of best-improvement or first-improvement swaps
    # Use first-improvement for speed, with random restarts.

    current_p = p.copy()
    current_cost = best_cost

    # Precompute a[i] - a[j] on the fly. For speed, precompute all pairwise a differences? n^2 vectors of length n = 1e6, fine.
    # But let's just compute on the fly.

    # To speed up: precompute b rows as a list of arrays
    b_rows = [b[i] for i in range(n)]

    # Local search: try all pairs (i,j), take best improvement
    # For n=100, there are ~5000 pairs. Each delta is O(n)=100 ops. Total per pass ~500k ops.
    # In Python this might be slow. Let's vectorize where possible.

    # Actually, let's use a smarter approach: for each i, compute the best j to swap with.
    # delta(i,j) = sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[p[j]]-b[p[i]])[p[k]]
    # = sum_k (a[i,k]-a[j,k]) * (b[p[j]]-b[p[i]])[p[k]]  (terms k=i,j are zero or small)
    # Let d_ij = a[i] - a[j]  (length n)
    # Let r_ij = (b[p[j]] - b[p[i]])[p]  (length n)
    # delta = d_ij @ r_ij

    # For fixed i, we want to find j minimizing delta(i,j).
    # d_ij = a[i] - a[j]
    # r_ij = (b[p[j]] - b[p[i]])[p] = b[p[j]][p] - b[p[i]][p]
    # delta = (a[i]-a[j]) @ (b[p[j]][p] - b[p[i]][p])
    # = a[i]@b[p[j]][p] - a[i]@b[p[i]][p] - a[j]@b[p[j]][p] + a[j]@b[p[i]][p]
    # = (a[i]@b[p[j]][p] + a[j]@b[p[i]][p]) - (a[i]@b[p[i]][p] + a[j]@b[p[j]][p])

    # Precompute: for each facility i and location l, val[i,l] = a[i] @ b[l][p] = sum_k a[i,k]*b[l,p[k]]
    # This is n*n = 10000 values, each O(n) to compute initially = 1e6 ops. Fine.
    # Then delta(i,j) = val[i,p[j]] + val[j,p[i]] - val[i,p[i]] - val[j,p[j]]

    # This is much faster! O(1) per pair after O(n^3) setup.

    def build_val(p):
        # val[i,l] = sum_k a[i,k] * b[l, p[k]]
        # = a[i] @ b[l, p] = a[i] @ (b[l][p])
        # Vectorized: for each i, for each l, dot(a[i], b[l][p])
        # b[l][p] is just b[l] indexed by p
        # val = np.zeros((n, n))
        # for i in range(n):
        #     for l in range(n):
        #         val[i,l] = np.dot(a[i], b[l][p])
        # Faster: val[i,:] = a[i] @ (b[:, p].T)  -- b[:,p] has shape (n,n), transposed (n,n)
        # b[:, p] shape (n, n): row l is b[l, p[0]], ..., b[l, p[n-1]]
        # We want for each i: a[i] @ B_p where B_p[l,k] = b[l, p[k]]
        # So val[i,l] = sum_k a[i,k] * B_p[l,k] = (a[i] @ B_p.T)[l]
        # B_p.T[k,l] = B_p[l,k] = b[l, p[k]]
        # So val[i, :] = a[i] @ B_p.T
        # B_p = b[:, p]  shape (n, n)
        # B_p.T = b[:, p].T  shape (n, n)
        # val = a @ B_p.T  shape (n, n)
        B_p = b[:, p]  # shape (n, n)
        val = a @ B_p.T  # shape (n, n)
        return val

    val = build_val(current_p)

    def delta_swap(i, j, val, p):
        # delta = val[i, p[j]] + val[j, p[i]] - val[i, p[i]] - val[j, p[j]]
        return val[i, p[j]] + val[j, p[i]] - val[i, p[i]] - val[j, p[j]]

    # Local search: best-improvement 2-opt
    def local_search(p, val, cost):
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            best_delta = 0
            best_pair = (-1, -1)
            # Try all pairs
            for i in range(n):
                for j in range(i+1, n):
                    d = delta_swap(i, j, val, p)
                    if d < best_delta:
                        best_delta = d
                        best_pair = (i, j)
            if best_pair[0] >= 0:
                i, j = best_pair
                # Perform swap
                p[i], p[j] = p[j], p[i]
                cost += best_delta
                # Rebuild val
                val = build_val(p)
                improved = True
        return p, val, cost

    # Run local search
    current_p, val, current_cost = local_search(current_p, val, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = current_p.copy()

    # Iterated local search: perturb and repeat
    while time.time() < deadline - 5:
        # Perturb: random k swaps
        p = best_p.copy()
        val = build_val(p)
        cost = best_cost

        # Random perturbation
        k = np.random.randint(2, 6)
        for _ in range(k):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
            cost = compute_total(p)  # Recompute for simplicity after perturbation
            val = build_val(p)

        # Local search
        p, val, cost = local_search(p, val, cost)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
