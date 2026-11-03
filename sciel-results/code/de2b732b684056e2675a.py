# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = 115.0  # safety margin before 120s kill

    # Ensure int64 for exact arithmetic if possible, though float64 is usually fine for delta comparisons
    # The problem states a and b are int64.
    
    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # Spectral initialization
    # We want to minimize sum a[i,j] b[p[i], p[j]].
    # This is similar to aligning the flow matrix A with the distance matrix B.
    # If A and B are symmetric, we can use eigenvectors.
    # For minimization, we want high flow pairs to have low distance.
    # So we should align the dominant eigenvector of A with the dominant eigenvector of B?
    # Let's check the sign.
    # If A has large positive entries and B has large positive entries, we want to put high A entries with low B entries.
    # This suggests anti-alignment if we just dot product.
    # However, the standard spectral heuristic for QAP (Taillard) uses:
    # p = argsort(argsort(v_a)) where v_a is eigenvector of A.
    # And similar for B.
    # Let's try both alignment and anti-alignment.
    
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    # Use the eigenvector corresponding to the largest eigenvalue
    # For minimization of quadratic form x^T A x, we usually look at the smallest eigenvalue?
    # No, for QAP it's different.
    # Let's just use the dominant eigenvector (largest magnitude) and try sorting.
    
    eigvals_a, eigvecs_a = np.linalg.eigh(A_sym)
    eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
    
    # Take the eigenvector with the largest absolute eigenvalue
    idx_a = np.argmax(np.abs(eigvals_a))
    idx_b = np.argmax(np.abs(eigvals_b))
    
    v_a = eigvecs_a[:, idx_a]
    v_b = eigvecs_b[:, idx_b]
    
    # Sort facilities by v_a and locations by v_b
    # Try 4 combinations: 
    # 1. Ascending A, Ascending B
    # 2. Descending A, Descending B
    # 3. Ascending A, Descending B
    # 4. Descending A, Ascending B
    
    candidates = []
    
    orders_a = [np.argsort(v_a), np.argsort(-v_a)]
    orders_b = [np.argsort(v_b), np.argsort(-v_b)]
    
    for oa in orders_a:
        for ob in orders_b:
            p = np.empty(n, dtype=np.int64)
            p[oa] = ob
            candidates.append(p)
            
    # Also add a random initialization
    candidates.append(np.random.permutation(n))
    
    # Select the best initial candidate
    best_p = None
    best_cost = np.inf
    for p in candidates:
        c = compute_cost(p)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
            
    # Local Search with 2-opt (swap) and Or-opt (move)
    # Vectorized delta for swap (i, j):
    # Let u = p[i], v = p[j].
    # Delta = sum_k (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) + ...
    # Actually, the exact delta for swapping i and j in permutation p:
    # C_new - C_old = 
    #   sum_k a[i,k] (b[v, p[k]] - b[u, p[k]]) 
    # + sum_k a[k,i] (b[p[k], v] - b[p[k], u])
    # + sum_k a[j,k] (b[u, p[k]] - b[v, p[k]])
    # + sum_k a[k,j] (b[p[k], u] - b[p[k], v])
    # - (a[i,j] (b[v, u] - b[u, v]) + a[j,i] (b[u, v] - b[v, u]))  # Correction for double counting i,j terms?
    
    # Let's derive it carefully.
    # C = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Swap i, j. Let u=p[i], v=p[j].
    # Terms involving i or j change.
    # Row i: sum_k a[i,k] b[u, p[k]] -> sum_k a[i,k] b[v, p[k]] (with p[i]=v, p[j]=u)
    #   Change: sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v])
    # Col i: sum_k a[k,i] b[p[k], u] -> sum_k a[k,i] b[p[k], v]
    #   Change: sum_{k!=i,j} a[k,i] (b[p[k], v] - b[p[k], u]) + a[i,i](b[v,v]-b[u,u]) + a[j,i](b[u,v]-b[v,u])
    # Row j: sum_k a[j,k] b[v, p[k]] -> sum_k a[j,k] b[u, p[k]]
    #   Change: sum_{k!=i,j} a[j,k] (b[u, p[k]] - b[v, p[k]]) + a[j,j](b[u,u]-b[v,v]) + a[j,i](b[u,v]-b[v,u])
    # Col j: sum_k a[k,j] b[p[k], v] -> sum_k a[k,j] b[p[k], u]
    #   Change: sum_{k!=i,j} a[k,j] (b[p[k], u] - b[p[k], v]) + a[i,j](b[v,u]-b[u,v]) + a[j,j](b[u,u]-b[v,v])
    
    # Summing these up:
    # Delta = sum_{k!=i,j} [ (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]]) + (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u]) ]
    #       + 2*a[i,i]*(b[v,v]-b[u,u]) + 2*a[j,j]*(b[u,u]-b[v,v])
    #       + a[i,j]*(2*b[v,u] - 2*b[u,v]) + a[j,i]*(2*b[u,v] - 2*b[v,u])
    
    # Wait, the terms for k=i and k=j in the sums above were excluded.
    # Let's just compute the delta for the full sums and subtract the "unchanged" parts?
    # No, it's easier to just compute the change in the 4 sums directly.
    
    # Let's use a simpler vectorized approach for the main part:
    # Let A_i = a[i, :], A_j = a[j, :]
    # Let A_col_i = a[:, i], A_col_j = a[:, j]
    # Let B_u = b[u, :], B_v = b[v, :]
    # Let B_col_u = b[:, u], B_col_v = b[:, v]
    
    # The change in sum_{k} a[i,k] b[p[i], p[k]] when p[i] changes from u to v:
    # Note: p[k] changes for k=i and k=j.
    # This is tricky to vectorize perfectly without masks.
    
    # Alternative: Just compute the cost of the affected sub-matrix?
    # For n=100, a full cost calculation is 100*100 = 10k ops.
    # A swap affects 4 rows/cols.
    # We can compute the delta by:
    # 1. Compute contribution of i, j to old cost.
    # 2. Compute contribution of i, j to new cost.
    # This is O(n) per swap if done carefully.
    
    def get_swap_delta(p, i, j):
        u, v = p[i], p[j]
        if u == v: return 0
        
        # Vector for b[u, p], b[v, p], b[p, u], b[p, v]
        # p is the permutation array.
        # b[u, p] is b[u, p[0]], b[u, p[1]], ...
        bu_p = b[u, p]
        bv_p = b[v, p]
        bp_u = b[p, u]
        bp_v = b[p, v]
        
        # Delta for row i: sum_k a[i,k] (b[v, p_new[k]] - b[u, p[k]])
        # p_new[i] = v, p_new[j] = u, p_new[k] = p[k]
        # We can approximate by assuming p[k] doesn't change for the vector dot product, then correct.
        # This approximation is often sufficient for "best improvement" selection if we verify the top candidates?
        # Or just compute exact delta.
        
        # Exact Delta:
        # Sum_{k} a[i,k] b[v, p_new[k]] - Sum_{k} a[i,k] b[u, p[k]]
        # = a[i,i] b[v,v] + a[i,j] b[v,u] + Sum_{k!=i,j} a[i,k] b[v, p[k]] 
        #   - (a[i,i] b[u,u] + a[i,j] b[u,v] + Sum_{k!=i,j} a[i,k] b[u, p[k]])
        
        # Let's compute the sums for k != i, j using vectorization with masks.
        # mask = np.ones(n, dtype=bool); mask[i]=False; mask[j]=False
        
        # This is getting slow if we create masks for every pair.
        # Let's stick to the formula derived earlier which allows O(n) vectorized ops without masks if we handle i,j separately.
        
        # Delta = 
        #   (a[i, :] - a[j, :]) @ (bv_p - bu_p)  # This assumes p[k] is same for all k.
        #   + (a[:, i] - a[:, j]) @ (bp_v - bp_u)
        #   - Corrections for i and j indices.
        
        # Let's check the correction.
        # The term (a[i, :] - a[j, :]) @ (bv_p - bu_p) expands to:
        # Sum_k (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
        # = Sum_{k!=i,j} ... + (a[i,i]-a[j,i])(b[v,u]-b[u,u]) + (a[i,j]-a[j,j])(b[v,v]-b[u,v])
        # 
        # Our target delta for Row i + Row j part (ignoring Col parts for a moment):
        # Row i change: a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + Sum_{k!=i,j} a[i,k](b[v,p[k]]-b[u,p[k]])
        # Row j change: a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v]) + Sum_{k!=i,j} a[j,k](b[u,p[k]]-b[v,p[k]])
        # Sum = a[i,i](b[v,v]-b[u,u]) - a[j,j](b[v,v]-b[u,u]) 
        #       + a[i,j](b[v,u]-b[u,v]) - a[j,i](b[v,u]-b[u,v])
        #       + Sum_{k!=i,j} (a[i,k]-a[j,k])(b[v,p[k]]-b[u,p[k]])
        
        # The vectorized part gives:
        # Sum_{k!=i,j} (a[i,k]-a[j,k])(b[v,p[k]]-b[u,p[k]]) 
        # + (a[i,i]-a[j,i])(b[v,u]-b[u,u]) + (a[i,j]-a[j,j])(b[v,v]-b[u,v])
        
        # So we need to add the difference between the target boundary terms and the vectorized boundary terms.
        
        # Target Boundary Terms (Row parts):
        # TB_R = a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v])
        #       = (a[i,i]-a[j,j])(b[v,v]-b[u,u]) + (a[i,j]-a[j,i])(b[v,u]-b[u,v])
        
        # Vectorized Boundary Terms (Row parts):
        # VB_R = (a[i,i]-a[j,i])(b[v,u]-b[u,u]) + (a[i,j]-a[j,j])(b[v,v]-b[u,v])
        
        # Correction_R = TB_R - VB_R
        
        # Similarly for Column parts.
        # Target Boundary Terms (Col parts):
        # Col i change: a[i,i](b[v,v]-b[u,u]) + a[j,i](b[u,v]-b[v,u]) + Sum...
        # Col j change: a[i,j](b[v,u]-b[u,v]) + a[j,j](b[u,u]-b[v,v]) + Sum...
        # Sum = (a[i,i]-a[j,j])(b[v,v]-b[u,u]) + (a[j,i]-a[i,j])(b[u,v]-b[v,u])
        #      = (a[i,i]-a[j,j])(b[v,v]-b[u,u]) - (a[i,j]-a[j,i])(b[v,u]-b[u,v])
        
        # Vectorized Boundary Terms (Col parts):
        # (a[i,i]-a[j,i])(b[v,v]-b[u,u]) + (a[j,i]-a[j,j])(b[u,v]-b[v,u])  -- Wait, let's re-derive.
        # Vectorized Col: (a[:, i] - a[:, j]) @ (bp_v - bp_u)
        # = Sum_k (a[k,i]-a[k,j]) (b[p[k], v] - b[p[k], u])
        # = Sum_{k!=i,j} ... + (a[i,i]-a[i,j])(b[u,v]-b[u,u]) + (a[j,i]-a[j,j])(b[v,v]-b[v,u])
        
        # Target Col Boundary:
        # TC_C = a[i,i](b[v,v]-b[u,u]) + a[j,i](b[u,v]-b[v,u]) + a[i,j](b[v,u]-b[u,v]) + a[j,j](b[u,u]-b[v,v])
        #       = (a[i,i]-a[j,j])(b[v,v]-b[u,u]) + (a[j,i]-a[i,j])(b[u,v]-b[v,u])
        
        # This is getting too complex to do analytically in the fly.
        # Given n=100, let's just compute the delta by summing the 4 vectors directly.
        
        # We can precompute for each location l:
        # row_diff[l] = b[l, :] - b[:, l] ? No.
        
        # Let's just use the O(n) loop but with NumPy vectorization for the inner sum?
        # No, a Python loop of 10000 iterations * 100 ops is 1M ops, which is ~10ms.
        # That's fast enough for one pass.
        # But we want to do many passes.
        
        # Let's try a different strategy: 
        # Use `np.einsum` or matrix multiplication to compute the delta for ALL pairs (i,j) at once?
        # Delta(i,j) is bilinear in i and j.
        
        # Actually, let's just optimize the local search by using "First Improvement" and a fast delta.
        # I will implement a fast delta using precomputed arrays.
        
        # Precompute:
        # For each location l, let R_l = b[l, :] and C_l = b[:, l].
        # For a permutation p, we can maintain:
        # S_row[i] = sum_k a[i,k] b[p[i], p[k]]
        # S_col[i] = sum_k a[k,i] b[p[k], p[i]]
        # Cost = sum_i (S_row[i] + S_col[i]) - sum_i a[i,i] b[p[i], p[i]] (to avoid double counting diagonal? No, C = sum_{i,j} a[i,j] b[p[i], p[j]]. The diagonal is included.)
        
        # If we swap p[i] and p[j] (u, v):
        # S_row[i] changes to sum_k a[i,k] b[v, p_new[k]]
        # This is hard to update incrementally without O(n).
        
        # Let's stick to the O(n) delta but vectorized per pair.
        # We will iterate i, j.
        
        return 0 # Placeholder, will implement in local_search
    
    # Local Search Implementation
    def local_search(p, time_limit):
        n = len(p)
        p = p.copy()
        cost = compute_cost(p)
        
        # Precompute a and b as float for speed? No, keep int.
        
        # We will use a "Best Improvement" strategy but limit the number of candidates?
        # Or "First Improvement".
        
        improved = True
        while improved:
            if time.time() - t_start > time_limit:
                break
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # To speed up, we can check a random subset of pairs?
            # No, let's check all pairs but compute delta efficiently.
            
            # We will compute delta for all pairs in a vectorized way?
            # Delta(i,j) = (a[i,:]-a[j,:]) @ (b[v,:]-b[u,:]) ... NO, p[k] varies.
            
            # Let's just loop.
            for i in range(n):
                if time.time() - t_start > time_limit:
                    break
                u = p[i]
                for j in range(i+1, n):
                    v = p[j]
                    if u == v: continue
                    
                    # Compute delta
                    # We need:
                    # 1. sum_k a[i,k] (b[v, p[k]] - b[u, p[k]])
                    # 2. sum_k a[k,i] (b[p[k], v] - b[p[k], u])
                    # 3. sum_k a[j,k] (b[u, p[k]] - b[v, p[k]])
                    # 4. sum_k a[k,j] (b[p[k], u] - b[p[k], v])
                    # 5. Corrections for k=i, j.
                    
                    # Let's compute the main sums first.
                    # b[v, p] is a vector. b[u, p] is a vector.
                    # a[i, :] is a vector.
                    # dot1 = a[i, :] @ (b[v, p] - b[u, p])
                    # dot2 = a[:, i] @ (b[p, v] - b[p, u])
                    # dot3 = a[j, :] @ (b[u, p] - b[v, p])
                    # dot4 = a[:, j] @ (b[p, u] - b[p, v])
                    
                    # Note: b[p, v] is b[p[0], v], b[p[1], v]... which is b[:, v][p].
                    # So b[p, v] = b[:, v][p].
                    
                    diff_row = b[v, p] - b[u, p]
                    diff_col = b[:, v][p] - b[:, u][p]
                    
                    d1 = a[i, :] @ diff_row
                    d2 = a[:, i] @ diff_col
                    d3 = a[j, :] @ (-diff_row)
                    d4 = a[:, j] @ (-diff_col)
                    
                    # Now corrections for k=i and k=j.
                    # The dot products above used p[k] for all k.
                    # But for k=i, p_new[i]=v, p[i]=u.
                    # For k=j, p_new[j]=u, p[j]=v.
                    
                    # Let's calculate the exact change for the terms involving i and j explicitly and subtract the "wrong" parts from the dot products.
                    
                    # In d1 (Row i), the term for k=i was a[i,i] (b[v, p[i]] - b[u, p[i]]) = a[i,i] (b[v, u] - b[u, u]).
                    # The correct term is a[i,i] (b[v, v] - b[u, u]).
                    # So d1 has error: a[i,i] (b[v, u] - b[v, v]).
                    
                    # In d1, the term for k=j was a[i,j] (b[v, p[j]] - b[u, p[j]]) = a[i,j] (b[v, v] - b[u, v]).
                    # The correct term is a[i,j] (b[v, u] - b[u, v]).
                    # So d1 has error: a[i,j] (b[v, v] - b[v, u]).
                    
                    # Total error in d1: a[i,i] (b[v, u] - b[v, v]) + a[i,j] (b[v, v] - b[v, u])
                    
                    # Similarly for d2 (Col i):
                    # Term k=i: a[i,i] (b[p[i], v] - b[p[i], u]) = a[i,i] (b[u, v] - b[u, u]).
                    # Correct: a[i,i] (b[v, v] - b[u, u]).
                    # Error: a[i,i] (b[u, v] - b[v, v]).
                    
                    # Term k=j: a[j,i] (b[p[j], v] - b[p[j], u]) = a[j,i] (b[v, v] - b[v, u]).
                    # Correct: a[j,i] (b[u, v] - b[v, u]).
                    # Error: a[j,i] (b[v, v] - b[u, v]).
                    
                    # Total error in d2: a[i,i] (b[u, v] - b[v, v]) + a[j,i] (b[v, v] - b[u, v])
                    
                    # Similarly for d3 (Row j):
                    # It uses -diff_row.
                    # Term k=i: -a[j,i] (b[v, u] - b[u, u]).
                    # Correct: a[j,i] (b[u, v] - b[v, u]) = -a[j,i] (b[v, u] - b[u, v]).
                    # Error: -a[j,i] (b[v, u] - b[u, u]) + a[j,i] (b[v, u] - b[u, v]) = a[j,i] (b[u, v] - b[u, u]).
                    
                    # Term k=j: -a[j,j] (b[v, v] - b[u, v]).
                    # Correct: a[j,j] (b[u, u] - b[v, v]) = -a[j,j] (b[v, v] - b[u, u]).
                    # Error: -a[j,j] (b[v, v] - b[u, v]) + a[j,j] (b[v, v] - b[u, u]) = a[j,j] (b[u, v] - b[u, u]).
                    
                    # Total error in d3: a[j,i] (b[u, v] - b[u, u]) + a[j,j] (b[u, v] - b[u, u])
                    
                    # Similarly for d4 (Col j):
                    # Uses -diff_col.
                    # Term k=i: -a[i,j] (b[u, v] - b[u, u]).
                    # Correct: a[i,j] (b[v, u] - b[u, v]) = -a[i,j] (b[u, v] - b[v, u]).
                    # Error: -a[i,j] (b[u, v] - b[u, u]) + a[i,j] (b[u, v] - b[v, u]) = a[i,j] (b[u, u] - b[v, u]).
                    
                    # Term k=j: -a[j,j] (b[v, v] - b[v, u]).
                    # Correct: a[j,j] (b[u, u] - b[v, v]) = -a[j,j] (b[v, v] - b[u, u]).
                    # Error: -a[j,j] (b[v, v] - b[v, u]) + a[j,j] (b[v, v] - b[u, u]) = a[j,j] (b[v, u] - b[u, u]).
                    
                    # Total error in d4: a[i,j] (b[u, u] - b[v, u]) + a[j,j] (b[v, u] - b[u, u])
                    
                    delta = d1 + d2 + d3 + d4
                    # Add corrections (subtract errors)
                    delta -= (a[i,i] * (b[v, u] - b[v, v]) + a[i,j] * (b[v, v] - b[v, u]))
                    delta -= (a[i,i] * (b[u, v] - b[v, v]) + a[j,i] * (b[v, v] - b[u, v]))
                    delta -= (a[j,i] * (b[u, v] - b[u, u]) + a[j,j] * (b[u, v] - b[u, u]))
                    delta -= (a[i,j] * (b[u, u] - b[v, u]) + a[j,j] * (b[v, u] - b[u, u]))
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
        
        return p, cost

    # Run Local Search
    p, best_cost = local_search(best_p, deadline)
    
    # Iterated Local Search
    rng = np.random.RandomState(42)
    
    while time.time() - t_start < deadline - 2.0:
        # Perturb
        p_pert = p.copy()
        n_swaps = rng.randint(3, 8)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        p_new, cost_new = local_search(p_pert, deadline)
        
        if cost_new < best_cost:
            p = p_new
            best_cost = cost_new
        # else: keep p
    
    return p.tolist()

# EVOLVE-BLOCK-END
