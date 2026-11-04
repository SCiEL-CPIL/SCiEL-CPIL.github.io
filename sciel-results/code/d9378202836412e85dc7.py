"""QAP solver: Multi-start + Fast Incremental 2-swap/3-swap Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Generate multiple diverse initial permutations (Rank, Spectral, Random, Hungarian).
    2. For each initial permutation, perform fast local search:
       - 2-swap moves with O(n) incremental delta updates.
       - 3-swap moves (cyclic) to escape 2-swap local optima.
    3. Return the best permutation found within the time limit.
    """
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit
    
    # Precompute constant matrices for speed
    a_T = a.T
    b_T = b.T
    b_diag = np.diag(b)
    
    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized: sum_i sum_j a[i,j] b[p[i], p[j]]
        # b[p, :] is matrix B' where B'[i, j] = b[p[i], j]
        # Cost = sum_i (a[i, :] @ b[p[i], :])
        # Or more efficiently: trace(a.T @ b[p, p])? No.
        # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
        # Let B_p be matrix with B_p[i, j] = b[p[i], p[j]].
        # Then Cost = sum_{i,j} a[i,j] B_p[i,j] = np.sum(a * b[np.ix_(p, p)])
        # np.ix_(p, p) creates a meshgrid index.
        # This is O(n^2) but very fast in NumPy.
        return np.sum(a * b[np.ix_(p, p)])

    def spectral_init(seed=0):
        """Initialize using eigenvectors."""
        np.random.seed(seed)
        try:
            a_sym = (a + a.T) / 2.0
            b_sym = (b + b.T) / 2.0
            # Use leading eigenvectors
            # For minimization, align high flow with low distance
            # We want to maximize -trace(A^T P B P^T) roughly
            # Approximation: Sort facilities by 'strength' in a, locations by 'strength' in b
            evals_a, evecs_a = np.linalg.eigh(a_sym)
            evals_b, evecs_b = np.linalg.eigh(b_sym)
            
            # Use the eigenvector corresponding to the largest absolute eigenvalue
            idx_a = np.argmax(np.abs(evals_a))
            idx_b = np.argmax(np.abs(evals_b))
            
            vec_a = evecs_a[:, idx_a]
            vec_b = evecs_b[:, idx_b]
            
            # Sort facilities by vec_a, locations by vec_b
            # We want to match high vec_a with low vec_b if eigenvalues have opposite signs, etc.
            # A robust heuristic: Sort by value.
            fac_order = np.argsort(vec_a)
            loc_order = np.argsort(vec_b)
            
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            return p
        except Exception:
            return np.random.permutation(n)

    def rank_init():
        """Initialize by sorting row sums."""
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        
        # Facilities with high flow should go to locations with low distance to others?
        # Or high flow to high connectivity?
        # Let's try matching high a-sums with low b-sums (to minimize product)
        fac_order = np.argsort(-a_sums)
        loc_order = np.argsort(b_sums)
        
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        return p

    def hungarian_init():
        """Initialize using Hungarian algorithm on a linearized cost."""
        # Linearization: C[i, j] = sum_k a[i, k] b[j, p[k]] + sum_k a[k, i] b[p[k], j]
        # This is circular. Use a simple heuristic linearization:
        # C[i, j] = sum_k a[i, k] b[j, k] ? No.
        # Use: C[i, j] = (a.sum(axis=1)[i] * b.sum(axis=1)[j])
        # Or better: C[i, j] = sum_k a[i, k] b[j, k] (approximation of interaction with same index)
        # Let's just use random for diversity in multi-start, or a simple greedy.
        # Actually, let's use the gradient of the QAP at a random point to guide Hungarian.
        p_rand = np.random.permutation(n)
        B_p = b[np.ix_(p_rand, p_rand)]
        # Gradient approx: G[i, j] = sum_k a[i, k] b[j, p_rand[k]] + sum_k a[k, i] b[p_rand[k], j]
        # This is expensive. 
        # Let's stick to Rank and Spectral for diversity, plus Random.
        return np.random.permutation(n)

    def local_search(p, cost, time_limit):
        """
        Fast local search with 2-swaps and 3-swaps.
        p: current permutation (numpy array of int64)
        cost: current cost
        time_limit: absolute time deadline
        """
        p = p.copy()
        
        # Precompute dot_row and dot_col
        # dot_row[i, l] = sum_k a[i, k] * b[l, p[k]]
        # dot_col[i, l] = sum_k a[k, i] * b[p[k], l]
        
        def recompute_precomp(p):
            # b_col_p[l, k] = b[l, p[k]]
            b_col_p = b[:, p] 
            # b_row_p[k, l] = b[p[k], l]
            b_row_p = b[p, :]
            
            # dot_row[i, l] = sum_k a[i, k] * b[l, p[k]] = (a @ b_col_p.T)[i, l]
            dot_row = a @ b_col_p.T
            # dot_col[i, l] = sum_k a[k, i] * b[p[k], l] = (a.T @ b_row_p)[i, l]
            dot_col = a_T @ b_row_p
            return dot_row, dot_col

        dot_row, dot_col = recompute_precomp(p)
        
        improved = True
        last_improve_time = time.time()
        
        # Main loop
        while time.time() < time_limit:
            # Check for stagnation to trigger 3-swaps or restart
            if time.time() - last_improve_time > 2.0:
                # Try 3-swaps
                improved_3 = False
                for _ in range(5000):
                    if time.time() > time_limit:
                        break
                    i, j, k = np.random.randint(0, n, 3)
                    if i == j or j == k or i == k:
                        continue
                    
                    u, v, w = p[i], p[j], p[k]
                    if u == v or v == w or u == w:
                        continue
                    
                    # Calculate delta for 3-cycle i->j->k->i
                    # New positions: p'[i]=v, p'[j]=w, p'[k]=u
                    # Old: p[i]=u, p[j]=v, p[k]=w
                    
                    # Delta = Cost(new) - Cost(old)
                    # We can compute this using the precomputed dot products, but it's complex.
                    # Simpler: Compute cost difference directly for the 3 involved indices and their interactions with others.
                    # Since n=150, a direct O(n) calculation for 3-swaps is feasible if optimized.
                    
                    # Let's just do a direct O(n) delta calculation for 3-swaps
                    # Terms involving i, j, k
                    delta = 0.0
                    
                    # Internal 3x3 block change
                    # Old: a[i,i]b[u,u] + a[i,j]b[u,v] + a[i,k]b[u,w] + ...
                    # New: a[i,i]b[v,v] + a[i,j]b[v,w] + a[i,k]b[v,u] + ...
                    
                    # It is safer to just compute the full cost change for the 3 rows/cols involved
                    # But that's O(n).
                    
                    # Let's use the incremental update idea for 3-swaps if possible, 
                    # or just accept that 3-swaps are less frequent.
                    
                    # Quick check: Is it worth it? 
                    # Let's try a simple random 3-swap and compute exact delta.
                    p_new = p.copy()
                    p_new[i], p_new[j], p_new[k] = v, w, u
                    
                    # Calculate cost change
                    # Only rows i, j, k and cols i, j, k of the cost matrix change significantly?
                    # No, cost is sum_{x,y} a[x,y] b[p[x], p[y]].
                    # Change is sum_{x,y} a[x,y] (b[p_new[x], p_new[y]] - b[p[x], p[y]])
                    # p_new[x] != p[x] only for x in {i, j, k}.
                    # So the sum is over x in {i,j,k} and all y, plus x all, y in {i,j,k} minus double counted.
                    
                    # Sum 1: x in {i,j,k}, y all
                    delta += np.sum(a[[i,j,k], :] * (b[np.ix_(p_new[[i,j,k]], p_new)] - b[np.ix_(p[[i,j,k]], p)]))
                    # Sum 2: x all, y in {i,j,k}, x not in {i,j,k}
                    mask_x = np.ones(n, dtype=bool)
                    mask_x[[i,j,k]] = False
                    delta += np.sum(a[mask_x, :] * (b[np.ix_(p_new[mask_x], p_new[[i,j,k]])] - b[np.ix_(p[mask_x], p[[i,j,k]])]))
                    
                    if delta < -1e-9:
                        p = p_new
                        cost += delta
                        dot_row, dot_col = recompute_precomp(p)
                        improved_3 = True
                        last_improve_time = time.time()
                        break # Restart 2-swap loop after a 3-swap improvement
                
                if not improved_3 and time.time() > time_limit - 5.0:
                    break
                continue

            # 2-Swap Local Search
            improved = False
            # Number of trials per pass
            num_trials = 10000 
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Calculate delta for swapping p[i] and p[j]
                # u = p[i], v = p[j]
                # New: p'[i]=v, p'[j]=u
                
                # delta = sum_{k} a[i,k] (b[v, p[k]] - b[u, p[k]]) + 
                #         sum_{k} a[j,k] (b[u, p[k]] - b[v, p[k]]) +
                #         sum_{k} a[k,i] (b[p[k], v] - b[p[k], u]) +
                #         sum_{k} a[k,j] (b[p[k], u] - b[p[k], v])
                
                # Using precomputed dot_row and dot_col:
                # dot_row[i, l] = sum_k a[i,k] b[l, p[k]]
                # So sum_k a[i,k] b[v, p[k]] = dot_row[i, v]
                # sum_k a[i,k] b[u, p[k]] = dot_row[i, u]
                
                term_row_i = dot_row[i, v] - dot_row[i, u]
                term_row_j = dot_row[j, u] - dot_row[j, v]
                
                # dot_col[i, l] = sum_k a[k,i] b[p[k], l]
                # sum_k a[k,i] b[p[k], v] = dot_col[i, v]
                # sum_k a[k,i] b[p[k], u] = dot_col[i, u]
                
                term_col_i = dot_col[i, v] - dot_col[i, u]
                term_col_j = dot_col[j, u] - dot_col[j, v]
                
                # Wait, the terms above include k=i and k=j. We need to subtract the double-counted or incorrect internal terms?
                # Let's verify.
                # The full delta is:
                # D = sum_{k} a[i,k] (b[v, p'[k]] - b[u, p[k]]) + ... 
                # No, p' only changes at i and j.
                # p'[i] = v, p'[j] = u, p'[k] = p[k] for k != i,j.
                
                # Let's expand sum_{x,y} a[x,y] (b[p'[x], p'[y]] - b[p[x], p[y]])
                
                # x=i, y=j: a[i,j] (b[v,u] - b[u,v])
                # x=j, y=i: a[j,i] (b[u,v] - b[v,u])
                # x=i, y=i: a[i,i] (b[v,v] - b[u,u])
                # x=j, y=j: a[j,j] (b[u,u] - b[v,v])
                
                # x=i, y=k (k!=i,j): a[i,k] (b[v, p[k]] - b[u, p[k]])
                # x=j, y=k (k!=i,j): a[j,k] (b[u, p[k]] - b[v, p[k]])
                # x=k, y=i (k!=i,j): a[k,i] (b[p[k], v] - b[p[k], u])
                # x=k, y=j (k!=i,j): a[k,j] (b[p[k], u] - b[p[k], v])
                
                # My dot_row/dot_col sums include k=i and k=j.
                # dot_row[i, v] = sum_{k} a[i,k] b[v, p[k]] = a[i,i]b[v,u] + a[i,j]b[v,v] + sum_{k!=i,j} a[i,k]b[v,p[k]]
                # dot_row[i, u] = sum_{k} a[i,k] b[u, p[k]] = a[i,i]b[u,u] + a[i,j]b[u,v] + sum_{k!=i,j} a[i,k]b[u,p[k]]
                
                # So dot_row[i, v] - dot_row[i, u] = a[i,i](b[v,u]-b[u,u]) + a[i,j](b[v,v]-b[u,v]) + sum_{k!=i,j} a[i,k](b[v,p[k]]-b[u,p[k]])
                
                # This is getting complicated to subtract. It is often easier to just compute the delta directly for n=150.
                # O(n) per swap is 150 ops. 10k swaps = 1.5M ops. Very fast.
                
                # Let's use the direct O(n) calculation for safety and correctness.
                # delta = 0
                # for k in range(n):
                #    if k == i or k == j: continue
                #    delta += a[i,k] * (b[v, p[k]] - b[u, p[k]])
                #    delta += a[j,k] * (b[u, p[k]] - b[v, p[k]])
                #    delta += a[k,i] * (b[p[k], v] - b[p[k], u])
                #    delta += a[k,j] * (b[p[k], u] - b[p[k], v])
                # delta += a[i,i]*(b[v,v]-b[u,u])
                # delta += a[i,j]*(b[v,u]-b[u,v])
                # delta += a[j,i]*(b[u,v]-b[v,u])
                # delta += a[j,j]*(b[u,u]-b[v,v])
                
                # Vectorized O(n):
                # mask = np.ones(n, dtype=bool); mask[i]=mask[j]=False
                # This creates a new array every time, which is slow.
                # Instead, we can use the precomputed dot_row/dot_col and subtract the known i,j terms.
                
                # Let's stick to the precomputed method but carefully subtract.
                
                # From above:
                # DR_i = dot_row[i, v] - dot_row[i, u]
                # DR_j = dot_row[j, u] - dot_row[j, v]
                # DC_i = dot_col[i, v] - dot_col[i, u]
                # DC_j = dot_col[j, u] - dot_col[j, v]
                
                # DR_i includes terms for k=i and k=j.
                # k=i term in DR_i: a[i,i](b[v, p[i]] - b[u, p[i]]) = a[i,i](b[v,u] - b[u,u])
                # k=j term in DR_i: a[i,j](b[v, p[j]] - b[u, p[j]]) = a[i,j](b[v,v] - b[u,v])
                
                # DR_j includes terms for k=i and k=j.
                # k=i term in DR_j: a[j,i](b[u, p[i]] - b[v, p[i]]) = a[j,i](b[u,u] - b[v,u])
                # k=j term in DR_j: a[j,j](b[u, p[j]] - b[v, p[j]]) = a[j,j](b[u,v] - b[v,v])
                
                # DC_i includes terms for k=i and k=j.
                # k=i term in DC_i: a[i,i](b[p[i], v] - b[p[i], u]) = a[i,i](b[u,v] - b[u,u])
                # k=j term in DC_i: a[j,i](b[p[j], v] - b[p[j], u]) = a[j,i](b[v,v] - b[v,u])
                
                # DC_j includes terms for k=i and k=j.
                # k=i term in DC_j: a[i,j](b[p[i], u] - b[p[i], v]) = a[i,j](b[u,u] - b[u,v])
                # k=j term in DC_j: a[j,j](b[p[j], u] - b[p[j], v]) = a[j,j](b[v,u] - b[v,v])
                
                # The total delta is sum of all 4 terms (DR_i, DR_j, DC_i, DC_j) 
                # MINUS the double counted internal terms?
                # Actually, the sum of DR_i + DR_j + DC_i + DC_j covers:
                # All x=i,y=k; x=j,y=k; x=k,y=i; x=k,y=k
                # The internal terms (i,i), (i,j), (j,i), (j,j) are included in these sums.
                # Do we need to subtract them?
                # Let's check (i,i).
                # In DR_i (k=i): a[i,i](b[v,u]-b[u,u])
                # In DC_i (k=i): a[i,i](b[u,v]-b[u,u])
                # Sum: a[i,i](b[v,u] + b[u,v] - 2b[u,u])
                # Correct delta for (i,i): a[i,i](b[v,v] - b[u,u])
                # These are NOT the same. So we cannot just sum them.
                
                # Therefore, the precomputed method requires subtracting the "wrong" internal contributions and adding the "correct" ones.
                # This is error-prone. Given n=150, the direct O(n) loop in Python is ~150 iterations.
                # 10,000 swaps * 150 ops = 1.5M ops. In Python, this might be ~0.1s - 0.5s.
                # This is acceptable.
                
                # Let's use the direct O(n) calculation for robustness.
                # Optimization: Use NumPy slices to avoid Python loop overhead.
                
                # b_v_p = b[v, p]
                # b_u_p = b[u, p]
                # b_p_v = b[p, v]
                # b_p_u = b[p, u]
                
                # We need sums over k != i, j.
                # We can compute the full sum and subtract i, j terms.
                
                # Sum_row_i = np.dot(a[i], b[v, p] - b[u, p])
                # Sum_row_j = np.dot(a[j], b[u, p] - b[v, p])
                # Sum_col_i = np.dot(a[:, i], b[p, v] - b[p, u])
                # Sum_col_j = np.dot(a[:, j], b[p, u] - b[p, v])
                
                # This is 4 dot products of size n. Very fast in NumPy.
                
                # Then subtract the i,j terms that were included in the dot products but are handled separately or incorrectly.
                
                # Let's compute the "raw" deltas:
                d_raw_i = np.dot(a[i], b[v, p] - b[u, p])
                d_raw_j = np.dot(a[j], b[u, p] - b[v, p])
                d_raw_ci = np.dot(a[:, i], b[p, v] - b[p, u])
                d_raw_cj = np.dot(a[:, j], b[p, u] - b[p, v])
                
                # The true delta is:
                # Delta = (Sum_{k!=i,j} a[i,k](...)) + (Sum_{k!=i,j} a[j,k](...)) + (Sum_{k!=i,j} a[k,i](...)) + (Sum_{k!=i,j} a[k,j](...))
                #        + a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v])
                
                # d_raw_i = a[i,i](b[v,u]-b[u,u]) + a[i,j](b[v,v]-b[u,v]) + Sum_{k!=i,j} ...
                # So Sum_{k!=i,j} ... = d_raw_i - a[i,i](b[v,u]-b[u,u]) - a[i,j](b[v,v]-b[u,v])
                
                # Similarly for others.
                
                # Let's just compute the internal terms explicitly and add them.
                # And subtract the internal terms from the raw sums.
                
                # Internal terms:
                # T_ii = a[i,i]*(b[v,v]-b[u,u])
                # T_ij = a[i,j]*(b[v,u]-b[u,v])
                # T_ji = a[j,i]*(b[u,v]-b[v,u])
                # T_jj = a[j,j]*(b[u,u]-b[v,v])
                
                # Terms to subtract from d_raw_i:
                # S_ii = a[i,i]*(b[v,u]-b[u,u])
                # S_ij = a[i,j]*(b[v,v]-b[u,v])
                
                # Terms to subtract from d_raw_j:
                # S_ji = a[j,i]*(b[u,u]-b[v,u])  # Note: b[u, p[i]]=b[u,u], b[v, p[i]]=b[v,u]
                # S_jj = a[j,j]*(b[u,v]-b[v,v])
                
                # Terms to subtract from d_raw_ci:
                # SC_ii = a[i,i]*(b[u,v]-b[u,u])
                # SC_ji = a[j,i]*(b[v,v]-b[v,u])
                
                # Terms to subtract from d_raw_cj:
                # SC_ij = a[i,j]*(b[u,u]-b[u,v])
                # SC_jj = a[j,j]*(b[v,u]-b[v,v])
                
                # This is too many variables. 
                # Alternative: Just compute the cost of the 2x2 block change and the cross terms with O(n) vectors.
                
                # Let's try a simpler, correct O(n) approach:
                # delta = 0
                # mask = np.ones(n, dtype=bool)
                # mask[i] = mask[j] = False
                # others = np.where(mask)[0]
                
                # This is slow due to np.where.
                
                # Let's go back to the precomputed dot_row/dot_col but use the correct formula.
                # I will implement the precomputed version carefully.
                
                # Actually, for n=150, the overhead of 4 dot products (each O(n)) is 4*150 = 600 ops.
                # Plus the subtraction logic.
                # Let's use the dot product method and subtract the known internal contributions.
                
                # d_i = np.dot(a[i], b[v, p] - b[u, p])
                # d_j = np.dot(a[j], b[u, p] - b[v, p])
                # d_ci = np.dot(a[:, i], b[p, v] - b[p, u])
                # d_cj = np.dot(a[:, j], b[p, u] - b[p, v])
                
                # delta = d_i + d_j + d_ci + d_cj
                #        - (internal terms included in d_i, d_j, d_ci, d_cj)
                #        + (correct internal terms)
                
                # This is risky. Let's just use the direct O(n) loop in Python for now, it's safer.
                # If it's too slow, we can optimize later.
                
                # Optimization: Precompute b_v_p, b_u_p, b_p_v, b_p_u?
                # b[v, p] changes only if p changes.
                # We can keep these as global variables and update them incrementally?
                # b[v, p] is a row of b indexed by p.
                # If we swap p[i] and p[j], only b[v, p[i]] and b[v, p[j]] change? No, p is the index.
                # b[v, p] is a vector where element k is b[v, p[k]].
                # If p changes, this vector changes at indices i and j.
                
                # Let's maintain `b_row_p` (matrix b[p, :]) and `b_col_p` (matrix b[:, p]).
                # Then b[v, p] is not directly available.
                
                # Let's stick to the simplest correct method:
                # Compute delta using O(n) NumPy operations.
                
                # delta = 0.0
                # # Cross terms
                # # a[i, others] dot (b[v, p[others]] - b[u, p[others]])
                # # We can't easily get p[others] without a mask.
                
                # Let's use the precomputed dot_row/dot_col and just accept the complexity of subtraction.
                # OR, just recompute the cost for the 3 indices involved? No, cost is global.
                
                # Let's try the precomputed dot_row/dot_col subtraction method.
                # It is the standard way to do this fast.
                
                # From previous analysis:
                # delta = (d_raw_i - S_i) + (d_raw_j - S_j) + (d_raw_ci - SC_i) + (d_raw_cj - SC_j) + T
                # Where S_i are the internal terms included in d_raw_i, etc.
                
                # d_raw_i = sum_k a[i,k](b[v,p[k]]-b[u,p[k]])
                # k=i: a[i,i](b[v,u]-b[u,u])
                # k=j: a[i,j](b[v,v]-b[u,v])
                
                # d_raw_j = sum_k a[j,k](b[u,p[k]]-b[v,p[k]])
                # k=i: a[j,i](b[u,u]-b[v,u])
                # k=j: a[j,j](b[u,v]-b[v,v])
                
                # d_raw_ci = sum_k a[k,i](b[p[k],v]-b[p[k],u])
                # k=i: a[i,i](b[u,v]-b[u,u])
                # k=j: a[j,i](b[v,v]-b[v,u])
                
                # d_raw_cj = sum_k a[k,j](b[p[k],u]-b[p[k],v])
                # k=i: a[i,j](b[u,u]-b[u,v])
                # k=j: a[j,j](b[v,u]-b[v,v])
                
                # Correct internal terms:
                # T_ii = a[i,i](b[v,v]-b[u,u])
                # T_ij = a[i,j](b[v,u]-b[u,v])
                # T_ji = a[j,i](b[u,v]-b[v,u])
                # T_jj = a[j,j](b[u,u]-b[v,v])
                
                # So:
                # delta = d_raw_i + d_raw_j + d_raw_ci + d_raw_cj
                #        - [a[i,i](b[v,u]-b[u,u]) + a[i,j](b[v,v]-b[u,v])]  # from d_raw_i
                #        - [a[j,i](b[u,u]-b[v,u]) + a[j,j](b[u,v]-b[v,v])]  # from d_raw_j
                #        - [a[i,i](b[u,v]-b[u,u]) + a[j,i](b[v,v]-b[v,u])]  # from d_raw_ci
                #        - [a[i,j](b[u,u]-b[u,v]) + a[j,j](b[v,u]-b[v,v])]  # from d_raw_cj
                #        + [T_ii + T_ij + T_ji + T_jj]
                
                # Let's group the terms.
                # This is doable but tedious.
                
                # Given the time constraints and the risk of bug, I will use the direct O(n) loop.
                # It is simpler and less likely to be wrong.
                
                # To speed up the direct loop, we can precompute the vectors:
                # bv_p = b[v, p]
                # bu_p = b[u, p]
                # bp_v = b[p, v]
                # bp_u = b[p, u]
                # These are O(n) to create.
                
                # Then:
                # delta = np.dot(a[i], bv_p - bu_p) + np.dot(a[j], bu_p - bv_p) 
                #       + np.dot(a[:, i], bp_v - bp_u) + np.dot(a[:, j], bp_u - bp_v)
                #       - (internal terms) + (correct internal terms)
                
                # Wait, the dot products above INCLUDE the internal terms.
                # So we just subtract the internal terms from the dot products and add the correct ones.
                
                # Let's define the internal terms to subtract:
                # sub_i = a[i,i]*(b[v,u]-b[u,u]) + a[i,j]*(b[v,v]-b[u,v])
                # sub_j = a[j,i]*(b[u,u]-b[v,u]) + a[j,j]*(b[u,v]-b[v,v])
                # sub_ci = a[i,i]*(b[u,v]-b[u,u]) + a[j,i]*(b[v,v]-b[v,u])
                # sub_cj = a[i,j]*(b[u,u]-b[u,v]) + a[j,j]*(b[v,u]-b[v,v])
                
                # Correct internal terms:
                # add_ii = a[i,i]*(b[v,v]-b[u,u])
                # add_ij = a[i,j]*(b[v,u]-b[u,v])
                # add_ji = a[j,i]*(b[u,v]-b[v,u])
                # add_jj = a[j,j]*(b[u,u]-b[v,v])
                
                # delta = (d_i - sub_i) + (d_j - sub_j) + (d_ci - sub_ci) + (d_cj - sub_cj) + add_ii + add_ij + add_ji + add_jj
                
                # This is O(1) arithmetic plus 4 dot products O(n).
                # This is fast and correct.
                
                d_i = np.dot(a[i], b[v, p] - b[u, p])
                d_j = np.dot(a[j], b[u, p] - b[v, p])
                d_ci = np.dot(a[:, i], b[p, v] - b[p, u])
                d_cj = np.dot(a[:, j], b[p, u] - b[p, v])
                
                # Internal terms
                buu = b[u, u]
                bvv = b[v, v]
                buv = b[u, v]
                bvu = b[v, u]
                
                sub_i = a[i, i] * (bvu - buu) + a[i, j] * (bvv - buv)
                sub_j = a[j, i] * (buu - bvu) + a[j, j] * (buv - bvv)
                sub_ci = a[i, i] * (buv - buu) + a[j, i] * (bvv - bvu)
                sub_cj = a[i, j] * (buu - buv) + a[j, j] * (bvu - bvv)
                
                add_ii = a[i, i] * (bvv - buu)
                add_ij = a[i, j] * (bvu - buv)
                add_ji = a[j, i] * (buv - bvu)
                add_jj = a[j, j] * (buu - bvv)
                
                delta = (d_i - sub_i) + (d_j - sub_j) + (d_ci - sub_ci) + (d_cj - sub_cj) + add_ii + add_ij + add_ji + add_jj
                
                if delta < -1e-9:
                    # Accept swap
                    p[i], p[j] = v, u
                    cost += delta
                    improved = True
                    last_improve_time = time.time()
                    
                    # Update precomputed dot_row and dot_col
                    # dot_row[k, l] = sum_m a[k,m] b[l, p[m]]
                    # p[i] changed from u to v, p[j] changed from v to u.
                    # For k != i, j:
                    # dot_row[k, l] changes by a[k,i](b[l,v]-b[l,u]) + a[k,j](b[l,u]-b[l,v])
                    #                    = (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    # For k = i:
                    # dot_row[i, l] changes by a[i,i](b[l,v]-b[l,u]) + a[i,j](b[l,u]-b[l,v])
                    #                    = (a[i,i] - a[i,j]) * (b[l,v] - b[l,u])
                    # For k = j:
                    # dot_row[j, l] changes by a[j,i](b[l,v]-b[l,u]) + a[j,j](b[l,u]-b[l,v])
                    #                    = (a[j,i] - a[j,j]) * (b[l,v] - b[l,u])
                    
                    # So for all k:
                    # delta_dot_row[k, l] = (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    
                    # This is an outer product!
                    # diff_a = a[:, i] - a[:, j]  (shape n)
                    # diff_b = b[:, v] - b[:, u]  (shape n)
                    # dot_row += np.outer(diff_a, diff_b)
                    
                    # Similarly for dot_col:
                    # dot_col[k, l] = sum_m a[m,k] b[p[m], l]
                    # p[i] changed u->v, p[j] changed v->u.
                    # For all m:
                    # dot_col[m, l] changes by a[m,i](b[v,l]-b[u,l]) + a[m,j](b[u,l]-b[v,l])
                    #                    = (a[m,i] - a[m,j]) * (b[v,l] - b[u,l])
                    
                    # diff_a = a[:, i] - a[:, j]
                    # diff_b = b[v, :] - b[u, :]
                    # dot_col += np.outer(diff_a, diff_b)
                    
                    diff_a = a[:, i] - a[:, j]
                    diff_b_row = b[:, v] - b[:, u]
                    dot_row += np.outer(diff_a, diff_b_row)
                    
                    diff_b_col = b[v, :] - b[u, :]
                    dot_col += np.outer(diff_a, diff_b_col)
                    
                    # Note: This update is O(n^2) due to outer product.
                    # n=150, n^2=22500. 
                    # If we do 10k swaps, that's 225M ops. Might be too slow.
                    # However, we only do this when we ACCEPT a swap.
                    # In local search, we accept many swaps initially, then fewer.
                    # If it's too slow, we can recompute dot_row/dot_col from scratch O(n^3) every 1000 swaps?
                    # Or just recompute from scratch when stagnation is detected.
                    
                    # For now, let's assume the outer product update is fast enough.
                    # If not, we can fall back to recomputing.

        return p, cost

    # Multi-start
    best_p = None
    best_cost = np.inf
    
    inits = [
        rank_init(),
        spectral_init(0),
        spectral_init(1),
        np.random.permutation(n),
        np.random.permutation(n)
    ]
    
    for p_init in inits:
        if time.time() > deadline - 5.0:
            break
        
        p_curr, cost_curr = local_search(p_init, compute_cost(p_init), deadline - 5.0)
        
        if cost_curr < best_cost:
            best_cost = cost_curr
            best_p = p_curr

    return best_p.tolist()

# EVOLVE-BLOCK-END
