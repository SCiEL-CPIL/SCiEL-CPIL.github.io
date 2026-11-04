"""
Multi-Start 3-Opt QAP Solver for wil100.
Strategy:
1. Diversified Initialization: Generate multiple diverse starting points using:
   - Hungarian on linear surrogate (centrality alignment)
   - Random permutations
   - Spectral-like sorting (eigenvalue-based alignment)
2. Vectorized Best-Improvement Local Search (BILS):
   - Instead of Simulated Annealing (which accepts bad moves), use strict Best-Improvement.
   - Expand neighborhood from 2-opt (single swap) to 3-opt (triple swap).
   - 3-opt allows escaping local minima that are 2-opt stable but 3-opt unstable.
   - Vectorized delta calculation for 3-opt to maintain speed within 120s limit.
3. Iterated Local Search (ILS):
   - Perturb the best local optimum with a small number of random swaps.
   - Re-run BILS.
   - Keep the best solution found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _delta_swap(a, b, p, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    O(n) vectorized.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms
    # Sum over k != i, j of (a[i,k]-a[j,k]) * (b[pj,k]-b[pi,k]) + (a[k,i]-a[k,j]) * (b[k,pj]-b[k,pi])
    
    # Row difference: a[i, :] - a[j, :]
    # Col difference in B for fixed positions pj, pi: b[pj, :] - b[pi, :]
    # We need dot product of (a[i, :] - a[j, :]) and (b[pj, p] - b[pi, p])
    # But we must exclude k=i and k=j.
    
    diff_a_row = a[i, :] - a[j, :]
    diff_b_row_p = b[pj, p] - b[pi, p]
    
    # The dot product includes k=i and k=j.
    # Let's compute the full dot product and subtract the i, j terms.
    sum_T1 = np.dot(diff_a_row, diff_b_row_p)
    
    # Subtract k=i term: (a[i,i]-a[j,i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    # Subtract k=j term: (a[i,j]-a[j,j]) * (b[pj, pj] - b[pi, pj])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Column difference: a[:, i] - a[:, j]
    # Row difference in B for fixed positions pj, pi: b[:, pj] - b[:, pi]
    diff_a_col = a[:, i] - a[:, j]
    diff_b_col_p = b[p, pj] - b[p, pi]
    
    sum_T2 = np.dot(diff_a_col, diff_b_col_p)
    
    # Subtract k=i term: (a[i,i]-a[i,j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    # Subtract k=j term: (a[j,i]-a[j,j]) * (b[pj, pj] - b[pj, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _delta_3opt(a, b, p, i, j, k):
    """
    Compute delta cost for cyclically shifting positions i, j, k.
    p[i] -> p[j], p[j] -> p[k], p[k] -> p[i].
    O(n) vectorized.
    """
    pi = p[i]
    pj = p[j]
    pk = p[k]
    
    # New positions:
    # pos i gets pk
    # pos j gets pi
    # pos k gets pj
    
    # The cost change involves all pairs (x, y) where x or y is in {i, j, k}.
    # Delta = Cost(New) - Cost(Old) for these pairs.
    
    # Pairs involving i, j, k:
    # (i, i), (i, j), (i, k)
    # (j, i), (j, j), (j, k)
    # (k, i), (k, j), (k, k)
    # And cross terms with all other m.
    
    # Direct terms (within {i, j, k}):
    # Old: a[i,i]b[pi,pi] + a[i,j]b[pi,pj] + a[i,k]b[pi,pk]
    #      a[j,i]b[pj,pi] + a[j,j]b[pj,pj] + a[j,k]b[pj,pk]
    #      a[k,i]b[pk,pi] + a[k,j]b[pk,pj] + a[k,k]b[pk,pk]
    
    # New: a[i,i]b[pk,pk] + a[i,j]b[pk,pi] + a[i,k]b[pk,pj]
    #      a[j,i]b[pi,pk] + a[j,j]b[pi,pi] + a[j,k]b[pi,pj]
    #      a[k,i]b[pj,pk] + a[k,j]b[pj,pi] + a[k,k]b[pj,pj]
    
    delta_direct = (
        a[i,i]*(b[pk,pk]-b[pi,pi]) + a[i,j]*(b[pk,pi]-b[pi,pj]) + a[i,k]*(b[pk,pj]-b[pi,pk]) +
        a[j,i]*(b[pi,pk]-b[pj,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj]) + a[j,k]*(b[pi,pj]-b[pj,pk]) +
        a[k,i]*(b[pj,pk]-b[pk,pi]) + a[k,j]*(b[pj,pi]-b[pk,pj]) + a[k,k]*(b[pj,pj]-b[pk,pk])
    )
    
    # Cross terms with m not in {i, j, k}:
    # For each m, the change is:
    # (a[i,m] * (b[pk, p[m]] - b[pi, p[m]]) + a[m,i] * (b[p[m], pk] - b[p[m], pi]))
    # + (a[j,m] * (b[pi, p[m]] - b[pj, p[m]]) + a[m,j] * (b[p[m], pi] - b[p[m], pj]))
    # + (a[k,m] * (b[pj, p[m]] - b[pk, p[m]]) + a[m,k] * (b[p[m], pj] - b[p[m], pk]))
    
    # We can vectorize this.
    # Let's define the change in B rows/cols for each position.
    # For pos i: B row changes from pi to pk. B col changes from pi to pk.
    # For pos j: B row changes from pj to pi. B col changes from pj to pi.
    # For pos k: B row changes from pk to pj. B col changes from pk to pj.
    
    # Sum over m:
    # Term 1 (i-m): a[i, m] * (b[pk, p[m]] - b[pi, p[m]])
    # Term 2 (m-i): a[m, i] * (b[p[m], pk] - b[p[m], pi])
    # Term 3 (j-m): a[j, m] * (b[pi, p[m]] - b[pj, p[m]])
    # Term 4 (m-j): a[m, j] * (b[p[m], pi] - b[p[m], pj])
    # Term 5 (k-m): a[k, m] * (b[pj, p[m]] - b[pk, p[m]])
    # Term 6 (m-k): a[m, k] * (b[p[m], pj] - b[p[m], pk])
    
    # We can compute these as dot products.
    # T1 = dot(a[i, :], b[pk, p] - b[pi, p])
    # T2 = dot(a[:, i], b[p, pk] - b[p, pi])
    # T3 = dot(a[j, :], b[pi, p] - b[pj, p])
    # T4 = dot(a[:, j], b[p, pi] - b[p, pj])
    # T5 = dot(a[k, :], b[pj, p] - b[pk, p])
    # T6 = dot(a[:, k], b[p, pj] - b[p, pk])
    
    # However, these dot products include m=i, j, k. We must subtract those contributions.
    
    T1 = np.dot(a[i, :], b[pk, p] - b[pi, p])
    T2 = np.dot(a[:, i], b[p, pk] - b[p, pi])
    T3 = np.dot(a[j, :], b[pi, p] - b[pj, p])
    T4 = np.dot(a[:, j], b[p, pi] - b[p, pj])
    T5 = np.dot(a[k, :], b[pj, p] - b[pk, p])
    T6 = np.dot(a[:, k], b[p, pj] - b[p, pk])
    
    # Subtract m=i, j, k from each T.
    # This is tedious but necessary for correctness.
    # T1 (i-m): m=i -> a[i,i](b[pk,pi]-b[pi,pi]), m=j -> a[i,j](b[pk,pj]-b[pi,pj]), m=k -> a[i,k](b[pk,pk]-b[pi,pk])
    T1 -= a[i,i]*(b[pk,pi]-b[pi,pi]) + a[i,j]*(b[pk,pj]-b[pi,pj]) + a[i,k]*(b[pk,pk]-b[pi,pk])
    
    # T2 (m-i): m=i -> a[i,i](b[pi,pk]-b[pi,pi]), m=j -> a[j,i](b[pj,pk]-b[pj,pi]), m=k -> a[k,i](b[pk,pk]-b[pk,pi])
    T2 -= a[i,i]*(b[pi,pk]-b[pi,pi]) + a[j,i]*(b[pj,pk]-b[pj,pi]) + a[k,i]*(b[pk,pk]-b[pk,pi])
    
    # T3 (j-m): m=i -> a[j,i](b[pi,pi]-b[pj,pi]), m=j -> a[j,j](b[pi,pj]-b[pj,pj]), m=k -> a[j,k](b[pi,pk]-b[pj,pk])
    T3 -= a[j,i]*(b[pi,pi]-b[pj,pi]) + a[j,j]*(b[pi,pj]-b[pj,pj]) + a[j,k]*(b[pi,pk]-b[pj,pk])
    
    # T4 (m-j): m=i -> a[i,j](b[pi,pi]-b[pi,pj]), m=j -> a[j,j](b[pj,pi]-b[pj,pj]), m=k -> a[k,j](b[pk,pi]-b[pk,pj])
    T4 -= a[i,j]*(b[pi,pi]-b[pi,pj]) + a[j,j]*(b[pj,pi]-b[pj,pj]) + a[k,j]*(b[pk,pi]-b[pk,pj])
    
    # T5 (k-m): m=i -> a[k,i](b[pj,pi]-b[pk,pi]), m=j -> a[k,j](b[pj,pj]-b[pk,pj]), m=k -> a[k,k](b[pj,pk]-b[pk,pk])
    T5 -= a[k,i]*(b[pj,pi]-b[pk,pi]) + a[k,j]*(b[pj,pj]-b[pk,pj]) + a[k,k]*(b[pj,pk]-b[pk,pk])
    
    # T6 (m-k): m=i -> a[i,k](b[pi,pj]-b[pi,pk]), m=j -> a[j,k](b[pj,pi]-b[pj,pk]), m=k -> a[k,k](b[pk,pj]-b[pk,pk])
    T6 -= a[i,k]*(b[pi,pj]-b[pi,pk]) + a[j,k](b[pj,pi]-b[pj,pk]) + a[k,k](b[pk,pj]-b[pk,pk])
    
    delta_cross = T1 + T2 + T3 + T4 + T5 + T6
    
    return int(delta_direct + delta_cross)

def _hungarian_initial(a, b):
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    C = -np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _spectral_initial(a, b):
    """
    Sort facilities by eigenvalue of A and locations by eigenvalue of B.
    Aligns high eigenvector components.
    """
    n = len(a)
    # Use power iteration or simple eigen decomposition for small n
    # For n=100, np.linalg.eigh is fine.
    evals_a, evecs_a = np.linalg.eigh(a)
    evals_b, evecs_b = np.linalg.eigh(b)
    
    # Take the eigenvector corresponding to the largest eigenvalue (or most negative for minimization? 
    # Usually largest magnitude. Let's try aligning the principal components.)
    # For QAP, aligning the "directions" of the matrices often helps.
    # Let's sort indices by the absolute value of the components of the principal eigenvector.
    
    v_a = evecs_a[:, -1] # Principal eigenvector
    v_b = evecs_b[:, -1]
    
    # Sort facilities by v_a
    idx_a = np.argsort(v_a)
    # Sort locations by v_b
    idx_b = np.argsort(v_b)
    
    p = np.empty(n, dtype=np.int64)
    # Assign facility idx_a[i] to location idx_b[i]
    p[idx_a] = idx_b
    
    return p

def _local_search_3opt(a, b, p, time_limit, max_passes=10):
    """
    Best-Improvement Local Search with 3-opt moves.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    for _ in range(max_passes):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        # Iterate over all 3-opt moves
        # O(n^3) moves. For n=100, 100^3 = 1,000,000 moves.
        # Each move is O(n) vectorized.
        # 1M * 100 ops = 100M ops. Might be too slow for a single pass in Python loop.
        # We need to be smart. Maybe restrict to 2-opt first, then 3-opt?
        # Or vectorize the 3-opt search more aggressively?
        
        # Alternative: Use 2-opt for speed, and 3-opt for escape.
        # Let's do 2-opt best improvement first.
        
        # 2-opt pass
        best_delta = 0
        best_i, best_j = -1, -1
        for i in range(n):
            if time.time() - start_time > time_limit:
                return best_p, best_cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i != -1:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            # No 2-opt improvement, try 3-opt
            best_delta_3 = 0
            best_i3, best_j3, best_k3 = -1, -1, -1
            # Sample some 3-opt moves to save time? Or do full?
            # Full 3-opt is expensive. Let's do a limited 3-opt search.
            # Random sampling of 3-opt moves.
            for _ in range(500): # 500 random 3-opt trials
                if time.time() - start_time > time_limit:
                    return best_p, best_cost
                i = np.random.randint(n)
                j = np.random.randint(n)
                k = np.random.randint(n)
                if i == j or j == k or i == k:
                    continue
                delta = _delta_3opt(a, b, p, i, j, k)
                if delta < best_delta_3:
                    best_delta_3 = delta
                    best_i3, best_j3, best_k3 = i, j, k
            
            if best_i3 != -1:
                # Apply 3-opt: i->j, j->k, k->i
                pi, pj, pk = p[best_i3], p[best_j3], p[best_k3]
                p[best_i3] = pk
                p[best_j3] = pi
                p[best_k3] = pj
                current_cost += best_delta_3
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                improved = True
        
        if not improved:
            break
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    np.random.seed(42)
    
    # Generate diverse initial solutions
    inits = []
    
    # 1. Hungarian
    p_hun = _hungarian_initial(a, b)
    inits.append(p_hun)
    
    # 2. Spectral
    p_spec = _spectral_initial(a, b)
    inits.append(p_spec)
    
    # 3. Random
    p_rand = np.random.permutation(n)
    inits.append(p_rand)
    
    # 4. Reverse Spectral
    p_rev_spec = p_spec[::-1].copy()
    inits.append(p_rev_spec)
    
    best_p = None
    best_cost = float('inf')
    
    # Iterate through initializations
    for p_init in inits:
        if time.time() - start_time > total_deadline:
            break
            
        # Allocate time for this start
        # If we have 4 starts, give each ~25s
        # But if we find a very good one early, we might want to spend more time polishing it?
        # For simplicity, fixed time per start.
        time_per_start = (total_deadline - (time.time() - start_time)) / (len(inits) - inits.index(p_init))
        
        p_ls, cost_ls = _local_search_3opt(a, b, p_init, time_per_start, max_passes=5)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Final Polish: Use remaining time on the best solution found
    if best_p is not None:
        remaining_time = total_deadline - (time.time() - start_time)
        if remaining_time > 5.0:
            p_final, cost_final = _local_search_3opt(a, b, best_p, remaining_time - 1.0, max_passes=20)
            if cost_final < best_cost:
                best_cost = cost_final
                best_p = p_final
    
    if best_p is None:
        # Fallback
        return np.random.permutation(n).tolist()
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
