"""
Hungarian Initialization + VND (Variable Neighborhood Descent) solver for QAPLIB wil100.
Strategy:
1. Compute facility flow centrality (row sums of A) and location distance centrality (row sums of B).
2. Construct a linear surrogate cost matrix C[i, j] = -a_row_sums[i] * b_row_sums[j] 
   (minimizing C maximizes the product of centralities, aligning high-flow facilities 
   with high-distance locations in a globally optimal linear assignment).
3. Solve the assignment problem using the Hungarian algorithm (scipy.optimize.linear_sum_assignment) 
   to get a structurally coherent initial permutation p.
4. Perform Variable Neighborhood Descent (VND) with steepest-descent local search:
   - N1: All 2-swap moves (steepest descent)
   - N2: 3-swap moves (cyclic shifts of 3 positions)
   - N3: Reversal moves (reverse a segment)
   - Cycle through neighborhoods; when no improvement in current neighborhood, move to next.
   - When all neighborhoods exhausted, apply random perturbation and restart VND.
5. Return the best permutation found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms (usually 0 for QAP but handle generally)
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    # Term 1 over all k:
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i: (a[i,i]-a[j,i]) * (b[pj,pi]-b[pi,pi])
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    # Subtract k=j: (a[i,j]-a[j,j]) * (b[pj,pj]-b[pi,pj])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Term 2 over all k:
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i: (a[i,i]-a[i,j]) * (b[pi,pj]-b[pi,pi])
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    # Subtract k=j: (a[j,i]-a[j,j]) * (b[pj,pj]-b[pj,pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian algorithm on a linear surrogate.
    Surrogate cost: C[i, j] = -a_row_sums[i] * b_row_sums[j]
    Minimizing C maximizes the product of flow centrality and distance centrality.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    C = -np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _steepest_swap_search(a, b, p, current_cost, deadline):
    """
    Steepest-descent 2-swap local search.
    Repeatedly find the best improving swap and apply it until no improvement exists.
    Returns (p, current_cost, num_swaps_made).
    """
    n = len(p)
    num_swaps = 0
    
    while time.time() < deadline:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Vectorized: compute deltas for all pairs involving each i
        # For efficiency, we'll scan pairs in batches
        for i in range(n):
            if time.time() >= deadline:
                break
            pi = p[i]
            # Compute delta for swapping i with all j > i
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break  # Local optimum
        
        # Apply the best swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        num_swaps += 1
    
    return p, current_cost, num_swaps

def _three_swap_delta(a, b, p, i, j, k):
    """
    Compute the cost change when rotating positions i, j, k:
    p[i], p[j], p[k] -> p[j], p[k], p[i]
    i.e., element at i goes to j, j goes to k, k goes to i.
    """
    pi = p[i]
    pj = p[j]
    pk = p[k]
    
    # New assignment: pos i gets pj, pos j gets pk, pos k gets pi
    # Old: pos i had pi, pos j had pj, pos k had pk
    
    # Compute delta = new_cost - old_cost
    # Terms involving positions i, j, k change
    
    # For each pair (x, y) where x or y is in {i, j, k}, the contribution changes.
    
    # Let's compute the old and new contributions for all pairs involving {i, j, k}
    
    # Old contributions:
    # a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[i,k]*b[pi,pk]
    # a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj] + a[j,k]*b[pj,pk]
    # a[k,i]*b[pk,pi] + a[k,j]*b[pk,pj] + a[k,k]*b[pk,pk]
    
    # New contributions:
    # a[i,i]*b[pj,pj] + a[i,j]*b[pj,pk] + a[i,k]*b[pj,pi]
    # a[j,i]*b[pk,pi] + a[j,j]*b[pk,pk] + a[j,k]*b[pk,pj]
    # a[k,i]*b[pi,pj] + a[k,j]*b[pi,pk] + a[k,k]*b[pi,pi]
    
    old = (a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[i,k]*b[pi,pk] +
           a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj] + a[j,k]*b[pj,pk] +
           a[k,i]*b[pk,pi] + a[k,j]*b[pk,pj] + a[k,k]*b[pk,pk])
    
    new = (a[i,i]*b[pj,pj] + a[i,j]*b[pj,pk] + a[i,k]*b[pj,pi] +
           a[j,i]*b[pk,pi] + a[j,j]*b[pk,pk] + a[j,k]*b[pk,pj] +
           a[k,i]*b[pi,pj] + a[k,j]*b[pi,pk] + a[k,k]*b[pi,pi])
    
    # But we also need to account for interactions with other positions
    # For each other position m (not i,j,k), the contribution involving position i changes from
    # a[i,m]*b[pi,p[m]] + a[m,i]*b[p[m],pi] to a[i,m]*b[pj,p[m]] + a[m,i]*b[p[m],pj]
    # Similarly for j and k.
    
    others = [m for m in range(len(p)) if m not in (i, j, k)]
    if others:
        others_arr = np.array(others, dtype=np.int64)
        po = p[others_arr]
        
        # Old cross terms with others:
        old_cross = (np.dot(a[i, others_arr], b[pi, po]) + np.dot(a[others_arr, i], b[po, pi]) +
                     np.dot(a[j, others_arr], b[pj, po]) + np.dot(a[others_arr, j], b[po, pj]) +
                     np.dot(a[k, others_arr], b[pk, po]) + np.dot(a[others_arr, k], b[po, pk]))
        
        # New cross terms with others:
        new_cross = (np.dot(a[i, others_arr], b[pj, po]) + np.dot(a[others_arr, i], b[po, pj]) +
                     np.dot(a[j, others_arr], b[pk, po]) + np.dot(a[others_arr, j], b[po, pk]) +
                     np.dot(a[k, others_arr], b[pi, po]) + np.dot(a[others_arr, k], b[po, pi]))
        
        old += old_cross
        new += new_cross
    
    return int(new - old)

def _reversal_delta(a, b, p, i, j):
    """
    Compute the cost change when reversing the segment p[i..j] (inclusive).
    """
    if i >= j:
        return 0
    
    n = len(p)
    # Create a copy and reverse
    p_new = p.copy()
    p_new[i:j+1] = p_new[i:j+1][::-1]
    
    cost_old = _qap_cost(a, b, p)
    cost_new = _qap_cost(a, b, p_new)
    
    return int(cost_new - cost_old)

def _vnd_search(a, b, p, current_cost, deadline, max_vnd_rounds=50):
    """
    Variable Neighborhood Descent with neighborhoods:
    N1: 2-swap (steepest)
    N2: 3-swap (cyclic)
    N3: Reversal
    """
    n = len(p)
    best_p = p.copy()
    best_cost = current_cost
    
    for vnd_round in range(max_vnd_rounds):
        if time.time() >= deadline:
            break
        
        improved = False
        
        # N1: Steepest 2-swap
        p, current_cost, num_swaps = _steepest_swap_search(a, b, p, current_cost, deadline)
        if num_swaps > 0:
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        if time.time() >= deadline:
            break
        
        # N2: 3-swap (sampled for efficiency)
        n2_improved = False
        # Try a limited number of 3-swaps
        num_3swap_tries = min(500, n * (n-1) * (n-2) // 6)
        for _ in range(num_3swap_tries):
            if time.time() >= deadline:
                break
            idx = np.random.choice(n, size=3, replace=False)
            i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
            delta = _three_swap_delta(a, b, p, i, j, k)
            if delta < 0:
                # Apply rotation: i->j, j->k, k->i
                p[i], p[j], p[k] = p[j], p[k], p[i]
                current_cost += delta
                n2_improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                break  # First improvement for 3-swap
        
        if n2_improved:
            improved = True
        
        if time.time() >= deadline:
            break
        
        # N3: Reversal (sampled)
        n3_improved = False
        num_rev_tries = min(200, n * (n-1) // 2)
        for _ in range(num_rev_tries):
            if time.time() >= deadline:
                break
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i > j:
                i, j = j, i
            if j - i < 2:
                continue
            delta = _reversal_delta(a, b, p, i, j)
            if delta < 0:
                p[i:j+1] = p[i:j+1][::-1]
                current_cost += delta
                n3_improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                break  # First improvement for reversal
        
        if n3_improved:
            improved = True
        
        if not improved:
            break  # All neighborhoods exhausted, local optimum
    
    return best_p, best_cost

def _perturb(p, num_swaps=5):
    """Apply random swaps to perturb the permutation."""
    n = len(p)
    p = p.copy()
    for _ in range(num_swaps):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 150s limit
    start_time = time.time()
    
    np.random.seed(42)  # Fixed seed for reproducibility within a run
    
    # Phase 1: Hungarian initialization
    p = _hungarian_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: VND with restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() >= total_deadline:
            break
        
        # Run VND
        p, current_cost = _vnd_search(a, b, p, current_cost, total_deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Check time
        if time.time() >= total_deadline:
            break
        
        # Perturb for next restart
        p = _perturb(best_p, num_swaps=3 + restart)
        current_cost = _qap_cost(a, b, p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
