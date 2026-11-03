"""Spectral Initialization + Vectorized ILS for QAPLIB wil100."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_init(a, b, k=10):
    """
    Initialize permutation using spectral embedding.
    Projects facilities and locations onto the top k eigenvectors of their respective
    matrices and matches them based on Euclidean distance in this subspace.
    This captures global structure better than simple rank matching.
    """
    n = len(a)
    
    # Ensure matrices are symmetric for eigendecomposition (QAP matrices are symmetric)
    # If not strictly symmetric due to noise, symmetrize
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Compute eigenvectors
    # np.linalg.eigh is for symmetric matrices, returns eigenvalues in ascending order
    evals_a, evecs_a = np.linalg.eigh(a_sym)
    evals_b, evecs_b = np.linalg.eigh(b_sym)
    
    # Take top k eigenvectors (largest eigenvalues)
    # eigh returns ascending, so top k are the last k
    top_k_a = evecs_a[:, -k:]
    top_k_b = evecs_b[:, -k:]
    
    # Project facilities (rows of A) and locations (rows of B) onto these subspace
    # Facility i is represented by row i of A. In spectral embedding for QAP,
    # we typically map facility i to the projection of its row.
    # Let's use the rows of the eigenvector matrix directly as coordinates.
    # Wait, the eigenvectors are columns.
    # The "position" of facility i in the eigenspace is given by the i-th row of the eigenvector matrix.
    
    coords_f = top_k_a  # Shape (n, k)
    coords_l = top_k_b  # Shape (n, k)
    
    # We want to minimize sum a[i,j] b[p[i],p[j]].
    # High flow pairs (large a[i,j]) should be assigned to low distance pairs (small b[p[i],p[j]]).
    # In the spectral embedding, "high flow" facilities tend to be far apart in the embedding?
    # Actually, for Laplacian-like matrices, large eigenvalues correspond to smooth functions.
    # Let's try matching based on distance in the embedded space.
    # If A and B are similar, we want p[i] to be close to i in the embedding?
    # Or rather, we want to align the geometric structures.
    
    # Compute pairwise distances between all facilities and all locations in the k-dim space
    # dist_f_l[i, j] = || coords_f[i] - coords_l[j] ||^2
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    
    norm_f_sq = np.sum(coords_f**2, axis=1, keepdims=True) # (n, 1)
    norm_l_sq = np.sum(coords_l**2, axis=1, keepdims=True).T # (1, n)
    dot_fl = coords_f @ coords_l.T # (n, n)
    
    dist_sq = norm_f_sq + norm_l_sq - 2 * dot_fl
    
    # We want to assign facility i to location p[i] such that the "structure" is preserved.
    # A common heuristic is to minimize the sum of distances between matched points.
    # This is the Linear Assignment Problem (LAP) on the cost matrix dist_sq.
    # However, scipy.optimize.linear_sum_assignment is available.
    
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(dist_sq)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except ImportError:
        # Fallback: greedy matching by sorted distances
        # This is less optimal but robust
        p = np.zeros(n, dtype=np.int64)
        used_locs = np.zeros(n, dtype=bool)
        # Sort all pairs by distance
        flat_idx = np.argsort(dist_sq)
        for idx in flat_idx:
            i = idx // n
            j = idx % n
            if not used_locs[j]:
                p[i] = j
                used_locs[j] = True
        return p

def _compute_swap_deltas(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns a scalar delta.
    """
    pi = p[i]
    pj = p[j]
    
    # If pi == pj, delta is 0 (should not happen in valid perm)
    if pi == pj:
        return 0.0
        
    delta = 0.0
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Mutual terms
    delta += a[i, j] * (b[pj, pi] - b[pi, pj])
    delta += a[j, i] * (b[pi, pj] - b[pj, pi])
    
    # Cross terms with all other indices k != i, j
    # Sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    # Sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    
    # Vectorized calculation
    # Create a mask for k != i, j
    # It's faster to just compute over all k and subtract i and j contributions manually
    # Or use slicing.
    
    # Let's compute over all k first, then correct for i and j if they were included.
    # But the formula derived earlier excludes i and j.
    
    # Let's stick to the explicit exclusion for correctness, using boolean indexing or slicing.
    # For n=100, boolean indexing is fast enough.
    
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    
    if np.any(mask):
        p_other = p[mask]
        
        # Term 1: sum over k (acting as l in previous derivation)
        # (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
        diff_a_row = a[i, mask] - a[j, mask]
        diff_b_col = b[pj, p_other] - b[pi, p_other]
        term1 = np.sum(diff_a_row * diff_b_col)
        
        # Term 2: sum over k (acting as k in previous derivation)
        # (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
        diff_a_col = a[mask, i] - a[mask, j]
        diff_b_row = b[p_other, pj] - b[p_other, pi]
        term2 = np.sum(diff_a_col * diff_b_row)
        
        delta += term1 + term2
        
    return delta

def _local_search(a, b, p, time_limit):
    """
    Fastest Descent Local Search using random sampling of moves.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500 # Stop if stuck for a while
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        
        if i == j:
            continue
            
        delta = _compute_swap_deltas(a, b, p, i, j)
        
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
                
    return best_p, best_cost

def _perturb(p, n, intensity=2):
    """
    Perturb the permutation by performing 'intensity' random swaps.
    """
    p_new = p.copy()
    for _ in range(intensity):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new

def solve(a, b):
    n = len(a)
    deadline = 145.0 # Safety margin before 150s limit
    start = time.time()
    
    np.random.seed(42) # Fixed seed for reproducibility of the "strategy", though spectral is deterministic
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Generate diverse initial solutions using Spectral Initialization
    # We can vary the spectral subspace dimension or add noise to get diversity
    inits = []
    
    # Standard spectral
    p_spec = _spectral_init(a, b, k=10)
    inits.append(p_spec)
    
    # Spectral with different k
    if n > 20:
        p_spec2 = _spectral_init(a, b, k=20)
        inits.append(p_spec2)
        
    # Random initialization for safety/diversity
    inits.append(np.random.permutation(n))
    
    # Phase 2: Iterated Local Search (ILS)
    # For each initial solution, run LS, then perturb and continue
    
    for p_init in inits:
        if time.time() - start > deadline - 5:
            break
            
        p_curr = p_init.copy()
        curr_cost = _qap_cost(a, b, p_curr)
        
        # Initial Local Search
        ls_time = min(5.0, max(1.0, (deadline - (time.time() - start)) * 0.2))
        p_ls, cost_ls = _local_search(a, b, p_curr, ls_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
        # ILS Loop
        while time.time() - start < deadline - 5:
            # Perturb the best found so far (or current best)
            p_perturbed = _perturb(best_p, n, intensity=3)
            
            # Local Search from perturbed
            # Allocate remaining time proportionally, but keep a buffer
            time_remaining = deadline - (time.time() - start) - 2
            if time_remaining < 1:
                break
                
            ls_time = min(2.0, time_remaining * 0.8)
            p_ls2, cost_ls2 = _local_search(a, b, p_perturbed, ls_time)
            
            if cost_ls2 < best_cost:
                best_cost = cost_ls2
                best_p = p_ls2.copy()
            else:
                # If no improvement, we could try a larger perturbation or different start
                # For now, just continue loop to try again with new perturbation
                pass

    # Final Polish: Run local search on the absolute best found with remaining time
    if best_p is not None:
        time_remaining = deadline - (time.time() - start)
        if time_remaining > 1:
            p_final, cost_final = _local_search(a, b, best_p.copy(), time_remaining)
            if cost_final < best_cost:
                best_p = p_final
                best_cost = cost_final

    return best_p.tolist()

# EVOLVE-BLOCK-END
