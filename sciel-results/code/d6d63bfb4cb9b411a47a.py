"""
Strategy: Spectral Bisection Initialization with Iterated Local Search (Simulated Annealing).
1. Initialize using the Fiedler vector of the flow matrix to create a structurally informed permutation.
2. Perform 2-opt local search to reach a local optimum.
3. Apply random perturbations and use Simulated Annealing to accept uphill moves.
4. Repeat until time limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym):
    """
    Compute the change in cost when swapping facilities at positions i and j.
    p is the permutation array where p[k] is the facility at location k? 
    Wait, contract says: "Facility i is placed at location p[i]".
    So p[i] is the location of facility i.
    Cost = sum_{i,j} a[i, j] * b[p[i], p[j]].
    
    Let's verify the delta formula again carefully.
    Let u = p[i], v = p[j].
    We swap the locations of facility i and facility j.
    New p'[i] = v, p'[j] = u.
    
    Delta = C(p') - C(p).
    
    The terms that change are those involving i or j in the first or second index of A, 
    which correspond to locations u or v in B.
    
    Using the standard fast swap delta for QAP (assuming B is symmetric, which is standard for distance matrices):
    Let r = i, s = j.
    Let x = p[r], y = p[s].
    
    Delta = sum_{k != r, s} [ a[r,k](b[y,p[k]] - b[x,p[k]]) + a[k,r](b[p[k],y] - b[p[k],x])
                           + a[s,k](b[x,p[k]] - b[y,p[k]]) + a[k,s](b[p[k],x] - b[p[k],y]) ]
          + (terms for k=r, s)
    
    If B is symmetric: b[p[k],y] = b[y,p[k]].
    Delta = sum_{k != r, s} [ (a[r,k] + a[k,r] - a[s,k] - a[k,s]) * (b[y,p[k]] - b[x,p[k]]) ]
          + a[r,r](b[y,y] - b[x,x]) + a[s,s](b[x,x] - b[y,y])
          + a[r,s](b[y,x] - b[x,y]) + a[s,r](b[x,y] - b[y,x])
    
    Since b is symmetric, b[y,x] = b[x,y], so the cross terms a[r,s] and a[s,r] cancel out in the difference?
    Wait:
    Old cross term: a[r,s]b[x,y] + a[s,r]b[y,x].
    New cross term: a[r,s]b[y,x] + a[s,r]b[x,y].
    If b is symmetric, b[x,y]=b[y,x], so Old == New. Cross terms contribute 0 to delta.
    
    So Delta = sum_{k != r, s} (a_sym[r,k] - a_sym[s,k]) * (b[y,p[k]] - b[x,p[k]])
              + (a[r,r] - a[s,s]) * (b[y,y] - b[x,x])
    
    We can vectorize the sum over k including r and s, then correct?
    Let's use the vectorized form over all k:
    diff_b = b[y, p] - b[x, p]
    coeff = a_sym[r, :] - a_sym[s, :]
    delta_sum = np.dot(coeff, diff_b)
    
    This sum includes k=r and k=s.
    For k=r: coeff[r] = a_sym[r,r] - a_sym[s,r] = 2a[r,r] - a[s,r] - a[r,s].
    diff_b[r] = b[y, p[r]] - b[x, p[r]] = b[y, x] - b[x, x].
    Term_r = (2a[r,r] - a[s,r] - a[r,s]) * (b[y, x] - b[x, x])
    
    For k=s: coeff[s] = a_sym[r,s] - a_sym[s,s] = a[r,s] + a[s,r] - 2a[s,s].
    diff_b[s] = b[y, p[s]] - b[x, p[s]] = b[y, y] - b[x, y].
    Term_s = (a[r,s] + a[s,r] - 2a[s,s]) * (b[y, y] - b[x, y])
    
    The manual derivation gave:
    Delta_manual = sum_{k != r, s} ... + (a[r,r] - a[s,s]) * (b[y,y] - b[x,x])
    
    Let's check if the vectorized sum equals the manual sum.
    Vectorized Sum = Delta_manual + Term_r + Term_s.
    
    We need to subtract Term_r and Term_s and add the correct manual self-terms.
    Actually, it is safer to just compute the delta using the full vectorized formula and then correct for the specific self-terms if the derivation above is slightly off.
    
    Alternative robust method:
    delta = 0
    # k != r, s
    mask = np.ones(n, dtype=bool)
    mask[r] = False
    mask[s] = False
    delta += np.dot(a_sym[r, mask] - a_sym[s, mask], b[y, p[mask]] - b[x, p[mask]])
    
    # Self terms
    delta += a[r,r] * (b[y,y] - b[x,x])
    delta += a[s,s] * (b[x,x] - b[y,y])
    
    # Cross terms (cancel if b symmetric)
    # delta += a[r,s] * (b[y,x] - b[x,y]) + a[s,r] * (b[x,y] - b[y,x])
    # If b is symmetric, this is 0.
    
    This is O(N) and robust.
    """
    r = i
    s = j
    x = p[r]
    y = p[s]
    
    if x == y:
        return 0
        
    n = len(p)
    # Vectorized calculation for k != r, s
    # We can compute the full dot product and subtract the r,s terms, or use masking.
    # Masking creates a new array, which is slow. 
    # Better: Compute full dot product, then adjust.
    
    diff_b = b[y, p] - b[x, p]
    coeff = a_sym[r, :] - a_sym[s, :]
    
    delta = np.dot(coeff, diff_b)
    
    # Subtract the contributions of k=r and k=s from the dot product
    # and add the correct manual self-terms.
    
    # Contribution of k=r in dot product:
    # coeff[r] * diff_b[r] = (a_sym[r,r] - a_sym[s,r]) * (b[y,x] - b[x,x])
    # = (2a[r,r] - a[r,s] - a[s,r]) * (b[y,x] - b[x,x])
    
    # Contribution of k=s in dot product:
    # coeff[s] * diff_b[s] = (a_sym[r,s] - a_sym[s,s]) * (b[y,y] - b[x,y])
    # = (a[r,s] + a[s,r] - 2a[s,s]) * (b[y,y] - b[x,y])
    
    # Correct manual self terms:
    # a[r,r](b[y,y] - b[x,x]) + a[s,s](b[x,x] - b[y,y])
    
    # So we do:
    # delta -= (2a[r,r] - a[r,s] - a[s,r]) * (b[y,x] - b[x,x])
    # delta -= (a[r,s] + a[s,r] - 2a[s,s]) * (b[y,y] - b[x,y])
    # delta += a[r,r] * (b[y,y] - b[x,x])
    # delta += a[s,s] * (b[x,x] - b[y,y])
    
    # Let's simplify the correction.
    # This is getting messy. Let's stick to the masking method for correctness, 
    # or just use the full dot product if the formula is known to be exact.
    # Actually, for N=256, even a simple loop is fast, but vectorized is better.
    # Let's use the masking approach but pre-allocate? No, just do the full dot and correct.
    
    # Correction for k=r:
    delta -= (a_sym[r,r] - a_sym[s,r]) * (b[y,x] - b[x,x])
    # Correction for k=s:
    delta -= (a_sym[r,s] - a_sym[s,s]) * (b[y,y] - b[x,y])
    
    # Add correct self terms:
    delta += a[r,r] * (b[y,y] - b[x,x])
    delta += a[s,s] * (b[x,x] - b[y,y])
    
    # Cross terms are 0 if b is symmetric.
    
    return delta

def _spectral_init(a, n):
    """
    Generate an initial permutation using spectral bisection.
    We compute the eigenvector of the Laplacian of the flow matrix.
    Facilities are ordered by the values of this eigenvector.
    """
    # Laplacian L = D - A
    # D is diagonal matrix of degrees
    # A is the flow matrix
    # We want to partition the graph defined by A.
    
    # For simplicity and speed, we can use the second smallest eigenvector of L.
    # But computing eigenvectors of a 256x256 matrix is fast enough.
    
    # However, QAP is about matching structure. 
    # A simpler heuristic: Sort facilities by their row sums? 
    # Or use the eigenvector of A itself?
    
    # Let's try: Compute the eigenvector corresponding to the largest eigenvalue of A.
    # This gives a "principal component" of the flow structure.
    # Sort facilities by this eigenvector.
    
    # Actually, standard spectral bisection uses the Fiedler vector (2nd smallest eigenvector of L).
    # Let's implement a simple version.
    
    # 1. Compute L
    # 2. Find 2nd smallest eigenvector
    # 3. Sort indices by eigenvector values
    
    # To save time, we can use a random projection or just row sums if spectral is too slow.
    # 256x256 eigenvector is ~0.1s. Acceptable.
    
    # But wait, the distance matrix B also has structure.
    # The "ideal" assignment aligns the "shape" of A with the "shape" of B.
    
    # Let's use the eigenvectors of A and B.
    # Project facilities onto the top eigenvectors of A.
    # Project locations onto the top eigenvectors of B.
    # Match the closest points?
    
    # Simpler: 
    # 1. Get Fiedler vector of A (facilities).
    # 2. Get Fiedler vector of B (locations).
    # 3. Sort facilities by Fiedler(A).
    # 4. Sort locations by Fiedler(B).
    # 5. Assign sorted facility k to sorted location k.
    
    # This assumes the "linear" structure of the spectral embedding should be preserved.
    
    # Let's implement this.
    
    def get_fiedler(mat):
        # Symmetrize
        m = (mat + mat.T) / 2.0
        # Laplacian
        d = np.sum(m, axis=1)
        L = np.diag(d) - m
        # Find 2nd smallest eigenvector
        # Use scipy? No, numpy only.
        # np.linalg.eigh is for symmetric.
        vals, vecs = np.linalg.eigh(L)
        # Sort by eigenvalues
        idx = np.argsort(vals)
        # 2nd smallest is idx[1]
        fiedler = vecs[:, idx[1]]
        return fiedler
    
    try:
        f_fac = get_fiedler(a)
    except:
        f_fac = a.sum(axis=1)
        
    # We don't have B here in this function signature, but we can pass it or compute in main.
    # Let's just use the flow structure for now to order facilities.
    # Then in main, we will order locations by their "centrality" or similar spectral property.
    
    # Actually, let's return the sorted indices of facilities based on f_fac.
    return np.argsort(f_fac)

def _solve_tai256c(a, b, time_limit=115.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    
    # 1. Initialization
    # Get facility order from spectral embedding of A
    fac_order = _spectral_init(a, n)
    
    # Get location order from spectral embedding of B
    # We want to align the structure.
    # Let's use the same logic for B.
    def get_fiedler_b(mat):
        m = (mat + mat.T) / 2.0
        d = np.sum(m, axis=1)
        L = np.diag(d) - m
        vals, vecs = np.linalg.eigh(L)
        idx = np.argsort(vals)
        return vecs[:, idx[1]]
    
    try:
        f_loc = get_fiedler_b(b)
    except:
        f_loc = b.sum(axis=1)
        
    loc_order = np.argsort(f_loc)
    
    # Initial permutation: Facility fac_order[k] gets location loc_order[k]
    p = np.empty(n, dtype=np.int64)
    for k in range(n):
        p[fac_order[k]] = loc_order[k]
        
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # 2. Local Search (2-opt)
    def local_search(p, cost):
        improved = True
        while improved and time.time() - start_time < time_limit - 5.0:
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit - 5.0:
                    break
                for j in range(i + 1, n):
                    delta = _swap_delta(a, b, p, i, j, a_sym)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
        return p, cost

    best_p, best_cost = local_search(best_p, best_cost)
    
    # 3. Iterated Local Search with Simulated Annealing
    T = 1000.0 # Initial temperature
    cooling = 0.99
    
    current_p = best_p.copy()
    current_cost = best_cost
    
    while time.time() - start_time < time_limit - 2.0:
        # Perturb current solution
        num_swaps = random.randint(5, 15)
        for _ in range(num_swaps):
            i, j = random.sample(range(n), 2)
            current_p[i], current_p[j] = current_p[j], current_p[i]
        
        # Local search on perturbed solution
        pert_p, pert_cost = local_search(current_p, current_cost)
        
        # Acceptance criterion
        delta = pert_cost - best_cost
        if delta < 0:
            # Better
            best_p = pert_p.copy()
            best_cost = pert_cost
            current_p = best_p.copy()
            current_cost = best_cost
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < np.exp(-delta / T):
                current_p = pert_p.copy()
                current_cost = pert_cost
            else:
                # Reject, keep current
                pass
        
        T *= cooling
        
        # If T is too low, restart or keep going?
        # ILS usually keeps the best and restarts perturbation from best.
        # Here we are doing a chain. Let's restart from best if we get stuck.
        # Actually, standard ILS:
        # 1. Local Search
        # 2. Perturb
        # 3. Local Search
        # 4. Accept if better or with prob SA.
        
        # Let's simplify: Always restart perturbation from best_p to avoid drifting too far?
        # Or keep the chain? Chain is better for SA.
        
    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
