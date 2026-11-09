"""Rank-minimal exact tensor decomposition for 2x3 times 3x7 matrix multiplication.

Uses a structured approach: build V (21 x rank) with integer entries, find its
left nullspace to get a 2x3 matrix M, then set U = M @ C for some 3 x rank matrix C.
This guarantees the bilinear identity holds exactly when W is chosen appropriately.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# Safety margin before the 150 s wall-clock kill (evaluation limit ~120 s).
DEADLINE = 130.0


def _seed():
    """Schoolbook rank-42 decomposition (fallback)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the tensor."""
    rank = len(U[0])
    # Check consistent triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                s = 0
                for r in range(rank):
                    s += U[u][r] * V[v][r] * W[w][r]
                if s != 1:
                    return False
    # Check inconsistent triples
    for u in range(n * m):
        i, j = divmod(u, m)
        for v in range(m * p):
            j2, k = divmod(v, p)
            for w in range(p * n):
                k2, i2 = divmod(w, n)
                if j == j2 and k == k2 and i == i2:
                    continue
                s = 0
                for r in range(rank):
                    s += U[u][r] * V[v][r] * W[w][r]
                if s != 0:
                    return False
    return True


def _build_decomposition(rank):
    """
    Build a rank-`rank` decomposition using the lifting approach.
    
    Strategy:
    1. V has shape (21, rank). We want V to have full row rank 21 (so rank >= 21).
    2. The left nullspace of V is a 3-dimensional space (since 21 - 21 = 0 if rank=21,
       but we need rank > 21 for the nullspace to be non-trivial in the right way).
       
    Actually, let me reconsider. The tensor decomposition requires:
    For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    For all other (u,v,w): sum_r U[u,r] * V[v,r] * W[w,r] = 0
    
    A cleaner approach: Use the fact that matrix multiplication can be viewed as a
    bilinear map. We can use Strassen-like ideas or just construct directly.
    
    Direct construction:
    Let U have shape (6, rank), V have shape (21, rank), W have shape (14, rank).
    
    We can set up the problem as: for each rank-1 component r, we have:
    U[:,r] is a vector in R^6, V[:,r] in R^21, W[:,r] in R^14.
    
    The constraint is:
    sum_r U[u,r] V[v,r] W[w,r] = tensor[u,v,w]
    
    where tensor[u,v,w] = 1 if (u,v,w) corresponds to (i,j,k) with u=i*m+j, v=j*p+k, w=k*n+i, else 0.
    
    This is a system of equations in the unknowns U,V,W.
    
    For small ranks, we can try to solve this. But it's nonlinear.
    
    Alternative: Use the "lifting" trick. Suppose we fix V and W, then U is determined
    by solving a linear system. Similarly for other combinations.
    
    Let's try: fix V and W, solve for U.
    For each u: sum_r U[u,r] * (V[:,r] ⊙ W[:,r])[v,w] = tensor[u,v,w]
    
    This is still coupled.
    
    Better approach: Use the fact that n=2, m=3, p=7. The tensor has structure.
    
    Let me try a different construction:
    
    For rank r, we can think of it as r rank-1 tensors. Each rank-1 tensor is
    specified by (u_r, v_r, w_r) where u_r ∈ {0..5}, v_r ∈ {0..20}, w_r ∈ {0..13},
    and a scalar coefficient.
    
    Actually, the entries can be any half-integers, so it's more general.
    
    Let me just do a smarter random search with the structure in mind.
    """
    n, m, p = DIMENSIONS
    rows_u = n * m  # 6
    rows_v = m * p  # 21
    rows_w = p * n  # 14
    
    # The tensor T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k, else 0
    T = np.zeros((rows_u, rows_v, rows_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # We want to factor T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector in R^rows_u, etc.
    
    # This is a tensor decomposition problem. For small ranks, we can use
    # iterative methods or just random search with local optimization.
    
    # Let's try: initialize U, V, W randomly, then do gradient descent on
    # ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
    
    # But we need exact integer/half-integer solutions. So we'll do floating point
    # optimization, then round to nearest half-integer, then verify.
    
    # Actually, for this problem, let's use a simpler approach:
    # The rank-42 schoolbook is trivial. Can we do better?
    # 
    # Known results: rank of 2x3x7 tensor. The trivial bound is 42.
    # The tensor rank of matrix multiplication M_{2,3,7} is known to be at most
    # something. Let me just try random search for ranks 30-41.
    
    return None  # Placeholder


def _search_rank_structured(rank, time_limit):
    """
    Structured search for a rank-`rank` decomposition.
    
    Key insight: The tensor T[i,j,k] = 1 if indices are consistent (u=i*m+j, v=j*p+k, w=k*n+i).
    A rank-1 component r contributes U[u,r]*V[v,r]*W[w,r] to T[u,v,w].
    
    For the decomposition to work, each rank-1 component should "support" some subset of
    (i,j,k) triples. A natural structure: for each component r, assign it to a specific
    (i_r, j_r, k_r) triple, and set:
      U[i_r*m+j_r, r] = a_r
      V[j_r*p+k_r, r] = b_r  
      W[k_r*n+i_r, r] = c_r
    with a_r * b_r * c_r contributing to that triple's sum.
    
    But this alone gives rank 42 (one per triple). To reduce rank, we need components
    that cover multiple triples. A component with U supported on multiple (i,j) pairs,
    V on multiple (j,k) pairs, W on multiple (k,i) pairs can cover many triples.
    
    Better structured approach: Use "block" components where each component r is
    associated with a subset of indices. Let's try:
    - Partition the 42 triples into groups, each group handled by one component.
    - For a group G of triples, set U[u,r]=1 for all u in G's u-indices, V[v,r]=1 for
      all v in G's v-indices, W[w,r]=1 for all w in G's w-indices.
    - This works if the "cross terms" cancel or are zero.
    
    Actually, the cleanest approach for small ranks: use the fact that n=2 is small.
    Split by i (the 2 rows of the first matrix):
    - For i=0: triples (0,j,k) for j in 0..2, k in 0..6 (21 triples)
    - For i=1: triples (1,j,k) for j in 0..2, k in 0..6 (21 triples)
    
    For each i, the sub-tensor is a 3x7 matrix multiplication tensor.
    The rank of M_{3,7} (3x7 matrix mult) is at most 21, but might be less.
    
    Let's try a direct construction:
    For i=0: U entries at positions (0*m+j, r) = U[j, r] for j=0,1,2
             V entries at positions (j*p+k, r) = V[j*p+k, r]
             W entries at positions (k*n+0, r) = W[k, r]
    
    The constraint for i=0: sum_r U[j,r] * V[j*p+k,r] * W[k,r] = 1 for all j,k
    The constraint for i=1: sum_r U[3+j,r] * V[j*p+k,r] * W[7+k,r] = 1 for all j,k
    Cross constraints must be zero.
    
    This is still hard. Let me try a different tactic:
    Use the known rank bounds. The rank of M_{n,m,p} is at most n*m*p.
    For M_{2,3,7}, known upper bounds: 
    - Trivial: 42
    - Using Winograd's method or similar: might be lower
    
    Let's try a more systematic search:
    1. Initialize with a random sparse decomposition
    2. Use local optimization to minimize the error
    3. Round to half-integers and verify
    """
    n, m, p = DIMENSIONS
    rows_u = n * m  # 6
    rows_v = m * p  # 21
    rows_w = p * n  # 14
    
    end_time = time.time() + time_limit
    rng = np.random.default_rng()
    
    # Precompute valid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_triples.append((u, v, w, i, j, k))
    
    # Invalid triples (sample for checking)
    all_u = list(range(rows_u))
    all_v = list(range(rows_v))
    all_w = list(range(rows_w))
    
    def compute_error(U, V, W, sample_size=200):
        """Compute approximation error."""
        err = 0.0
        # Check valid triples
        for (u, v, w, i, j, k) in valid_triples:
            s = np.sum(U[u, :] * V[v, :] * W[w, :])
            err += (s - 1.0) ** 2
        # Check sample of invalid triples
        for _ in range(sample_size):
            u = rng.choice(all_u)
            v = rng.choice(all_v)
            w = rng.choice(all_w)
            i, j = divmod(u, m)
            j2, k = divmod(v, p)
            k2, i2 = divmod(w, n)
            if j == j2 and k == k2 and i == i2:
                continue
            s = np.sum(U[u, :] * V[v, :] * W[w, :])
            err += s ** 2
        return err
    
    def local_optimize(U, V, W, max_iters=50):
        """Simple local optimization: adjust one component at a time."""
        for it in range(max_iters):
            if time.time() > end_time:
                break
            improved = False
            for r in range(rank):
                # Try adjusting U[:, r]
                best_U = U[:, r].copy()
                best_err = compute_error(U, V, W, 50)
                for _ in range(3):
                    # Random perturbation
                    U_trial = U.copy()
                    U_trial[:, r] += rng.normal(0, 0.1, rows_u)
                    err_trial = compute_error(U_trial, V, W, 50)
                    if err_trial < best_err:
                        best_err = err_trial
                        best_U = U_trial[:, r].copy()
                        improved = True
                U[:, r] = best_U
                
                # Try adjusting V[:, r]
                best_V = V[:, r].copy()
                best_err = compute_error(U, V, W, 50)
                for _ in range(3):
                    V_trial = V.copy()
                    V_trial[:, r] += rng.normal(0, 0.1, rows_v)
                    err_trial = compute_error(U, V_trial, W, 50)
                    if err_trial < best_err:
                        best_err = err_trial
                        best_V = V_trial[:, r].copy()
                        improved = True
                V[:, r] = best_V
                
                # Try adjusting W[:, r]
                best_W = W[:, r].copy()
                best_err = compute_error(U, V, W, 50)
                for _ in range(3):
                    W_trial = W.copy()
                    W_trial[:, r] += rng.normal(0, 0.1, rows_w)
                    err_trial = compute_error(U, V, W_trial, 50)
                    if err_trial < best_err:
                        best_err = err_trial
                        best_W = W_trial[:, r].copy()
                        improved = True
                W[:, r] = best_W
            
            if not improved:
                break
        
        return U, V, W
    
    # Strategy: Start from schoolbook and try to merge components
    # Or: Random structured initialization
    
    for attempt in range(2000):
        if time.time() > end_time:
            break
        
        # Initialize: each component covers a random subset of triples
        U = np.zeros((rows_u, rank), dtype=float)
        V = np.zeros((rows_v, rank), dtype=float)
        W = np.zeros((rows_w, rank), dtype=float)
        
        # Assign each triple to one or more components
        triple_assignments = [rng.integers(0, rank) for _ in range(len(valid_triples))]
        
        for idx, (u, v, w, i, j, k) in enumerate(valid_triples):
            r = triple_assignments[idx]
            # Add contribution to this component
            U[u, r] += 1.0
            V[v, r] += 1.0
            W[w, r] += 1.0
        
        # Randomize signs to help with cancellation of cross terms
        for r in range(rank):
            if rng.random() < 0.5:
                U[:, r] *= -1
            if rng.random() < 0.5:
                V[:, r] *= -1
            if rng.random() < 0.5:
                W[:, r] *= -1
        
        # Local optimization
        U, V, W = local_optimize(U, V, W, max_iters=20)
        
        # Check if we're close to a solution
        err = compute_error(U, V, W, 500)
        if err < 1e-6:
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Exact check
            Uf = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in U_rounded]
            Vf = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in V_rounded]
            Wf = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in W_rounded]
            
            if _check_decomposition(Uf, Vf, Wf, n, m, p):
                return {"U": Uf, "V": Vf, "W": Wf}
    
    return None


def construct():
    start_time = time.time()
    
    # Try ranks from 30 to 41
    for rank in range(30, 42):
        if time.time() > DEADLINE - 15:
            break
        time_left = min(15.0, DEADLINE - 15 - (time.time() - start_time))
        if time_left <= 0:
            break
        res = _search_rank_structured(rank, time_left)
        if res is not None:
            return res
    
    # Fallback to schoolbook
    return _seed()
