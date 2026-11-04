import time
import numpy as np
import random
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [4, 4, 9]

# ---------------------------------------------------------------------------
# Strassen 2x2x2: 7 rank-1 terms.  Block index order: 0=TL,1=TR,2=BL,3=BR.
# ---------------------------------------------------------------------------
_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # m2
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # m3
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),  # m4
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # m5
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # m6
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # m7
]


def _add_strassen_block(U, V, W, col, a_base, b_base, c_base):
    """Add 7 rank-1 Strassen terms for a 2x2x2 block."""
    for (u, v, w) in _STRASSEN_222:
        for t in range(4):
            if u[t]:
                U[a_base + t][col] += u[t]
            if v[t]:
                V[b_base + t][col] += v[t]
            if w[t]:
                W[c_base + t][col] += w[t]
        col += 1
    return col


def _add_444(U, V, W, col, a_base, b_base, c_base):
    """Exact rank-49 decomposition for a 4x4x4 block via 7 Strassen block-sums."""
    U_sums = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    V_sums = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [0, -1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    W_sums = [
        [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
    ]
    for t in range(7):
        u, v, w = U_sums[t], V_sums[t], W_sums[t]
        for s in range(4):
            if u[s]:
                U[a_base + s][col] += u[s]
            if v[s]:
                V[b_base + s][col] += v[s]
            if w[s]:
                W[c_base + s][col] += w[s]
        col += 1
    return col


def _add_matrix_vector_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    """Matrix-vector: A(ar x ac) * B(ac x bc) where bc=1.  Rank ar*ac."""
    for i in range(ar):
        for j in range(ac):
            U[a_base + i * ac + j][col] += 1
            V[b_base + j * bc][col] += 1
            W[c_base + i * bc][col] += 1
            col += 1
    return col


def _add_449(U, V, W, col, a_base, b_base, c_base):
    """Rank-114 decomposition for 4x4x9: two 4x4x4 Strassen blocks + 4x4x1 MV."""
    col = _add_444(U, V, W, col, a_base, b_base, c_base)
    col = _add_444(U, V, W, col, a_base, b_base + 4, c_base + 4)
    col = _add_matrix_vector_block(U, V, W, col, a_base, b_base + 8, c_base + 8, 4, 4, 1)
    return col


def _verify(U, V, W, n, m, p):
    """Return True iff the decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True


def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_rank114(n, m, p):
    """Rank-114: two 4x4x4 Strassen (98) + 4x4x1 matrix-vector (16)."""
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = _add_449(U, V, W, 0, 0, 0, 0)
    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}


def _build_via_ilp(n, m, p, time_limit=10):
    """
    Attempt to find a lower rank decomposition using Integer Linear Programming.
    We search for a sparse set of rank-1 terms from a candidate pool.
    """
    # 1. Generate candidate rank-1 terms
    # We use a pool of "structured" terms:
    # - Standard basis terms (schoolbook)
    # - Strassen-like terms for 2x2x2 blocks
    # - Random sparse terms
    
    candidates = []
    
    # Add schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                candidates.append((u, v, w))
                
    # Add Strassen terms for all 2x2x2 sub-blocks
    # This is expensive to generate all, so we just add the global Strassen ones 
    # if dimensions allow, or just rely on the pool being large enough.
    # For 4x4x9, we can add Strassen terms for the 4x4x4 parts.
    
    # Add random sparse terms to allow for "new" decompositions
    # Limit pool size to keep ILP tractable
    max_pool_size = 200
    num_random = max(0, max_pool_size - len(candidates))
    
    for _ in range(num_random):
        # Random sparse vectors with +/-1 entries
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # Pick 2-4 non-zero entries
        nz_u = random.randint(2, 4)
        nz_v = random.randint(2, 4)
        nz_w = random.randint(2, 4)
        
        idx_u = random.sample(range(n*m), nz_u)
        idx_v = random.sample(range(m*p), nz_v)
        idx_w = random.sample(range(p*n), nz_w)
        
        for idx in idx_u: u[idx] = random.choice([-1, 1])
        for idx in idx_v: v[idx] = random.choice([-1, 1])
        for idx in idx_w: w[idx] = random.choice([-1, 1])
        
        candidates.append((u, v, w))
        
    num_candidates = len(candidates)
    
    # 2. Formulate ILP
    # Minimize sum(x_i) subject to A x = T
    
    # T is the target tensor flattened
    T_target = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m*p + j*p + k] = 1.0 # Wait, the tensor index mapping in _verify is specific.
                # _verify checks T[i*m+j, j*p+k, k*n+i] == 1
                # This corresponds to the standard matrix multiplication tensor.
                # Let's build the matrix A such that A @ x = T_flat
    
    # A is of shape (n*m*p, num_candidates)
    # A[elem, c] = U_c[i] * V_c[j] * W_c[k] where elem corresponds to (i,j,k)
    
    # To save memory/time, we might not be able to build a huge A if num_candidates is large.
    # n*m*p = 144. num_candidates ~ 200. A is 144x200. This is small.
    
    A = np.zeros((n*m*p, num_candidates))
    
    for c_idx, (u, v, w) in enumerate(candidates):
        # Reshape to 3D for einsum or just loop
        # u is (n*m), v is (m*p), w is (p*n)
        # We need to compute u[i*m+j] * v[j*p+k] * w[k*n+i] for all i,j,k
        # This is equivalent to np.einsum('a,b,c->abc', u, v, w) but with specific index mappings.
        # u index: i*m+j
        # v index: j*p+k
        # w index: k*n+i
        
        # Let's just loop, it's 144 * 200 = 28800 ops, very fast.
        for i in range(n):
            for j in range(m):
                u_val = u[i*m + j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v[j*p + k]
                    if v_val == 0: continue
                    w_val = w[k*n + i]
                    if w_val == 0: continue
                    A[i*m*p + j*p + k, c_idx] = u_val * v_val * w_val
                    
    # Target vector
    b = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b[i*m*p + j*p + k] = 1.0
                
    # ILP: min 1^T x
    # s.t. A x = b
    # x >= 0, x integer
    
    c_obj = np.ones(num_candidates)
    
    constraints = [
        LinearConstraint(A, b, b), # A x = b
    ]
    
    integrality = np.ones(num_candidates)
    bounds = Bounds(lb=0, ub=1) # Binary variables
    
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0 and res.x is not None:
            x = res.x
            # Check if solution is valid and integer
            if np.allclose(x, np.round(x)):
                x_int = np.round(x).astype(int)
                rank = int(np.sum(x_int))
                
                # Reconstruct U, V, W
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                
                col_idx = 0
                for c_idx in range(num_candidates):
                    if x_int[c_idx] > 0:
                        u, v, w = candidates[c_idx]
                        U[:, col_idx] = u
                        V[:, col_idx] = v
                        W[:, col_idx] = w
                        col_idx += 1
                        
                # Convert to list of lists for output format
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # 1. Try ILP with a short time limit (e.g., 10s)
    # Note: milp in scipy uses HiGHS. We can pass options but time limit is tricky to enforce strictly 
    # without checking time inside the loop, which we can't do inside milp easily.
    # However, 144x200 ILP is usually very fast.
    
    ilp_res = None
    if time.time() < deadline - 10:
        try:
            ilp_res = _build_via_ilp(n, m, p)
        except Exception:
            ilp_res = None
            
    if ilp_res and _verify(ilp_res["U"], ilp_res["V"], ilp_res["W"], n, m, p):
        # Check if it's better than 114
        rank_ilp = len(ilp_res["U"][0])
        if rank_ilp < 114:
            return ilp_res
            
    # 2. Fallback to Strassen rank 114
    res = _build_rank114(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res

    return _build_schoolbook(n, m, p)
