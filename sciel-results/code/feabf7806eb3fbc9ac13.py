"""
Tensor decomposition for 2x2 * 2x10 matrix multiplication.
Strategy: Build rank-35 block-diagonal Strassen, then use algebraic
linear-dependence detection (linear-algebraic elimination via tensor
coefficient matching) to find genuine rank reductions below 35.

The key insight: the multiplication tensor T in V(A) x V(B) x V(C) where
A is 2x2, B is 2x10, C is 2x10.  The block-diagonal Strassen gives rank 35.
We check whether the 35 rank-1 terms are linearly dependent over the tensor
space, i.e. whether some subset of 34 (or fewer) still spans the tensor.

We use the fact that the tensor is determined by its 2*2*10 = 40 trilinear
coordinates.  A rank-r decomposition gives r parameters per side (r U-cols,
r V-cols, r W-cols) but we need only to verify that a subset of terms
still reproduces the tensor.

Better approach: set up the linear system.  For each candidate subset S of
{0..34}, check if the tensor is in the span of the rank-1 tensors
U_s V_s W_s for s in S.  This is a linear problem: the 40-dim tensor must
be a linear combination of the |S| rank-1 tensors.  We check rank of the
|S| x 40 matrix formed by vectorizing each U_s V_s W_s.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return rank-7 Strassen factor matrices for 2x2 * 2x2.
    U: (4,7), V: (4,7), W: (4,7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    C indices (k*n+i with n=2): k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    """
    # P1..P7
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5       -> k=0, i=1 -> index 1
    # C10 = P2 + P4       -> k=1, i=0 -> index 2
    # C11 = P1 - P2 + P3 + P6  -> k=1, i=1 -> index 3
    
    strassen_A = np.array([
        [1, 0, 0, 1],   # P1: (a+d)(e+h)
        [0, 0, 1, 1],   # P2: (c+d)e
        [1, 0, 0, 0],   # P3: a(f-h)
        [0, 0, 0, 1],   # P4: d(g-e)
        [1, 1, 0, 0],   # P5: (a+b)h
        [-1, 0, 1, 0],  # P6: (c-a)(e+f)
        [0, 1, 0, -1],  # P7: (b-d)(g+h)
    ], dtype=float)  # shape (7,4)
    
    strassen_B = np.array([
        [1, 0, 0, 1],   # P1: (e+h)
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: (f-h)
        [-1, 0, 1, 0],  # P4: (g-e)
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: (e+f)
        [0, 0, 1, 1],   # P7: (g+h)
    ], dtype=float)  # shape (7,4)
    
    # W: contribution of each P_r to each C entry
    # C local indices: 0=C00, 1=C01, 2=C10, 3=C11
    # But in our W[k*2+i] convention: k=0,i=0 -> 0 (C00); k=0,i=1 -> 1 (C01); k=1,i=0 -> 2 (C10); k=1,i=1 -> 3 (C11)
    w_contrib = np.zeros((7, 4), dtype=float)
    # P1 -> C00 (+1), C11 (+1)
    w_contrib[0, 0] = 1
    w_contrib[0, 3] = 1
    # P2 -> C10 (+1), C11 (-1)
    w_contrib[1, 2] = 1
    w_contrib[1, 3] = -1
    # P3 -> C01 (+1), C11 (+1)
    w_contrib[2, 1] = 1
    w_contrib[2, 3] = 1
    # P4 -> C00 (+1), C10 (+1)
    w_contrib[3, 0] = 1
    w_contrib[3, 2] = 1
    # P5 -> C00 (-1), C01 (+1)
    w_contrib[4, 0] = -1
    w_contrib[4, 1] = 1
    # P6 -> C11 (+1)
    w_contrib[5, 3] = 1
    # P7 -> C00 (+1)
    w_contrib[6, 0] = 1
    
    # Transpose to get (4,7)
    U = strassen_A.T  # (4,7)
    V = strassen_B.T  # (4,7)
    W = w_contrib.T   # (4,7)
    
    return U, V, W


def _build_block_strassen(n, m, p):
    """
    Build block-diagonal Strassen for n=2, m=2, p=10 (5 blocks of 2x2).
    Returns U (n*m, 35), V (m*p, 35), W (p*n, 35).
    """
    num_blocks = p // 2  # 5
    block_rank = 7
    total_rank = num_blocks * block_rank  # 35
    
    Ub, Vb, Wb = _strassen_2x2()
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    for b in range(num_blocks):
        offset = b * block_rank
        for r in range(block_rank):
            col = offset + r
            # U: same across blocks
            U[:, col] = Ub[:, r]
            
            # V: block b uses B columns 2b, 2b+1
            # V[j*p + k, col] = Vb[j*2 + k_local, r] where k_local = 0 or 1
            for j in range(m):
                for kl in range(2):
                    k = 2 * b + kl
                    V[j * p + k, col] = Vb[j * 2 + kl, r]
            
            # W: C[k*n + i, col] = Wb[k_local*2 + i, r] where k_local = 0 or 1
            for i in range(n):
                for kl in range(2):
                    k = 2 * b + kl
                    W[k * n + i, col] = Wb[kl * 2 + i, r]
    
    return U, V, W


def _tensor_vectorize(U, V, W, n, m, p):
    """
    For each rank-1 term r, compute the vectorized contribution to the tensor.
    Returns a matrix of shape (rank, n*m*p) where each row is the vectorized
    (i,j,k) -> U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    """
    rank = U.shape[1]
    dim = n * m * p
    vecs = np.zeros((rank, dim), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vecs[r, i * m * p + j * p + k] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    return vecs


def _check_subset_span(S, vecs, target_tensor, tol=1e-10):
    """
    Check if target_tensor is in the span of {vecs[s] : s in S}.
    Returns True if yes.
    """
    if len(S) == 0:
        return np.allclose(target_tensor, 0, atol=tol)
    M = vecs[list(S), :]  # shape (|S|, dim)
    # Solve M^T x = target_tensor in least squares
    # Check residual
    try:
        # Use np.linalg.lstsq
        sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, target_tensor, rcond=tol)
        residual = np.linalg.norm(target_tensor - M.T @ sol)
        return residual < tol * (1 + np.linalg.norm(target_tensor))
    except:
        return False


def _find_minimal_subset(vecs, target_tensor, n_terms, deadline):
    """
    Find a minimal subset of terms that spans the target tensor.
    Use greedy: start with all terms, try removing one at a time.
    Then try pairs, etc.
    """
    all_idx = list(range(n_terms))
    current = set(all_idx)
    
    # Greedy removal: try removing each term
    changed = True
    while changed:
        changed = False
        for r in list(current):
            if time.time() > deadline - 30:
                return current
            trial = current - {r}
            if _check_subset_span(list(trial), vecs, target_tensor):
                current = trial
                changed = True
    
    # Try removing pairs from the reduced set
    if len(current) > 2:
        idx_list = list(current)
        for i in range(len(idx_list)):
            if time.time() > deadline - 20:
                return current
            for j in range(i + 1, len(idx_list)):
                if time.time() > deadline - 20:
                    return current
                trial = current - {idx_list[i], idx_list[j]}
                if _check_subset_span(list(trial), vecs, target_tensor):
                    current = trial
    
    return current


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    deadline = time.time() + 110  # safety margin before 120s
    n, m, p = DIMENSIONS
    
    # Build rank-35 block-diagonal Strassen
    U, V, W = _build_block_strassen(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook rank 40
        rank = n * m * p
        Uf = np.zeros((n * m, rank))
        Vf = np.zeros((m * p, rank))
        Wf = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, col] = 1
                    Vf[j * p + k, col] = 1
                    Wf[k * n + i, col] = 1
                    col += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
    
    # Compute vectorized terms
    vecs = _tensor_vectorize(U, V, W, n, m, p)
    
    # Target tensor: T[i,j,k] = 1 if (i,j,k) is a valid (i,j,k) coordinate
    # Actually the multiplication tensor T has T[i,j,k] = delta (it's 1 for all valid coords)
    # Wait, the tensor is: T(A,B) = A*B, so the trilinear form is
    # sum_{i,j,k} C[k*n+i] A[i*m+j] B[j*p+k] = sum_{i,j,k} (A*B)[k*n+i] A[i*m+j] B[j*p+k]
    # The "tensor" T is defined by T[i,j,k] such that the trilinear form equals the matrix product.
    # Actually, the tensor is the 3-way array with entries T[i, j, k] where
    # the contraction gives C = A*B.  Specifically:
    # (A*B)[k,i] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the identity mapping in tensor form).
    
    target = np.ones(n * m * p, dtype=float)
    # But wait: the trilinear form is sum_{i,j,k} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i*m+j] B[j*p+k] C[k*n+i]
    # and this should equal sum_{i,j,k} A[i*m+j] B[j*p+k] (A*B)[k*n+i]
    # = sum_{i,j,k} A[i*m+j] B[j*p+k] sum_l A[i,l] B[l,k]
    # The tensor T[i,j,k] is such that the coefficient of A[i*m+j] B[j*p+k] C[k*n+i] in the product is...
    # Actually it's simpler: T is the tensor with T[i,j,k] = delta_{...} 
    # The standard matrix multiplication tensor has T[i,j,k] = 1 iff the indices are consistent.
    # For 2x2 * 2x10: T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,9}.
    # Because (A*B)[k,i] = sum_j A[i,j] B[j,k], so the tensor entry T[i,j,k] = 1.
    
    # So target[i*m*p + j*p + k] = 1 for all valid i,j,k
    target = np.ones(n * m * p, dtype=float)
    
    # Find minimal spanning subset
    minimal = _find_minimal_subset(vecs, target, U.shape[1], deadline)
    
    if len(minimal) < U.shape[1]:
        keep = sorted(minimal)
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Return the full rank-35
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
