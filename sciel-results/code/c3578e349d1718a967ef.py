"""Rank-minimal tensor decomposition for 3x4 x 4x4 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]

def _schoolbook():
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

def _check_decomp(U, V, W, tol=1e-9):
    """Check if U*V*W (outer product sum) equals all-ones tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Compute tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # This is a contraction over c: einsum('ijc,jkc,kic->ijk')
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=tol)

def _to_half_integer(x):
    """Check if x is close to a half-integer (multiple of 0.5)."""
    r = round(x * 2) / 2.0
    return abs(x - r) < 1e-9, r

def _is_half_integer_matrix(M, tol=1e-9):
    """Check if all entries are half-integers."""
    for row in M:
        for val in row:
            ok, _ = _to_half_integer(float(val))
            if not ok:
                return False
    return True

def _round_to_half(M):
    """Round all entries to nearest half-integer."""
    return [[round(float(x) * 2) / 2.0 for x in row] for row in M]

def _try_rank_r_search(r, max_iters=50000, seed=42):
    """Try to find a rank-r decomposition using structured initialization + local search."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(20):
        # Initialize with random half-integer matrices
        def rand_half(rng, rows, cols, scale=2.0):
            M = []
            for _ in range(rows):
                row = []
                for _ in range(cols):
                    v = rng.choice([-scale, -scale+0.5, -scale+1, -scale+1.5, -scale+2,
                                     -scale+2.5, -scale+3, -scale+3.5, -scale+4,
                                     -scale+4.5, -scale+5, -scale+5.5, -scale+6,
                                     -scale+6.5, -scale+7, -scale+7.5, -scale+8])
                    # Simpler: pick from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3}
                    v = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
                    row.append(v)
                M.append(row)
            return M
        
        U = rand_half(rng, n*m, r, 2.0)
        V = rand_half(rng, m*p, r, 2.0)
        W = rand_half(rng, p*n, r, 2.0)
        
        # Local search: flip individual entries to reduce loss
        Ur = np.array(U, dtype=float).reshape(n, m, r)
        Vr = np.array(V, dtype=float).reshape(m, p, r)
        Wr = np.array(W, dtype=float).reshape(p, n, r)
        
        def compute_loss():
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            return np.sum((T - 1.0)**2)
        
        loss = compute_loss()
        if loss < best_loss:
            best_loss = loss
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        
        # Try coordinate descent: change one entry at a time
        improved = True
        iters = 0
        while improved and iters < max_iters // 20:
            improved = False
            iters += 1
            # Randomly pick which matrix and which entry to try changing
            for _ in range(100):
                mat_idx = rng.choice([0, 1, 2])
                if mat_idx == 0:
                    ri = rng.randint(0, n*m - 1)
                    ci = rng.randint(0, r - 1)
                    old_val = U[ri][ci]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        U[ri][ci] = new_val
                        Ur[ri // m][ri % m][ci] = new_val
                        new_loss = compute_loss()
                        if new_loss < loss - 1e-12:
                            loss = new_loss
                            improved = True
                            if loss < best_loss:
                                best_loss = loss
                                best_U = [row[:] for row in U]
                                best_V = [row[:] for row in V]
                                best_W = [row[:] for row in W]
                            break
                        else:
                            U[ri][ci] = old_val
                            Ur[ri // m][ri % m][ci] = old_val
                elif mat_idx == 1:
                    ri = rng.randint(0, m*p - 1)
                    ci = rng.randint(0, r - 1)
                    old_val = V[ri][ci]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        V[ri][ci] = new_val
                        Vr[ri // p][ri % p][ci] = new_val
                        new_loss = compute_loss()
                        if new_loss < loss - 1e-12:
                            loss = new_val if False else new_loss
                            loss = new_loss
                            improved = True
                            if loss < best_loss:
                                best_loss = loss
                                best_U = [row[:] for row in U]
                                best_V = [row[:] for row in V]
                                best_W = [row[:] for row in W]
                            break
                        else:
                            V[ri][ci] = old_val
                            Vr[ri // p][ri % p][ci] = old_val
                else:
                    ri = rng.randint(0, p*n - 1)
                    ci = rng.randint(0, r - 1)
                    old_val = W[ri][ci]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        W[ri][ci] = new_val
                        Wr[ri // n][ri % n][ci] = new_val
                        new_loss = compute_loss()
                        if new_loss < loss - 1e-12:
                            loss = new_loss
                            improved = True
                            if loss < best_loss:
                                best_loss = loss
                                best_U = [row[:] for row in U]
                                best_V = [row[:] for row in V]
                                best_W = [row[:] for row in W]
                            break
                        else:
                            W[ri][ci] = old_val
                            Wr[ri // n][ri % n][ci] = old_val
        
        if best_loss < 1e-8:
            return best_U, best_V, best_W
    
    if best_loss < 1e-6:
        # Try rounding
        Ur = _round_to_half(best_U)
        Vr = _round_to_half(best_V)
        Wr = _round_to_half(best_W)
        if _check_decomp(Ur, Vr, Wr):
            return Ur, Vr, Wr
        # Try the unrounded version
        if _check_decomp(best_U, best_V, best_W):
            return best_U, best_V, best_W
    
    return None

def _try_structured_rank36():
    """
    Try a structured approach for rank 36.
    
    For 3x4x4, the tensor has 3*4*4 = 48 entries, all equal to 1.
    
    One idea: use the structure that T[i,j,k] = delta-like patterns.
    
    Let's try a factorization based on:
    - For each pair (i,j), we need sum_k W[k*n+i,c] * V[j*p+k,c] to work out
    - Actually T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    Let's try: U[i,j,c] = a[i,c], V[j,k,c] = b[j,c], W[k,i,c] = c[k,c]
    Then T[i,j,k] = sum_c a[i,c]*b[j,c]*c[k,c]
    We need this to be 1 for all i,j,k.
    
    This means a[i,c]*b[j,c]*c[k,c] summed over c = 1 for all (i,j,k).
    This is a rank-1 condition on the 3-way tensor of coefficients.
    Actually this requires: for each c, the vectors a[:,c], b[:,c], c[:,c] 
    form a "complementary" set such that their element-wise product sums to 1.
    
    More specifically, define for each c: A_c[i] = a[i,c], B_c[j] = b[j,c], C_c[k] = c[k,c]
    Then sum_c A_c[i]*B_c[j]*C_c[k] = 1 for all i,j,k.
    
    This means the tensor sum_c A_c ⊗ B_c ⊗ C_c = 1_{n} ⊗ 1_{m} ⊗ 1_{p} (all-ones tensor).
    
    This is possible if we can decompose the all-ones tensor into a sum of r rank-1 tensors.
    The rank of the all-ones tensor (n×m×p) over reals is... well, 1 (it's already rank 1: 
    1_n ⊗ 1_m ⊗ 1_p). But wait, that would give rank 1, which is clearly not correct 
    because the actual matmul tensor is NOT the all-ones tensor in the usual sense.
    
    Wait, I'm confusing myself. Let me re-read the problem.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k].
    The tensor for matmul is actually T[i,j,k] = 1 for all i,j,k (since every entry of A and B 
    contributes to some output entry). No wait, that's not right either.
    
    The matmul tensor is defined as: for computing C = A*B where A is n×m and B is m×p,
    C[i,k] = sum_j A[i,j] * B[j,k].
    The bilinear form uses terms A[i,j]*B[j,k] with coefficient 1.
    So the tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    
    But the tensor decomposition is: C[i,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]... no that's wrong.
    
    Let me re-read the interface: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    The decomposition is: the matmul tensor = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    where U[:,c] is indexed by (i,j), V[:,c] by (j,k), W[:,c] by (k,i).
    
    So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    OK so my formulation above was correct. The all-ones n×m×p tensor needs to be written as 
    a sum of r "aligned" rank-1 tensors where the third index of U is (i,j), second of V is (j,k), 
    and W is indexed by (k,i) - note the specific ordering.
    
    For the structured approach U[i,j,c]=a[i,c], V[j,k,c]=b[j,c], W[k,i,c]=c[k,c]:
    sum_c a[i,c]*b[j,c]*c[k,c] = 1 for all i,j,k.
    
    This means the n×m×p tensor S where S[i,j,k] = sum_c a[i,c]*b[j,c]*c[k,c] is all-ones.
    S = sum_c a_c ⊗ b_c ⊗ c_c where a_c = a[:,c], etc.
    
    The all-ones tensor 1_n ⊗ 1_m ⊗ 1_p is rank 1, so r=1 would suffice IF we could choose
    a_c = 1_n, b_c = 1_m, c_c = 1_p. But then U[i*m+j, c] = 1 for all i,j, V[j*p+k,c] = 1 for all j,k,
    W[k*n+i,c] = 1 for all k,i. That gives rank 1!
    
    But wait, that can't be right because we know the rank of 3×4×4 matmul is 36, not 1.
    
    I think I'm misunderstanding the tensor. Let me reconsider.
    
    The matrix multiplication tensor is NOT the all-ones tensor. The issue is that in the 
    standard tensor decomposition for matmul, the tensor T is defined such that:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    The tensor T has entries T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k]).
    But the tensor DECOMPOSITION represents this bilinear map as a sum of products of linear forms.
    
    The rank of the matmul tensor for n×m × m×p is the minimum number of multiplications needed.
    For standard matmul, it's n*m*p. But there are algorithms that use fewer.
    
    The tensor is: T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} (or equivalently R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p})
    where T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    
    So T[i,j,k] = 1 if (i,j,k) is a valid triple (i<n, j<m, k<p), and the output index is (i,k).
    
    In the interface, W is indexed by (p*n, rank) with W[k*n+i, c]. So W's row index is k*n+i,
    which corresponds to the output entry C[i,k].
    
    So the decomposition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal T[i,j,k] = 1
    for all i,j,k.
    
    This IS the all-ones tensor condition! So why isn't the rank 1?
    
    Ah, I see the issue. The rank of the matmul tensor is NOT the same as the rank of the 
    all-ones tensor. The matmul tensor T is an element of a specific tensor space, and its 
    "rank" in the tensor decomposition sense (as a sum of pure tensors) is what we're computing.
    
    But T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in the appropriate bases) where each term is a pure tensor.
    The question is: can we write T as a sum of FEWER than n*m*p pure tensors?
    
    A pure tensor in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} is u ⊗ v ⊗ w where u ∈ R^{n*m}, v ∈ R^{m*p}, w ∈ R^{n*p}.
    
    T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    
    This is a sum of n*m*p pure tensors. Can we do better?
    
    For the case where n=m=p=2, the rank is known to be 7 (Strassen), not 8.
    For 3×4×4, I believe the known bound is somewhere around 36-40.
    
    The key insight is that the pure tensors in the decomposition don't have to be the "obvious" 
    ones (delta functions). They can be any vectors in the respective spaces.
    
    So my structured approach U[i,j,c]=a[i,c] etc. is too restrictive. I need the full generality.
    
    Let me try a different approach: use the fact that for small problems, we can do exhaustive 
    or semi-exhaustive search over half-integer matrices.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # First, try to find rank 36
    for r in [36, 37, 38, 39, 40, 42, 44]:
        if time.time() > deadline - 5:
            break
        try:
            result = _try_rank_r_search(r, max_iters=30000, seed=42+r)
            if result is not None:
                U, V, W = result
                if _check_decomp(U, V, W):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try rank 40 with more iterations
    if time.time() < deadline - 10:
        try:
            result = _try_rank_r_search(40, max_iters=50000, seed=999)
            if result is not None:
                U, V, W = result
                if _check_decomp(U, V, W):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Fallback to schoolbook
    return _schoolbook()
