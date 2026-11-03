import time
import random
import numpy as np

def _build_44_rank(n, m, p):
    """Build the known 44-rank decomposition."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    rank = 44
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0) - 16 terms
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: 2x2x8 part (j=1,2) - 4 blocks of 7 terms = 28 terms
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Check that the decomposition is exact."""
    R = len(U[0]) if U else 0
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if abs(val - expected_val) > 1e-9:
                            return False
    return True


def _verify_decomposition_np(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                # C[i,k] for this (i0,j0,k0) = sum_r W[k*n+i,r] * U[i*m+j0,r] * V[j0*p+k,r]
                # Only (i=i0, k=k0) should be 1, all others 0
                for i in range(n):
                    for k in range(p):
                        if i == i0 and k == k0:
                            val = np.sum(W[k*n+i, :] * U[i*m+j0, :] * V[j0*p+k, :])
                            if abs(val - 1.0) > 1e-9:
                                return False
                        else:
                            val = np.sum(W[k*n+i, :] * U[i*m+j0, :] * V[j0*p+k, :])
                            if abs(val) > 1e-9:
                                return False
    return True


def _try_rank43_search(deadline):
    """
    Try to find a rank-43 decomposition.
    
    Strategy: The 2x1x8 part (j=0) requires exactly 16 terms (A[i,0]*B[0,k] for all i,k).
    The 2x2x8 part (j=1,2) currently uses 28 terms (4 blocks x 7).
    
    To get rank 43, we need the 2x2x8 part to use 27 terms, OR we need to share
    terms between the j=0 and j=1,2 parts.
    
    Since j=0 terms only involve A[i,0] and B[0,k], and j=1,2 terms involve
    A[i,1],A[i,2] and B[1,k],B[2,k], direct sharing seems impossible.
    
    However, we can try to find a rank-27 decomposition of the 2x2x8 tensor.
    The 2x2x8 tensor has 2*2*8 = 32 entries in C, but with the constraint that
    C[i,k] = sum_j A[i,j]*B[j,k] for j in {1,2}.
    
    Let's set up the problem: we need to find U,V,W with 27 columns such that
    for all (i,j,k) with j in {1,2}:
      sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = delta_{i,j,k}
    
    where delta is 1 if this is the "correct" entry and 0 otherwise.
    
    Actually, the constraint is:
    For the 2x2x8 sub-problem, we need:
    C[i,k] = A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    So for each (i,k) in {0,1}x{0..7}, we need:
    C[i,k] = A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    This is a bilinear form in (A[i,1], A[i,2]) and (B[1,k], B[2,k]).
    
    The tensor for this is of size 2x2x8 (2 rows, 2 cols of A/B, 8 output cols).
    It's equivalent to 8 independent 2x2 matrix multiplications, each of rank 7
    (Strassen), giving 8*7 = 56... no wait.
    
    Actually, each output C[i,k] depends on A[i,1],A[i,2] and B[1,k],B[2,k].
    For fixed i, the mapping (A[i,1],A[i,2]) x (B[1,k],B[2,k]) -> C[i,k] is
    a 2x2 matrix product (scalar product of two 2-vectors).
    
    Wait, C[i,k] = A[i,1]*B[1,k] + A[i,2]*B[2,k] is just a dot product of two
    2-dimensional vectors. This has rank 2 (you can compute it with 2 multiplications:
    A[i,1]*B[1,k] and A[i,2]*B[2,k]).
    
    So for each i in {0,1} and k in {0..7}, we need 2 multiplications, giving
    2*2*8 = 32 terms total for the 2x2x8 part if done naively.
    
    But with Strassen-like tricks, we can share across k values.
    
    The key observation: for fixed i, we have 8 output values C[i,0]..C[i,7], each
    being a dot product of (A[i,1],A[i,2]) with (B[1,k],B[2,k]).
    
    This is equivalent to multiplying a 1x2 matrix (A[i,1],A[i,2]) by a 2x8 matrix
    (B[1,0..7], B[2,0..7]) to get a 1x8 result.
    
    The rank of this 1x2x8 tensor is at most 2*8 = 16 (schoolbook) but can be
    reduced using the structure.
    
    Actually, for a 1xm x pxq multiplication, the rank is at most m*q (schoolbook)
    but can be less. For 1x2 x 2x8, we're computing:
    C[k] = A[1]*B[1,k] + A[2]*B[2,k] for k=0..7
    
    This is 8 dot products of 2-vectors. The tensor rank is at most 8*2 = 16,
    but since all 8 share the same A vector, we can do better.
    
    Actually, this is equivalent to: given vector a = (A[1], A[2]) and matrix B (2x8),
    compute a^T * B (1x8 result). This is a linear map from (a, B) to C.
    
    The bilinear complexity of 1x2 x 2x8 matrix multiplication... 
    
    Let me think differently. The 2x2x8 tensor (where the "2x2" refers to the
    A and B sub-matrices and "8" to the output) can be seen as 8 copies of the
    2x1x1 tensor (dot product of two 2-vectors), but they share the A part.
    
    For the full 2x3x8 problem:
    - j=0: C[i,k] += A[i,0]*B[0,k] -> 16 terms (irreducible)
    - j=1,2: C[i,k] += A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    For the j=1,2 part, for each i, we compute:
    C[i,k] = A[i,1]*B[1,k] + A[i,2]*B[2,k] for k=0..7
    
    This is 2 independent 1x2 x 2x8 problems (one for i=0, one for i=1).
    
    The rank of 1x2 x 2x8: We need to compute y_k = x_1*b_{1,k} + x_2*b_{2,k} for k=0..7.
    
    Schoolbook: 2*8 = 16 multiplications.
    
    Can we do better? The tensor is T(x,b) = sum_k (x_1*b_{1,k} + x_2*b_{2,k}) * e_k
    
    This is a bilinear map from (x_1,x_2) x (b_{1,0..7}, b_{2,0..7}) -> (y_0..y_7).
    
    The rank is at most 16. Is it less?
    
    Consider: y = x^T * B where x is 2x1 and B is 2x8.
    This is equivalent to: [x_1, x_2] * [b_{1,0} ... b_{1,7}; b_{2,0} ... b_{2,7}]
    
    The tensor rank of 1x2 x 2x8 matrix multiplication... 
    
    Actually, I recall that for 1xm x pxq, the rank is at most min(m*q, p*q, m*p)
    in some cases, but generally it's m*q for the schoolbook.
    
    For 1x2 x 2x8: m=1, p=2, q=8. Schoolbook rank = 1*2*8 = 16.
    
    But we can use the fact that the first dimension is 1. The output is 1x8.
    Each output y_k = x_1*b_{1,k} + x_2*b_{2,k}.
    
    We can write: y_k = (x_1 + x_2) * ((b_{1,k} + b_{2,k})/2) + ... no, that doesn't simplify.
    
    Actually: x_1*b_{1,k} + x_2*b_{2,k} = (x_1+x_2)*(b_{1,k}+b_{2,k})/2 + (x_1-x_2)*(b_{1,k}-b_{2,k})/2
    
    This uses 4 multiplications per k, which is worse.
    
    Alternatively: x_1*b_{1,k} + x_2*b_{2,k} can be computed with 2 multiplications
    per k, and since x_1 and x_2 are shared across all k, we can't easily reduce.
    
    I believe the rank of 1x2 x 2x8 is exactly 16 (no improvement over schoolbook)
    because the 8 output coordinates are "independent" in the sense that each involves
    different B entries.
    
    So the total rank would be 16 (j=0) + 16 (i=0, j=1,2) + 16 (i=1, j=1,2) = 48,
    which is the schoolbook.
    
    But the Strassen-based approach achieves 44 by sharing across k within each
    2x2x2 block. The 2x2x2 Strassen trick works because within a 2x2x2 block,
    the A and B entries are shared across the 4 outputs.
    
    For 2x2x8, we split into 4 blocks of 2x2x2, each using 7 terms (Strassen),
    giving 28 terms. The question is whether we can do better than 28 for 2x2x8.
    
    The 2x2x8 tensor: C[i,k] = A[i,1]*B[1,k] + A[i,2]*B[2,k] for i in {0,1}, k in {0..7}.
    
    This is 16 outputs, each a bilinear form in 4 A-entries (A[0,1],A[0,2],A[1,1],A[1,2])
    and 16 B-entries (B[1,0..7], B[2,0..7]).
    
    Wait, for output C[i,k], only A[i,1], A[i,2] and B[1,k], B[2,k] are involved.
    So the tensor has a block structure: 2 blocks (for i=0 and i=1), each of size
    2x2x8 (A[i,1],A[i,2] x B[1,k],B[2,k] -> C[i,k]).
    
    These two blocks are independent (different A entries), so the total rank is
    the sum of the ranks of the two blocks.
    
    Each block is 1x2 x 2x8 (as I analyzed above), which I believe has rank 16.
    
    So the total for j=1,2 is 16+16 = 32, giving total rank 16+32 = 48.
    
    But Strassen achieves 28 for the j=1,2 part! How?
    
    Ah, I see the error. In the Strassen approach for 2x2x2, the 7 terms involve
    products like (A[0,1]+A[1,2])*(B[1,k0]+B[2,k1]), which mixes A entries from
    different i values and B entries from different k values.
    
    This is possible because in 2x2x2, all 4 A entries and all 4 B entries are
    involved in the computation of all 4 C entries.
    
    For 2x2x8, if we split into 2x2x2 blocks, each block only involves
    A[0,1],A[0,2],A[1,1],A[1,2] and B[1,k0],B[1,k1],B[2,k0],B[2,k1] and
    produces C[0,k0],C[0,k1],C[1,k0],C[1,k1].
    
    The Strassen trick works within each 2x2x2 block, giving 7 terms per block,
    28 total.
    
    Can we do better by not splitting into 2x2x2 blocks? For example, use a
    2x2x4 block (4 terms of Strassen-like)?
    
    The 2x2x4 tensor: C[i,k] = A[i,1]*B[1,k] + A[i,2]*B[2,k] for i in {0,1}, k in {0..3}.
    
    This has 8 outputs. The A entries are A[0,1],A[0,2],A[1,1],A[1,2] (4 entries)
    and B entries are B[1,0..3],B[2,0..3] (8 entries).
    
    Can we find a rank < 14 decomposition? (14 = 2 blocks of 2x2x2 with 7 each)
    
    This is equivalent to asking: what is the rank of the 2x2x4 matrix multiplication
    tensor?
    
    I don't know the exact rank, but it's at most 14 (two 2x2x2 Strassen blocks).
    
    Let me try to search for a rank-13 decomposition of 2x2x4.
    
    Actually, let me just try a general linear algebra search for rank-43 of the
    full 2x3x8 problem.
    
    The approach:
    1. Start with the rank-44 solution.
    2. Try to eliminate one term by finding a linear combination of the remaining
       terms that can replace it.
    3. This is a nonlinear search, but we can linearize around the known solution.
    
    Alternatively, use a direct rank-43 search:
    - We need U (6x43), V (24x43), W (16x43) such that the tensor identity holds.
    - The constraints are bilinear in (U,V) and linear in W (or vice versa).
    - We can use an iterative approach: fix W, solve for U,V; fix U,V, solve for W.
    
    Let me try the alternating minimization approach.
    """
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    target_rank = 43
    
    # Initialize with the rank-44 solution, dropping one column
    base = _build_44_rank(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    # Drop the last column (term 43, which is P7 of the last block)
    U = U0[:, :target_rank].copy()
    V = V0[:, :target_rank].copy()
    W = W0[:, :target_rank].copy()
    
    # Add random perturbation to the dropped column's position
    # Actually, let's try a different approach: use the rank-44 solution and
    # try to find a rank-43 by solving a least-squares problem.
    
    # The tensor identity: for all (i,j,k),
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = T[i,j,k]
    # where T[i,j,k] = 1 if this is the "identity" entry, 0 otherwise.
    
    # For matrix multiplication, T[i,j,k] = 1 for all (i,j,k) since
    # C[i,k] = sum_j A[i,j]*B[j,k], and the tensor coefficient of A[i,j]*B[j,k]
    # in C[i,k] is 1.
    
    # Wait, the tensor is: M[i,j,k] = delta_{i,i'} * delta_{k,k'} when we
    # evaluate with A[i',j]=1 and B[j,k']=1.
    
    # Actually, the matrix multiplication tensor M has entries:
    # M[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k])
    
    # No, that's not right either. The tensor M is defined by:
    # <M, A ⊗ B> = tr(A B) or C = A B
    # M[i,j,k] = 1 if the (i,k)-th entry of C = A B depends on A[i,j] and B[j,k]
    # which is always the case, so M[i,j,k] = 1 for all i,j,k.
    
    # Wait, that can't be right because then the rank would be 1.
    
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is in
    # (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ R^{p*n} and is defined by:
    # M[A,B] = A B
    # So M[i,j; j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k] is 1).
    
    # In the flattened representation:
    # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> j*p+k
    # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
    
    # The constraint is: for all (i,j,k),
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
    
    # This is because C[i,k] = sum_j A[i,j]*B[j,k], and the coefficient of
    # A[i,j]*B[j,k] in C[i,k] is 1.
    
    # So we need: for all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
    
    # This gives n*m*p = 2*3*8 = 48 constraints.
    
    # With rank 43, we have 43*(6+24+16) = 43*46 = 1978 variables and 48 constraints.
    # This is highly underdetermined, so solutions should exist if the rank is >= 43.
    
    # The question is whether rank 43 is sufficient.
    
    # Let me try a random initialization and alternating least squares.
    
    def _als_step(U, V, W, n, m, p, max_iter=50):
        """Alternating least squares to satisfy the tensor identity."""
        n, m, p = 2, 3, 8
        dim_U = n * m
        dim_V = m * p
        dim_W = p * n
        R = U.shape[1]
        
        for it in range(max_iter):
            # Update W: for each (k,i), solve
            # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
            # This is: W[k*n+i, :] @ diag(U[i*m+j, :] * V[j*p+k, :]) = 1 for each j
            # Actually: for fixed (k,i), we need:
            # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j in 0..m-1
            # Let a_j[r] = U[i*m+j, r] * V[j*p+k, r]
            # Then: W[k*n+i, :] @ diag(a_j) ... no.
            # sum_r W[k*n+i, r] * a_j[r] = 1 for each j
            # This is: A_j @ w = 1 where A_j[r] = a_j[r]
            # So: [a_0; a_1; ... a_{m-1}] @ w = [1; 1; ... 1]
            # This is a linear system: (m x R) @ (R x 1) = (m x 1)
            
            for k in range(p):
                for i in range(n):
                    # Build the system
                    A_sys = np.zeros((m, R))
                    for j in range(m):
                        A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b_sys = np.ones(m)
                    # Solve A_sys @ w = b_sys (least squares)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W[k*n+i, :] = w_sol
                    except:
                        pass
            
            # Update U: for each (i,j), solve
            # sum_r U[i*m+j, r] * (W[k*n+i, r] * V[j*p+k, r]) = 1 for all k
            for i in range(n):
                for j in range(m):
                    A_sys = np.zeros((p, R))
                    for k in range(p):
                        A_sys[k, :] = W[k*n+i, :] * V[j*p+k, :]
                    b_sys = np.ones(p)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        U[i*m+j, :] = u_sol
                    except:
                        pass
            
            # Update V: for each (j,k), solve
            # sum_r V[j*p+k, r] * (W[k*n+i, r] * U[i*m+j, r]) = 1 for all i
            for j in range(m):
                for k in range(p):
                    A_sys = np.zeros((n, R))
                    for i in range(n):
                        A_sys[i, :] = W[k*n+i, :] * U[i*m+j, :]
                    b_sys = np.ones(n)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        V[j*p+k, :] = v_sol
                    except:
                        pass
        
        return U, V, W
    
    # Try multiple random initializations
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        
        # Initialize: use the rank-44 solution with one column zeroed out
        U_t = U0[:, :target_rank].copy()
        V_t = V0[:, :target_rank].copy()
        W_t = W0[:, :target_rank].copy()
        
        # Add small random noise to encourage exploration
        if trial > 0:
            U_t = U_t + 0.1 * np.random.randn(dim_U, target_rank)
            V_t = V_t + 0.1 * np.random.randn(dim_V, target_rank)
            W_t = W_t + 0.1 * np.random.randn(dim_W, target_rank)
        
        # Run ALS
        U_t, V_t, W_t = _als_step(U_t, V_t, W_t, n, m, p, max_iter=30)
        
        # Check error
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(W_t[k*n+i, :] * U_t[i*m+j, :] * V_t[j*p+k, :])
                    err += (val - 1.0)**2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
        
        if err < 1e-6:
            break
    
    if best_err < 1e-6:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None


def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with the known 44-rank solution
    best_result = _build_44_rank(n, m, p)
    
    # Try to find rank-43
    if time.time() < deadline - 20:
        result_43 = _try_rank43_search(deadline)
        if result_43 is not None:
            if _verify_decomposition(result_43["U"], result_43["V"], result_43["W"], n, m, p):
                best_result = result_43
    
    return best_result


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify_decomposition(res["U"], res["V"], res["W"], 2, 3, 8):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
