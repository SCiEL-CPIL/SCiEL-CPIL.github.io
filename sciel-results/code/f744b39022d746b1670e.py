"""Matrix multiplication 2,4,7 — Kronecker-product decomposition exploiting 4=2×2."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _find_2x2x7_decomposition(max_rank=14, max_iters=200, tol=1e-10):
    """
    Find a low-rank decomposition of the 2×2×7 matrix-multiplication tensor.
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    where U is 4×r, V is 14×r, W is 14×r.
    """
    n, m, p = 2, 2, 7
    # Build the tensor T of shape (n, m, p) where T[i,j,k] = 1 if the (i,k) element
    # of the product depends on A[i,j]*B[j,k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # every triple contributes
    
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a rank-r decomposition of the 3-tensor.
    # Use iterative method: fix two factors, optimize the third.
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    for target_rank in range(4, min(max_rank + 1, n * m * p + 1)):
        if best_U is not None and best_rank <= target_rank:
            break
        
        # Try multiple random initializations
        for trial in range(30):
            rng = np.random.RandomState(42 + trial * 1000 + target_rank)
            
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            
            # Alternating least squares
            converged = False
            for it in range(max_iters):
                # Update W: for each (k,i), solve min_W sum_r (U[i*m+j] * V[j*p+k]) * W[k*n+i] = T[i,j,k]
                # This is a linear system in W for fixed U, V.
                # W[k*n+i, r] appears in T[i,j,k] as U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                # This couples all r for a given (i,j,k) through W[k*n+i, :].
                
                # Better approach: vectorize. Let A = U (n*m × r), B = V (m*p × r), C = W (p*n × r).
                # T[i,j,k] = sum_r A[i*m+j, r] * B[j*p+k, r] * C[k*n+i, r]
                # This is the "tensor train" / CP decomposition form.
                
                # For ALS: fix A, B, update C.
                # For each (k,i): T[i,j,k] for all j gives:
                #   sum_j? No, T[i,j,k] is separate for each (i,j,k).
                # Actually C[k*n+i, r] appears in T[i,j,k] with coefficient A[i*m+j, r]*B[j*p+k, r].
                # So for fixed (k,i), varying j: T[i,j,k] = sum_r C[k*n+i, r] * (A[i*m+j, r] * B[j*p+k, r])
                # This is a system: for each j, T[i,j,k] = dot(C[k*n+i, :], A[i*m+j, :] * B[j*p+k, :])
                # So C[k*n+i, :] solves: sum over j? No, each j gives one equation.
                # We have m equations (for j=0,...,m-1) and r unknowns (C[k*n+i, 0..r-1]).
                # If r > m, this is underdetermined. Use least squares over all (i,j,k) jointly.
                
                # Let's set up the full linear system for C given A, B:
                # For each (i,j,k): T[i,j,k] = sum_r C[k*n+i, r] * M[i,j,k,r]
                # where M[i,j,k,r] = A[i*m+j, r] * B[j*p+k, r]
                # This is a linear system in C with (n*m*p) equations and (p*n*r) unknowns.
                
                # Build matrix: rows indexed by (i,j,k), columns by (k,i,r)
                num_eq = n * m * p
                num_unk = p * n * target_rank
                
                M_mat = np.zeros((num_eq, num_unk))
                rhs = np.zeros(num_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rhs[eq_idx] = T[i, j, k]
                            for r in range(target_rank):
                                col_idx = (k * n + i) * target_rank + r
                                M_mat[eq_idx, col_idx] = A[i * m + j, r] * B[j * p + k, r]
                            eq_idx += 1
                
                # Solve least squares
                C_new, _, _, _ = np.linalg.lstsq(M_mat, rhs, rcond=None)
                C = C_new.reshape(p * n, target_rank)
                
                # Similarly update A given B, C:
                # T[i,j,k] = sum_r A[i*m+j, r] * B[j*p+k, r] * C[k*n+i, r]
                # For fixed (i,j): T[i,j,k] for all k: sum_k? No, each k is separate.
                # A[i*m+j, r] appears in T[i,j,k] with coefficient B[j*p+k, r]*C[k*n+i, r].
                # For fixed (i,j), varying k: T[i,j,k] = sum_r A[i*m+j, r] * (B[j*p+k, r] * C[k*n+i, r])
                # p equations, r unknowns.
                
                M_mat_A = np.zeros((num_eq, n * m * target_rank))
                rhs_A = np.zeros(num_eq)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rhs_A[eq_idx] = T[i, j, k]
                            for r in range(target_rank):
                                col_idx = (i * m + j) * target_rank + r
                                M_mat_A[eq_idx, col_idx] = B[j * p + k, r] * C[k * n + i, r]
                            eq_idx += 1
                
                A_new, _, _, _ = np.linalg.lstsq(M_mat_A, rhs_A, rcond=None)
                A = A_new.reshape(n * m, target_rank)
                
                # Update B given A, C:
                # T[i,j,k] = sum_r A[i*m+j, r] * B[j*p+k, r] * C[k*n+i, r]
                # For fixed (j,k): varying i: T[i,j,k] = sum_r B[j*p+k, r] * (A[i*m+j, r] * C[k*n+i, r])
                # n equations, r unknowns.
                
                M_mat_B = np.zeros((num_eq, m * p * target_rank))
                rhs_B = np.zeros(num_eq)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rhs_B[eq_idx] = T[i, j, k]
                            for r in range(target_rank):
                                col_idx = (j * p + k) * target_rank + r
                                M_mat_B[eq_idx, col_idx] = A[i * m + j, r] * C[k * n + i, r]
                            eq_idx += 1
                
                B_new, _, _, _ = np.linalg.lstsq(M_mat_B, rhs_B, rcond=None)
                B = B_new.reshape(m * p, target_rank)
                
                # Check convergence
                if it % 10 == 0 or it == max_iters - 1:
                    # Compute residual
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(A[i*m+j, :], B[j*p+k, :] * C[k*n+i, :])
                                err += (val - T[i,j,k])**2
                    err = np.sqrt(err)
                    if err < tol:
                        converged = True
                        break
            
            # Final error check
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(A[i*m+j, :], B[j*p+k, :] * C[k*n+i, :])
                        err += (val - T[i,j,k])**2
            err = np.sqrt(err)
            
            if err < tol and (best_U is None or target_rank < best_rank):
                best_rank = target_rank
                best_U = A
                best_V = B
                best_W = C
                break
            if best_U is None or target_rank < best_rank:
                if err < (getattr(_find_2x2x7_decomposition, '_best_err', float('inf'))):
                    _find_2x2x7_decomposition._best_err = err
                    _find_2x2x7_decomposition._best_U = A.copy()
                    _find_2x2x7_decomposition._best_V = B.copy()
                    _find_2x2x7_decomposition._best_W = C.copy()
                    _find_2x2x7_decomposition._best_rank = target_rank
    
    if best_U is not None:
        return best_rank, best_U, best_V, best_W
    
    # Return best found even if not fully converged
    if hasattr(_find_2x2x7_decomposition, '_best_U'):
        return _find_2x2x7_decomposition._best_rank, _find_2x2x7_decomposition._best_U, _find_2x2x7_decomposition._best_V, _find_2x2x7_decomposition._best_W
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return rank, U, V, W


def _kronecker_decompose(rank_2x2x7, U_small, V_small, W_small):
    """
    Build the 2×4×7 decomposition from the 2×2×7 decomposition using Kronecker product.
    
    The 2×4×7 tensor T[i, (j1,j2), k] = T_2x2x7[i, j1, k] * delta[j1, j2]
    where the j-index in 2×4 is split into (j1, j2) with j1,j2 in {0,1}.
    
    Actually, the mapping is: j in {0,1,2,3} → (j1, j2) where j = j1*2 + j2 or j = j2*2 + j1.
    Let's use j = j1*2 + j2, so j1 = j//2, j2 = j%2.
    
    The 2×2×1 identity tensor: E[j1, j2] = delta[j1, j2], rank 2.
    Decomposition of E: U_E[j1, r] * V_E[j2, r] * W_E[r] where we need E[j1,j2] = sum_r U_E[j1,r]*V_E[j2,r]*W_E[r].
    For identity: E[j1,j2] = delta[j1,j2]. Use r=0: U_E[0,0]=1, V_E[0,0]=1, W_E[0]=1; r=1: U_E[1,1]=1, V_E[1,1]=1, W_E[1]=1.
    So U_E = I_2, V_E = I_2, W_E = [1,1].
    
    Kronecker product decomposition:
    T[i, (j1,j2), k] = sum_{r1,r2} U_small[i*j1_idx + ?, r1] * V_small[?, r1] * W_small[?, r1] * U_E[j1, r2] * V_E[j2, r2] * W_E[r2]
    
    More carefully:
    The full tensor is T_full[i, j, k] where j = j1*2+j2.
    T_full[i, j1*2+j2, k] = T_small[i, j1, k] * delta[j1, j2]
    
    CP decomposition of T_small: T_small[i,j1,k] = sum_{r1} U_s[i*2+j1, r1] * V_s[j1*7+k, r1] * W_s[k*2+i, r1]
    CP decomposition of E: E[j1,j2] = sum_{r2} U_e[j1, r2] * V_e[j2, r2] * W_e[r2]
    
    Kronecker: T_full[i, j1*2+j2, k] = sum_{r1,r2} U_s[i*2+j1, r1] * V_s[j1*7+k, r1] * W_s[k*2+i, r1] * U_e[j1, r2] * V_e[j2, r2] * W_e[r2]
    
    But we need the factors to be in the form:
    U_full[i*4+j, r] * V_full[j*7+k, r] * W_full[k*2+i, r]
    
    where j = j1*2+j2, so i*4+j = i*4+j1*2+j2.
    
    U_full[i*4+j1*2+j2, (r1,r2)] = U_s[i*2+j1, r1] * U_e[j1, r2] * (something for j2?)
    
    Hmm, this doesn't factor cleanly because U depends on j2 but U_s and U_e don't have j2.
    
    Let me reconsider. The Kronecker product of two CP decompositions:
    If T = T1 ⊗ T2 where T1 is (n1×m1×p1) and T2 is (n2×m2×p2),
    and T1[i1,j1,k1] = sum_r U1[i1*m1+j1, r] * V1[j1*p1+k1, r] * W1[k1*n1+i1, r],
    T2[i2,j2,k2] = sum_s U2[i2*m2+j2, s] * V2[j2*p2+k2, s] * W2[k2*n2+i2, s],
    
    Then T[(i1,i2),(j1,j2),(k1,k2)] = T1[i1,j1,k1] * T2[i2,j2,k2]
    = sum_{r,s} U1[i1*m1+j1, r] * V1[j1*p1+k1, r] * W1[k1*n1+i1, r] * U2[i2*m2+j2, s] * V2[j2*p2+k2, s] * W2[k2*n2+i2, s]
    
    With combined index R = (r,s), rank = rank1 * rank2:
    U_full[i1*n2*m2 + i2*m2 + j1*m2 + j2, R] ... this is getting complicated with the index mappings.
    
    For our case:
    T1 = 2×2×7 tensor, T2 = 2×2×1 identity tensor.
    Full tensor: 2×4×7 where the 4-dim is (j1 from T1, j2 from T2).
    
    Let me define the index mapping explicitly:
    - i in {0,1} (from T1's i1)
    - j in {0,1,2,3} = j1*2 + j2 where j1 in {0,1} (T1's j1), j2 in {0,1} (T2's j2)
    - k in {0,...,6} (from T1's k1; T2's k2 is always 0 since p2=1)
    
    T_full[i, j, k] = T1[i, j1, k] * T2[j2, j2, 0] ... wait, T2 is (n2×m2×p2) = (2×2×1).
    T2[i2, j2, k2] with i2 in {0,1}, j2 in {0,1}, k2 in {0}.
    For identity: T2[i2, j2, 0] = delta[i2, j2].
    
    But in the full tensor, the "i2" index is... hmm, this is where it gets tricky.
    
    Actually, I think the correct decomposition is:
    The 2×4×7 matrix multiplication can be seen as:
    A is 2×4, B is 4×7, C is 2×7.
    Split A into two 2×2 blocks: A = [A1, A2] where A1 = A[:, 0:2], A2 = A[:, 2:4].
    Split B into two 4×7 blocks vertically: B = [B1; B2] where B1 = B[0:2, :], B2 = B[2:4, :].
    Then C = A1*B1 + A2*B2.
    
    Each term A1*B1 is 2×2 * 2×7, which is the 2×2×7 tensor.
    So C = C1 + C2 where C1 = A1*B1 and C2 = A2*B2.
    
    This means we need rank(2×2×7) + rank(2×2×7) = 2*rank(2×2×7) for the full decomposition!
    That's just 2 * rank of the small tensor, which matches the Kronecker product idea.
    
    So the construction is:
    - Find CP decomposition of 2×2×7 tensor with rank r.
    - The full 2×4×7 decomposition has rank 2r.
    - U_full has shape (8, 2r), V_full has shape (28, 2r), W_full has shape (14, 2r).
    
    For the first block (j1=0, i.e., j=0,1):
    U_full[i*4 + j, r] = U_small[i*2 + j, r] for j in {0,1} (j1=0, j2=j)
    Wait, j = j1*2 + j2, so for j1=0: j = j2 = 0 or 1.
    For j1=1: j = 2+j2 = 2 or 3.
    
    U_full[i*4 + j, (block, r_small)] where block in {0,1} corresponds to j1.
    U_full[i*4 + j1*2 + j2, block*rank_small + r_small] = U_small[i*2 + j1, r_small] if block == j1, else 0.
    
    Similarly for V and W.
    """
    n, m, p = 2, 4, 7
    r_small = rank_2x2x7
    rank_full = 2 * r_small
    
    U_full = np.zeros((n * m, rank_full))
    V_full = np.zeros((m * p, rank_full))
    W_full = np.zeros((p * n, rank_full))
    
    # U_small has shape (n*2, r_small) = (4, r_small)
    # V_small has shape (2*7, r_small) = (14, r_small)  
    # W_small has shape (7*2, r_small) = (14, r_small)
    
    for j1 in range(2):
        offset = j1 * r_small
        for i in range(2):
            for j2 in range(2):
                j = j1 * 2 + j2
                U_full[i * 4 + j, offset:offset + r_small] = U_small[i * 2 + j1, :]
        
        for j2 in range(2):
            for k in range(7):
                j = j1 * 2 + j2
                V_full[j * 7 + k, offset:offset + r_small] = V_small[j1 * 7 + k, :]
        
        for k in range(7):
            for i in range(2):
                W_full[k * 2 + i, offset:offset + r_small] = W_small[k * 2 + i, :]
    
    return rank_full, U_full, V_full, W_full


def _verify_decomposition(U, V, W, n=2, m=4, p=7):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = U.shape[1]
    # Check: C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each (i,j,k): the coefficient should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-8:
                    return False, val, (i,j,k)
    return True, 0.0, None


def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2×4 matrix by a 4×7 matrix.
    
    Uses the factorization 4=2×2 to reduce to a Kronecker product of
    a 2×2×7 tensor and a 2×2×1 identity tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Step 1: Find decomposition of 2×2×7 tensor
    rank_small, U_small, V_small, W_small = _find_2x2x7_decomposition(
        max_rank=14, max_iters=100, tol=1e-10
    )
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 5:
        # Not enough time, fall back to schoolbook
        pass
    
    # Step 2: Build full 2×4×7 decomposition via Kronecker product
    rank_full, U_full, V_full, W_full = _kronecker_decompose(rank_small, U_small, V_small, W_small)
    
    # Step 3: Verify
    valid, err, detail = _verify_decomposition(U_full, V_full, W_full)
    
    if not valid:
        # Try with more iterations or fall back
        if time.time() < deadline:
            rank_small, U_small, V_small, W_small = _find_2x2x7_decomposition(
                max_rank=14, max_iters=200, tol=1e-12
            )
            rank_full, U_full, V_full, W_full = _kronecker_decompose(rank_small, U_small, V_small, W_small)
            valid, err, detail = _verify_decomposition(U_full, V_full, W_full)
    
    if not valid:
        # Final fallback: schoolbook decomposition
        n, m, p = 2, 4, 7
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists for output
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
