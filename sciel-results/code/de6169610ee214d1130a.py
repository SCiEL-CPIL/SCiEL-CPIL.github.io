"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
    col_offset = 0
    
    for block in range(4):
        # Block handles columns 2*block and 2*block+1 of B
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for the 2x2 * 2x2 part (A cols 0,1 times B rows 0,1)
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1   # a00
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1   # a10
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        W[2*k0+1, t] += 1  # C10
        W[2*k1+1, t] += -1  # C11
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1   # a00
        V[k1, t] += 1  # b01
        V[8+k1, t] += -1  # b11
        W[2*k1, t] += 1  # C01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1   # a11
        V[8+k0, t] += 1  # b10
        V[k0, t] += -1  # b00
        W[2*k0, t] += 1  # C00
        W[2*k0+1, t] += 1  # C10
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1   # a00
        U[1, t] += 1   # a01
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += -1  # C00
        W[2*k1, t] += 1  # C01
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1   # a10
        U[0, t] += -1  # a00
        V[k0, t] += 1  # b00
        V[k1, t] += 1  # b01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1   # a01
        U[4, t] += -1  # a11
        V[8+k0, t] += 1  # b10
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        col_offset += 1
        
        # Now the 4 rank-1 terms for A_right * B_bottom
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1   # A[ii, 2]
                V[16+k_val, t] = 1  # B[2, k_val]
                W[k_val*2+ii, t] = 1  # C[ii, k_val]
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44 columns, got {col_offset}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_rank42_decomposition():
    """
    Build explicit rank-42 decomposition for 2x3 * 3x8.
    Strategy: Split B (3x8) into two blocks of 3x4: B = [B_0, B_1]
    Each C_t = A * B_t is a 2x3 * 3x4 = 2x4 multiplication.
    
    For each 2x3*3x4 sub-problem, we can potentially use a rank-21 scheme.
    Let's analyze M_{2,3,4}.
    A is 2x3, B is 3x4, C is 2x4.
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    We can view this as two independent 2x3 * 3x2 blocks?
    B = [B_left, B_right] where B_left is 3x2 (cols 0,1), B_right is 3x2 (cols 2,3).
    C = [C_left, C_right] where C_left = A * B_left (2x2), C_right = A * B_right (2x2).
    
    If we treat them independently, cost is 2 * 11 = 22 per block. Total 44.
    Can we share?
    The A matrix is the same.
    In the rank-11 scheme for 2x3*3x2:
    7 terms for A_left * B_top (Strassen on 2x2*2x2 using A cols 0,1 and B rows 0,1)
    4 terms for A_right * B_bottom (A col 2 and B row 2)
    
    For the second 3x2 block (cols 2,3 of B, i.e. rows 0,1,2 of B_right):
    We need A_left * B_right_top (A cols 0,1, B_right rows 0,1) -> Strassen 7 terms.
    We need A_right * B_right_bottom (A col 2, B_right row 2) -> 4 terms.
    
    Total 11 + 11 = 22.
    Is there a way to reduce this?
    
    Let's look at the structure of the 4 terms for A_right * B_bottom.
    They are:
    t0: U[2, t]=1, V[16, t]=1, W[0, t]=1  (A[0,2] * B[2,0] -> C[0,0])
    t1: U[2, t]=1, V[17, t]=1, W[1, t]=1  (A[0,2] * B[2,1] -> C[0,1])
    t2: U[5, t]=1, V[16, t]=1, W[2, t]=1  (A[1,2] * B[2,0] -> C[1,0])
    t3: U[5, t]=1, V[17, t]=1, W[3, t]=1  (A[1,2] * B[2,1] -> C[1,1])
    
    For the second block (cols 2,3 of B):
    B_right rows are B[0,2], B[0,3], B[1,2], B[1,3], B[2,2], B[2,3].
    V indices:
    B[0,2] -> 0*8+2 = 2
    B[0,3] -> 0*8+3 = 3
    B[1,2] -> 1*8+2 = 10
    B[1,3] -> 1*8+3 = 11
    B[2,2] -> 2*8+2 = 18
    B[2,3] -> 2*8+3 = 19
    
    W indices for C_right (cols 2,3 of C):
    C[0,2] -> 2*2+0 = 4
    C[0,3] -> 3*2+0 = 6
    C[1,2] -> 2*2+1 = 5
    C[1,3] -> 3*2+1 = 7
    
    The 4 terms for A_right * B_right_bottom (A col 2, B_right row 2 i.e. B[2,2], B[2,3]):
    t4: U[2, t]=1, V[18, t]=1, W[4, t]=1  (A[0,2] * B[2,2] -> C[0,2])
    t5: U[2, t]=1, V[19, t]=1, W[6, t]=1  (A[0,2] * B[2,3] -> C[0,3])
    t6: U[5, t]=1, V[18, t]=1, W[5, t]=1  (A[1,2] * B[2,2] -> C[1,2])
    t7: U[5, t]=1, V[19, t]=1, W[7, t]=1  (A[1,2] * B[2,3] -> C[1,3])
    
    Notice that U[2, t] and U[5, t] are the same in both blocks.
    Can we combine t0 and t4?
    t0: U[2]=1, V[16]=1, W[0]=1
    t4: U[2]=1, V[18]=1, W[4]=1
    They have different V and W, so they are distinct rank-1 tensors.
    However, if we look at the Strassen part:
    Block 1 Strassen uses V[0], V[1], V[8], V[9] (B[0,0], B[0,1], B[1,0], B[1,1])
    Block 2 Strassen uses V[2], V[3], V[10], V[11] (B[0,2], B[0,3], B[1,2], B[1,3])
    These are disjoint.
    
    So simple block decomposition gives 44.
    
    Can we do better?
    What if we use a different splitting?
    Split A into A_top (row 0) and A_bottom (row 1)?
    A is 2x3.
    C[0, k] = A[0, 0]B[0, k] + A[0, 1]B[1, k] + A[0, 2]B[2, k]
    C[1, k] = A[1, 0]B[0, k] + A[1, 1]B[1, k] + A[1, 2]B[2, k]
    
    This is just 2 independent row computations.
    Row 0: 1x3 * 3x8 = 1x8. Rank 8 (schoolbook) or less?
    Row 1: 1x3 * 3x8 = 1x8. Rank 8.
    Total 16? No, the tensor is not separable in that way because the "rank" of the bilinear map counts the number of products.
    For 1x3 * 3x8, the bilinear map is (a, b) -> a*b.
    a is 3-vector, b is 8-vector (actually 3x8 matrix, but for one row it's 3x1 * 3x8? No.
    A is 2x3. B is 3x8.
    Row i of A is 1x3.
    Row i of C is 1x8.
    C_i = A_i * B.
    A_i has 3 elements. B has 24 elements.
    The bilinear map is from (3, 24) to 8.
    C_i[k] = sum_{j=0}^2 A_i[j] B[j, k].
    This requires 3*8 = 24 multiplications for one row if done naively?
    No, A_i[j] is a scalar. B[j, k] is a scalar.
    We compute 3 sums for each k?
    C_i[k] = A_i[0]B[0,k] + A_i[1]B[1,k] + A_i[2]B[2,k].
    This is 3 multiplications per k, total 24 multiplications for one row.
    Two rows: 48.
    
    But we can share?
    No, A_i[0] and A_i[1] are different.
    
    What if we use Strassen-like tricks on the 3x3 * 3x8?
    
    Let's try to construct a rank-42 decomposition manually or via search.
    The prompt suggests "applying the minimal rank 4 decomposition for 2x2 matrices to each block scaled appropriately".
    Rank 4 for 2x2? That's not Strassen (7). Rank 4 is for 2x2 * 2x2? No, Strassen is 7.
    Maybe it means something else.
    
    Let's stick to the rank-44 construction and try to optimize it or find a rank-42 one via search.
    """
    # This function is a placeholder for a potential rank-42 construction.
    # For now, we return None to indicate we don't have a guaranteed one yet.
    # We will rely on the search to find one if it exists.
    return None

def _als_decompose(rank, max_iters=50, deadline=None, init=None):
    """
    Alternating Least Squares for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    
    if init:
        U = init[0].copy()
        V = init[1].copy()
        W = init[2].copy()
    else:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V[b, :] = x
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W[c, :] = x
                except:
                    pass
                    
        # Check convergence
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    err = abs(T[a,b,c] - 1.0)
                    if err > max_err:
                        max_err = err
        target = np.zeros((n*m, m*p, p*n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = True
        non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 140
    
    # First, try the explicit rank-44 decomposition
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            rounded = _round_and_verify(U44, V44, W44)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
            else:
                best_U, best_V, best_W = U44, V44, W44
                best_rank = 44
    except Exception as e:
        pass
    
    # Try to find a rank-42 decomposition using the structural insight
    # We initialize ALS with a perturbed rank-44 solution and try to reduce rank
    if best_rank == 44:
        # Try rank 42 directly with multiple random restarts
        for rank in range(42, 20, -1):
            if time.time() > deadline:
                break
            
            # Increase trials for lower ranks as they are harder
            trials = 10 if rank > 40 else 20
            
            for trial in range(trials):
                if time.time() > deadline:
                    break
                
                # Initialize with random noise
                result = _als_decompose(rank, max_iters=100, deadline=deadline)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                            break
                        else:
                            best_U, best_V, best_W = U, V, W
                            best_rank = rank
                            break
            
            if best_rank < 44:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
