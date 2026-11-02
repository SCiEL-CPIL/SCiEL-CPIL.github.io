"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _verify_full(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _laderman_3x3x3():
    """
    Laderman's 7-rank decomposition for 3x3x3 matrix multiplication.
    U (9,7) indexed [i*3+j], V (9,7) indexed [j*3+k], W (9,7) indexed [k*3+i]
    """
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    # r=0: a00*b00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # r=1: a00*(b10+b20)
    U[0, 1] = 1; V[3, 1] = 1; V[6, 1] = 1; W[3, 1] = 1; W[6, 1] = 1
    # r=2: a10*(b01-b00)
    U[3, 2] = 1; V[1, 2] = 1; V[0, 2] = -1; W[4, 2] = 1; W[1, 2] = -1
    # r=3: a20*(b02-b00)
    U[6, 3] = 1; V[2, 3] = 1; V[0, 3] = -1; W[8, 3] = 1; W[2, 3] = -1
    # r=4: (a11+a21)*(b10-b00)
    U[4, 4] = 1; U[7, 4] = 1; V[3, 4] = 1; V[0, 4] = -1; W[4, 4] = 1; W[1, 4] = -1; W[7, 4] = 1; W[4, 4] = 1
    # Wait, W indexing: W[k*3+i]. For c11: k=1,i=1 -> W[4]. For c10: k=0,i=1 -> W[1]. For c21: k=1,i=2 -> W[5]. For c20: k=0,i=2 -> W[2].
    # Let me redo r=4 and r=5 and r=6 carefully.
    
    # Reset and redo
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    # r=0: a00*b00 -> c00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1  # c[0,0]: k=0,i=0 -> W[0*3+0]=W[0]
    
    # r=1: a00*(b10+b20) -> c01, c02
    U[0, 1] = 1
    V[3, 1] = 1  # b[1,0]: j=1,k=0 -> V[1*3+0]=V[3]
    V[6, 1] = 1  # b[2,0]: j=2,k=0 -> V[2*3+0]=V[6]
    W[3, 1] = 1  # c[0,1]: k=1,i=0 -> W[1*3+0]=W[3]
    W[6, 1] = 1  # c[0,2]: k=2,i=0 -> W[2*3+0]=W[6]
    
    # r=2: a10*(b01-b00) -> c11, -c10
    U[3, 2] = 1  # a[1,0]: i=1,j=0 -> U[1*3+0]=U[3]
    V[1, 2] = 1  # b[0,1]: j=0,k=1 -> V[0*3+1]=V[1]
    V[0, 2] = -1 # b[0,0]
    W[4, 2] = 1  # c[1,1]: k=1,i=1 -> W[1*3+1]=W[4]
    W[1, 2] = -1 # c[1,0]: k=0,i=1 -> W[0*3+1]=W[1]
    
    # r=3: a20*(b02-b00) -> c22, -c20
    U[6, 3] = 1  # a[2,0]: i=2,j=0 -> U[2*3+0]=U[6]
    V[2, 3] = 1  # b[0,2]: j=0,k=2 -> V[0*3+2]=V[2]
    V[0, 3] = -1 # b[0,0]
    W[8, 3] = 1  # c[2,2]: k=2,i=2 -> W[2*3+2]=W[8]
    W[2, 3] = -1 # c[2,0]: k=0,i=2 -> W[0*3+2]=W[2]
    
    # r=4: (a11+a21)*(b10-b00) -> c11, -c10, c21, -c20
    U[4, 4] = 1  # a[1,1]: i=1,j=1 -> U[1*3+1]=U[4]
    U[7, 4] = 1  # a[2,1]: i=2,j=1 -> U[2*3+1]=U[7]
    V[3, 4] = 1  # b[1,0]
    V[0, 4] = -1 # b[0,0]
    W[4, 4] = 1  # c[1,1]
    W[1, 4] = -1 # c[1,0]
    W[5, 4] = 1  # c[2,1]: k=1,i=2 -> W[1*3+2]=W[5]
    W[2, 4] = -1 # c[2,0]
    
    # r=5: (a12+a22)*(b20-b00) -> c12, -c10, c22, -c20
    U[5, 5] = 1  # a[1,2]: i=1,j=2 -> U[1*3+2]=U[5]
    U[8, 5] = 1  # a[2,2]: i=2,j=2 -> U[2*3+2]=U[8]
    V[6, 5] = 1  # b[2,0]
    V[0, 5] = -1 # b[0,0]
    W[7, 5] = 1  # c[1,2]: k=2,i=1 -> W[2*3+1]=W[7]
    W[1, 5] = -1 # c[1,0]
    W[8, 5] = 1  # c[2,2]
    W[2, 5] = -1 # c[2,0]
    
    # r=6: (a01+a02)*(b21-b20) -> c02, -c01
    U[1, 6] = 1  # a[0,1]: i=0,j=1 -> U[0*3+1]=U[1]
    U[2, 6] = 1  # a[0,2]: i=0,j=2 -> U[0*3+2]=U[2]
    V[7, 6] = 1  # b[2,1]: j=2,k=1 -> V[2*3+1]=V[7]
    V[6, 6] = -1 # b[2,0]
    W[6, 6] = 1  # c[0,2]
    W[3, 6] = -1 # c[0,1]
    
    return U, V, W

def _matmul_3x3x2():
    """
    Decomposition for 3x3 * 3x2 = 3x2.
    A is 3x3 (U: 9 entries), B is 3x2 (V: 6 entries), C is 3x2 (W: 6 entries)
    W indexed by [k*3+i] for k in {0,1}, i in {0,1,2}
    """
    rank = 18  # schoolbook
    U = np.zeros((9, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i*3+j, col] = 1
                V[j*2+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return U, V, W

def _combine_3x3x8():
    """
    Combine Laderman 3x3x3 blocks with 3x3x2 block for full 3x3x8.
    """
    n, m, p = 3, 3, 8
    U_l, V_l, W_l = _laderman_3x3x3()
    U_32, V_32, W_32 = _matmul_3x3x2()
    
    # Verify Laderman
    if not _verify_full(U_l, V_l, W_l, 3, 3, 3):
        return None
    if not _verify_full(U_32, V_32, W_32, 3, 3, 2):
        return None
    
    rank_l = U_l.shape[1]  # 7
    rank_32 = U_32.shape[1]  # 18
    total_rank = 2 * rank_l + rank_32  # 32
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    # Block 0: B columns 0-2, C columns 0-2
    for r in range(rank_l):
        U[:, col] = U_l[:, r]
        for j in range(3):
            for k in range(3):
                V[j*8+k, col] = V_l[j*3+k, r]
        for k in range(3):
            for i in range(3):
                W[k*3+i, col] = W_l[k*3+i, r]
        col += 1
    
    # Block 1: B columns 3-5, C columns 3-5
    for r in range(rank_l):
        U[:, col] = U_l[:, r]
        for j in range(3):
            for k in range(3):
                V[j*8+(3+k), col] = V_l[j*3+k, r]
        for k in range(3):
            for i in range(3):
                W[(3+k)*3+i, col] = W_l[k*3+i, r]
        col += 1
    
    # Block 2: B columns 6-7, C columns 6-7
    for r in range(rank_32):
        U[:, col] = U_32[:, r]
        for j in range(3):
            for k in range(2):
                V[j*8+(6+k), col] = V_32[j*2+k, r]
        for k in range(2):
            for i in range(3):
                W[(6+k)*3+i, col] = W_32[k*3+i, r]
        col += 1
    
    if _verify_full(U, V, W, n, m, p):
        return U, V, W
    return None

def _try_alternating_minimization(n, m, p, target_rank, max_iter=20, tol=1e-10, deadline=None):
    """Try to find a target_rank decomposition via alternating minimization."""
    np.random.seed(42)
    U_r = np.random.randn(n*m, target_rank)
    V_r = np.random.randn(m*p, target_rank)
    W_r = np.random.randn(p*n, target_rank)
    
    num_eq = n * m * p
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve for W
        A_mat = np.zeros((num_eq, p*n*target_rank))
        b_vec = np.ones(num_eq)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        A_mat[eq, (k*n+i)*target_rank + r] = U_r[i*m+j, r] * V_r[j*p+k, r]
                    eq += 1
        W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W_r = W_flat.reshape(p*n, target_rank)
        
        # Solve for U
        A_mat = np.zeros((num_eq, n*m*target_rank))
        b_vec = np.ones(num_eq)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        A_mat[eq, (i*m+j)*target_rank + r] = V_r[j*p+k, r] * W_r[k*n+i, r]
                    eq += 1
        U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        U_r = U_flat.reshape(n*m, target_rank)
        
        # Solve for V
        A_mat = np.zeros((num_eq, m*p*target_rank))
        b_vec = np.ones(num_eq)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        A_mat[eq, (j*p+k)*target_rank + r] = U_r[i*m+j, r] * W_r[k*n+i, r]
                    eq += 1
        V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        V_r = V_flat.reshape(m*p, target_rank)
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r] for r in range(target_rank))
                    err += (s - 1.0)**2
        err = np.sqrt(err / num_eq)
        
        if err < tol:
            return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 72
    
    # Try structured decomposition
    if time.time() < deadline - 30:
        result = _combine_3x3x8()
        if result is not None:
            U_struct, V_struct, W_struct = result
            struct_rank = U_struct.shape[1]
            if struct_rank < best_rank:
                best_U, best_V, best_W = U_struct, V_struct, W_struct
                best_rank = struct_rank
    
    # Try alternating minimization to reduce further
    if time.time() < deadline - 15:
        for target_rank in range(max(best_rank - 10, 20), best_rank):
            if time.time() > deadline - 10:
                break
            result = _try_alternating_minimization(n, m, p, target_rank, max_iter=10, 
                                                     tol=1e-10, deadline=deadline - 10)
            if result is not None:
                U_new, V_new, W_new = result
                if _verify_full(U_new, V_new, W_new, n, m, p, tol=1e-8):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = target_rank
                    break
    
    # Try rounding to half-integers
    U_round = np.round(best_U * 2) / 2
    V_round = np.round(best_V * 2) / 2
    W_round = np.round(best_W * 2) / 2
    if _verify_full(U_round, V_round, W_round, n, m, p, tol=1e-10):
        best_U, best_V, best_W = U_round, V_round, W_round
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
# construct() is the entry point
# EVOLVE-BLOCK-END
