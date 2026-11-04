import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Build target tensor
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Strategy: Compose 4x4x4 (Strassen, rank 49) + 4x4x3 (schoolbook, rank 48)
    # Total rank = 97

    # Part 1: 4x4x4 using Strassen decomposition (rank 49)
    # 4x4 = 2*(2x2), so use Kronecker product of 2x2x2 Strassen
    # Strassen 2x2x2 has rank 7
    
    # Get Strassen 2x2x2 decomposition
    U_2, V_2, W_2 = _strassen_2x2x2()
    # U_2, V_2, W_2 have shape (4, 7)
    
    # Kronecker product: 4x4x4 rank = 7*7 = 49
    U_444 = []
    V_444 = []
    W_444 = []
    for r1 in range(7):
        for r2 in range(7):
            u_vec = np.kron(U_2[:, r1], U_2[:, r2])  # length 16
            v_vec = np.kron(V_2[:, r1], V_2[:, r2])  # length 16
            w_vec = np.kron(W_2[:, r1], W_2[:, r2])  # length 16
            U_444.append(u_vec)
            V_444.append(v_vec)
            W_444.append(w_vec)
    
    U_444 = np.array(U_444).T  # (16, 49)
    V_444 = np.array(V_444).T  # (16, 49)
    W_444 = np.array(W_444).T  # (16, 49)
    
    # Verify 4x4x4 part
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    T_444_recon = np.einsum('ak,bk,ck->abc', U_444, V_444, W_444)
    # Check against target for first 4 columns of W (k=0..3)
    # The 4x4x4 tensor has T[i*4+j, j*4+k, k*4+i] = 1 for k in 0..3
    T_444_target = np.zeros((16, 16, 16), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 4 + k
                c = k * 4 + i
                T_444_target[a, b, c] = 1
    
    if not np.allclose(T_444_recon, T_444_target, atol=1e-10):
        # Fallback to schoolbook for entire thing
        return _schoolbook(n, m, p)
    
    # Part 2: 4x4x3 schoolbook (rank 48)
    # For k in 0..2 (3 columns), use schoolbook
    U_443 = []
    V_443 = []
    W_443 = []
    for i in range(4):
        for j in range(4):
            for k in range(3):  # k = 0, 1, 2
                u_vec = np.zeros(16)
                v_vec = np.zeros(16)
                w_vec = np.zeros(16)
                u_vec[i * 4 + j] = 1
                v_vec[j * 4 + k] = 1
                w_vec[k * 4 + i] = 1
                U_443.append(u_vec)
                V_443.append(v_vec)
                W_443.append(w_vec)
    
    U_443 = np.array(U_443).T  # (16, 48)
    V_443 = np.array(V_443).T  # (16, 48)
    W_443 = np.array(W_443).T  # (16, 48)
    
    # Now combine: The full problem is 4x4x7
    # U has dim 16, V has dim 28 (4*7), W has dim 28 (7*4)
    # For the 4x4x4 part: V indices are j*4+k for k in 0..3, W indices are k*4+i for k in 0..3
    # For the 4x4x3 part: V indices are j*7+k for k in 0..2, W indices are k*4+i for k in 0..2
    
    # Wait, I need to be more careful. The V dimension is m*p = 4*7 = 28
    # V index = j*p + k = j*7 + k
    # W index = k*n + i = k*4 + i
    
    # For the "4x4x4" interpretation, I was treating it as if p=4, but actually p=7.
    # Let me reconsider.
    
    # Actually, the correct approach:
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # = T[i*4+j, j*7+k, k*4+i] = 1
    
    # I can split by k: for k in 0..6
    # For each k, we have a 4x4 sub-problem (i,j vary)
    # But these are not independent because they share the same i,j structure.
    
    # Better: Use the block structure differently.
    # Let's just use pure schoolbook but try to find a better structured decomposition.
    
    # Alternative: Try to use a rank-1 peeling with better initialization
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Try greedy peeling with the full tensor
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    max_terms = 200
    
    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break
        
        # Greedy: find best rank-1 term
        best_score = 0
        best_u, best_v, best_w = None, None, None
        
        # Multiple random restarts
        for restart in range(10):
            if time.time() > deadline:
                break
            
            if restart == 0:
                cu = np.sign(np.sum(R, axis=(1, 2)))
                cv = np.sign(np.sum(R, axis=(0, 2)))
                cw = np.sign(np.sum(R, axis=(0, 1)))
            else:
                cu = np.random.choice([-1, 0, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 0, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 0, 1], size=dim_w).astype(np.float64)
            
            for it in range(20):
                if time.time() > deadline:
                    break
                
                M_u = np.einsum('abc,b,c->a', R, cv, cw)
                new_u = np.sign(M_u)
                
                M_v = np.einsum('abc,a,c->b', R, new_u, cw)
                new_v = np.sign(M_v)
                
                M_w = np.einsum('abc,a,b->c', R, new_u, new_v)
                new_w = np.sign(M_w)
                
                cu, cv, cw = new_u, new_v, new_w
            
            score = np.einsum('abc,a,b,c', R, cu, cv, cw)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()
        
        if best_score <= 0 or best_u is None:
            break
        
        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(R, 0, atol=1e-10):
            break
    
    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            if rank < n * m * p:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return _schoolbook(n, m, p)


def _strassen_2x2x2():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    
    # Strassen products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors (for A indices: a11=0, a12=1, a21=2, a22=3)
    # V vectors (for B indices: b11=0, b12=1, b21=2, b22=3)
    # W vectors (for C indices: c11=0, c12=1, c21=2, c22=3)
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21 += M2, C22 -= M2
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 += M3, C22 += M3
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11 += M4, C21 += M4
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 -= M5, C12 += M5
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22 += M6
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11 += M7
    
    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
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

# EVOLVE-BLOCK-END
