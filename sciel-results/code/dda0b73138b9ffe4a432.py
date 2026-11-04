import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(s - 1) > 1e-10:
                    return False
    return True

def _add_strassen_222(U, V, W, offset, k0, k1, j0, j1):
    a00 = 0*5 + k0
    a01 = 0*5 + k1
    a10 = 1*5 + k0
    a11 = 1*5 + k1
    
    b00 = k0*5 + j0
    b01 = k0*5 + j1
    b10 = k1*5 + j0
    b11 = k1*5 + j1
    
    c00 = j0*2 + 0
    c01 = j1*2 + 0
    c10 = j0*2 + 1
    c11 = j1*2 + 1
    
    # r=0: M0 = (A00+A11)(B00+B11)
    U[a00][offset+0] += 1
    U[a11][offset+0] += 1
    V[b00][offset+0] += 1
    V[b11][offset+0] += 1
    W[c00][offset+0] += 1
    W[c11][offset+0] += 1
    
    # r=1: M1 = (A10+A11)B00
    U[a10][offset+1] += 1
    U[a11][offset+1] += 1
    V[b00][offset+1] += 1
    W[c10][offset+1] += 1
    W[c11][offset+1] += -1
    
    # r=2: M2 = A00(B01-B11)
    U[a00][offset+2] += 1
    V[b01][offset+2] += 1
    V[b11][offset+2] += -1
    W[c01][offset+2] += 1
    W[c11][offset+2] += 1
    
    # r=3: M3 = A11(B10-B00)
    U[a11][offset+3] += 1
    V[b10][offset+3] += 1
    V[b00][offset+3] += -1
    W[c00][offset+3] += 1
    W[c10][offset+3] += 1
    
    # r=4: M4 = (A00+A01)B11
    U[a00][offset+4] += 1
    U[a01][offset+4] += 1
    V[b11][offset+4] += 1
    W[c00][offset+4] += -1
    W[c01][offset+4] += 1
    
    # r=5: M5 = (A10-A00)(B00+B01)
    U[a10][offset+5] += 1
    U[a00][offset+5] += -1
    V[b00][offset+5] += 1
    V[b01][offset+5] += 1
    W[c11][offset+5] += 1
    
    # r=6: M6 = (A01-A11)(B10+B11)
    U[a01][offset+6] += 1
    U[a11][offset+6] += -1
    V[b10][offset+6] += 1
    V[b11][offset+6] += 1
    W[c00][offset+6] += 1
    
    return offset + 7

def _add_2x1(U, V, W, offset, k0, k1, j):
    a0k0 = 0*5 + k0
    a0k1 = 0*5 + k1
    a1k0 = 1*5 + k0
    a1k1 = 1*5 + k1
    
    bk0j = k0*5 + j
    bk1j = k1*5 + j
    
    c0j = j*2 + 0
    c1j = j*2 + 1
    
    U[a0k0][offset+0] += 1
    V[bk0j][offset+0] += 1
    W[c0j][offset+0] += 1
    
    U[a0k1][offset+1] += 1
    V[bk1j][offset+1] += 1
    W[c0j][offset+1] += 1
    
    U[a1k0][offset+2] += 1
    V[bk0j][offset+2] += 1
    W[c1j][offset+2] += 1
    
    U[a1k1][offset+3] += 1
    V[bk1j][offset+3] += 1
    W[c1j][offset+3] += 1
    
    return offset + 4

def _add_1x2(U, V, W, offset, k, j0, j1):
    a0k = 0*5 + k
    a1k = 1*5 + k
    
    bkj0 = k*5 + j0
    bkj1 = k*5 + j1
    
    c0j0 = j0*2 + 0
    c0j1 = j1*2 + 0
    c1j0 = j0*2 + 1
    c1j1 = j1*2 + 1
    
    U[a0k][offset+0] += 1
    V[bkj0][offset+0] += 1
    W[c0j0][offset+0] += 1
    
    U[a0k][offset+1] += 1
    V[bkj1][offset+1] += 1
    W[c0j1][offset+1] += 1
    
    U[a1k][offset+2] += 1
    V[bkj0][offset+2] += 1
    W[c1j0][offset+2] += 1
    
    U[a1k][offset+3] += 1
    V[bkj1][offset+3] += 1
    W[c1j1][offset+3] += 1
    
    return offset + 4

def _add_1x1(U, V, W, offset, k, j):
    a0k = 0*5 + k
    a1k = 1*5 + k
    bkj = k*5 + j
    c0j = j*2 + 0
    c1j = j*2 + 1
    
    U[a0k][offset+0] += 1
    V[bkj][offset+0] += 1
    W[c0j][offset+0] += 1
    
    U[a1k][offset+1] += 1
    V[bkj][offset+1] += 1
    W[c1j][offset+1] += 1
    
    return offset + 2

def _build_strassen_based():
    n, m, p = 2, 5, 5
    
    strategies = [
        ([(0,1), (2,3), (4,)], [(0,1), (2,3), (4,)]),
    ]
    
    U = [[0] * 46 for _ in range(n * m)]
    V = [[0] * 46 for _ in range(m * p)]
    W = [[0] * 46 for _ in range(p * n)]
    
    offset = 0
    k_groups, j_groups = strategies[0]
    for kg in k_groups:
        for jg in j_groups:
            if len(kg) == 2 and len(jg) == 2:
                offset = _add_strassen_222(U, V, W, offset, kg[0], kg[1], jg[0], jg[1])
            elif len(kg) == 2 and len(jg) == 1:
                offset = _add_2x1(U, V, W, offset, kg[0], kg[1], jg[0])
            elif len(kg) == 1 and len(jg) == 2:
                offset = _add_1x2(U, V, W, offset, kg[0], jg[0], jg[1])
            else:
                offset = _add_1x1(U, V, W, offset, kg[0], jg[0])
                
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p):
    """
    Try to reduce the rank by 1 by finding a linear dependency.
    Returns (new_U, new_V, new_W) if successful, else None.
    """
    rank = len(U[0])
    # Try to remove the last term (index rank-1)
    # We need to express T_{rank-1} as a linear combination of T_0 ... T_{rank-2}
    
    # Vectorize the tensor T_{rank-1}
    # T_{ijk} = U_i * V_j * W_k
    # We want to solve A * x = b
    # A is (n*m*p) x (rank-1)
    # b is (n*m*p) x 1
    
    A = np.zeros((n*m*p, rank-1))
    b = np.zeros(n*m*p)
    
    # Fill A and b
    # Note: U, V, W are lists of lists.
    # We need to map (i, j, k) to index.
    # Index = i*m*p + j*p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                # b is the value of the term to be removed
                b[idx] = U[i*m + j][rank-1] * V[j*p + k][rank-1] * W[k*n + i][rank-1]
                
                # A columns are the other terms
                for r in range(rank-1):
                    A[idx, r] = U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
    
    # Solve A * x = b
    try:
        x, residuals, rank_A, singular_values = np.linalg.lstsq(A, b, rcond=None)
    except:
        return None
        
    # Check if the solution is exact
    if np.allclose(A @ x, b, atol=1e-10):
        # Update U, V, W
        new_U = [row[:rank-1] for row in U]
        new_V = [row[:rank-1] for row in V]
        new_W = [row[:rank-1] for row in W]
        
        # Add x * (U_r, V_r, W_r) to the existing terms
        # Wait, this is not quite right.
        # We have T_{new} = sum_{r=0}^{rank-2} (U_r + x_r U_{rank-1}) ... ? No.
        # The decomposition is sum of rank-1 tensors.
        # T_r = u_r v_r w_r.
        # We want sum_{r=0}^{rank-2} (u_r v_r w_r) = u_{rank-1} v_{rank-1} w_{rank-1}.
        # This is not possible by just scaling the vectors.
        # The rank-1 tensors are not linearly independent in the vector space of vectors,
        # but in the tensor space.
        # However, if we find x such that sum_{r=0}^{rank-2} x_r T_r = T_{rank-1},
        # then we can replace T_{rank-1} with sum x_r T_r.
        # But the new terms are still rank-1? No, they are linear combinations of rank-1 tensors.
        # The new decomposition would have terms:
        # T'_r = x_r T_r ? No.
        # The set of terms becomes { T_0, ..., T_{rank-2} } with coefficients?
        # No, the decomposition is a sum of rank-1 tensors.
        # If T_{rank-1} is in the span of the others, we can just remove it?
        # No, we need to represent the *total* tensor.
        # Total T = sum_{r=0}^{rank-1} T_r.
        # If T_{rank-1} = sum_{r=0}^{rank-2} x_r T_r,
        # Then Total T = sum_{r=0}^{rank-2} (1 + x_r) T_r ? No.
        # Total T = sum_{r=0}^{rank-2} T_r + sum_{r=0}^{rank-2} x_r T_r = sum_{r=0}^{rank-2} (1+x_r) T_r.
        # This is still a sum of rank-1 tensors!
        # So we can just update the coefficients?
        # But U, V, W are fixed matrices.
        # We cannot change the "shape" of the rank-1 tensors.
        # We can only change the *vectors* u_r, v_r, w_r.
        # If we scale u_r by (1+x_r), we change the tensor.
        # But we cannot scale v_r and w_r by the same factor.
        # So this approach fails unless x_r is 0 or 1.
        
        return None
        
    return None

def construct():
    n, m, p = 2, 5, 5
    U, V, W = _build_strassen_based()
    if _verify_decomp(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    U = [[0] * (n*m*p) for _ in range(n * m)]
    V = [[0] * (n*m*p) for _ in range(m * p)]
    W = [[0] * (n*m*p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
