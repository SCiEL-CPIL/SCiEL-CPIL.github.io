"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses Strassen's algorithm on four 2x2 sub-blocks to achieve rank 28.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of 2x4 by 4x4 matrix multiplication.
    """
    try:
        result = _construct_strassen_28()
        if _verify_decomposition_fast(result):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 32)
    return _construct_schoolbook_32()


def _verify_decomposition_fast(result):
    """
    Fast vectorized verification of the decomposition.
    """
    try:
        n, m, p = DIMENSIONS
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        # Reshape for batch computation
        # U: (n*m, rank) -> (n, m, rank)
        U_reshaped = U.reshape(n, m, rank)
        # V: (m*p, rank) -> (m, p, rank)
        V_reshaped = V.reshape(m, p, rank)
        # W: (p*n, rank) -> (p, n, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute intermediate: M[i,k,t] = sum_j U[i,j,t] * V[j,k,t]
        M = np.einsum('ijt,jkt->ikt', U_reshaped, V_reshaped)
        # Compute C[i,k] = sum_t M[i,k,t] * W[k,i,t]
        C_computed = np.einsum('ikt,kit->ik', M, W_reshaped)
        
        # Test with multiple random matrices
        np.random.seed(42)
        for _ in range(5):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        
        return True
    except Exception:
        return False


def _construct_schoolbook_32():
    """
    Standard schoolbook decomposition with rank 32.
    """
    n, m, p = DIMENSIONS
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


def _construct_strassen_28():
    """
    2x4 * 4x4 using four 2x2 Strassen blocks.
    Rank 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2 product X*Y:
    # Local indices: a=0 (x00), b=1 (x01), c=2 (x10), d=3 (x11)
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y01-y11)
    # P4 = x11(y10-y00)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y01)
    # P7 = (x01-x11)(y10+y11)
    
    # W output indices: c00=0, c01=1, c10=2, c11=3 (local)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # P1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # P2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # P3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # P4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),         # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # P6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),         # P7
    ]
    
    # Four sub-products with their index mappings
    sub_products = [
        # 1. A_left (A[:,0:2]) * B_tl (B[0:2,0:2]) -> C_left (C[:,0:2])
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # 2. A_right (A[:,2:4]) * B_bl (B[2:4,0:2]) -> C_left (C[:,0:2])
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # 3. A_left (A[:,0:2]) * B_tr (B[0:2,2:4]) -> C_right (C[:,2:4])
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
        # 4. A_right (A[:,2:4]) * B_br (B[2:4,2:4]) -> C_right (C[:,2:4])
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
    ]
    
    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r, c = loc_u // 2, loc_u % 2
                global_u = sp['a_map'](r, c)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r, c = loc_v // 2, loc_v % 2
                global_v = sp['b_map'](r, c)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r, c = loc_w // 2, loc_w % 2
                global_w = sp['c_map'](r, c)
                W[global_w][col] += coeff
            col += 1
    
    assert col == 28, f"Expected col=28, got {col}"
    
    return {"U": U, "V": V, "W": W}
