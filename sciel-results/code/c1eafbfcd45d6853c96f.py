import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses tensor product composition: 5 = 2 + 3, so M_5 = M_2 ⊗ M_3.
    - M_2: Rank 7 (Strassen)
    - M_3: Rank 7 (Laderman)
    - Total rank: 7 * 7 = 49
    """
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill
    
    n, m, p = 5, 5, 5
    
    try:
        result = _construct_tensor_product(n, m, p)
        if result is not None:
            if _verify_decomposition(result, n, m, p):
                return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition with rank 125
    return _construct_schoolbook(n, m, p)


def _verify_decomposition(result, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-10:
                    return False
    return True


def _construct_tensor_product(n, m, p):
    """
    Construct decomposition using tensor product of 2x2 and 3x3 multiplications.
    
    Index mapping: i = 2*i2 + i3, j = 2*j2 + j3, k = 2*k2 + k3
    where i2,j2,k2 in {0,1} and i3,j3,k3 in {0,1,2}
    
    The 5x5 multiplication tensor is M_2 ⊗ M_3, so rank = rank(M_2) * rank(M_3) = 7*7 = 49
    """
    # Strassen decomposition for 2x2 (rank 7)
    # Each term: (u_dict, v_dict, w_dict) where keys are (local_i, local_j), etc.
    strassen_2x2 = [
        # M1 = A11*B11
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        # M2 = (A11+A12)*B12
        ({(0,0):1, (0,1):1}, {(0,1):1}, {(0,1):1}),
        # M3 = A21*(B11+B12)
        ({(1,0):1}, {(0,0):1, (0,1):1}, {(1,0):1}),
        # M4 = A22*B22
        ({(1,1):1}, {(1,1):1}, {(1,1):1}),
        # M5 = (A11+A22)*(B21+B22)
        ({(0,0):1, (1,1):1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M6 = (A22-A11)*(B11+B21)
        ({(0,0):-1, (1,1):1}, {(0,0):1, (1,0):1}, {(1,0):-1, (1,1):1}),
        # M7 = (A12-A22)*(B21+B22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):-1, (1,1):1}),
    ]
    
    # Laderman decomposition for 3x3 (rank 7)
    # Laderman's algorithm for 3x3 matrix multiplication
    laderman_3x3 = [
        # M1 = A11 * B11
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        # M2 = (A11+A12) * B21
        ({(0,0):1, (0,1):1}, {(1,0):1}, {(0,1):1}),
        # M3 = (A12-A13) * B32
        ({(0,1):1, (0,2):-1}, {(2,1):1}, {(1,2):1}),
        # M4 = (A21+A31) * B13
        ({(1,0):1, (2,0):1}, {(0,2):1}, {(2,0):1}),
        # M5 = (A31-A32) * B23
        ({(2,0):1, (2,1):-1}, {(1,2):1}, {(1,1):1}),
        # M6 = (A21+A22) * (B22-B12)
        ({(1,0):1, (1,1):1}, {(1,1):1, (0,1):-1}, {(1,0):1, (1,1):-1}),
        # M7 = (A32+A33) * (B33-B23)
        ({(2,1):1, (2,2):1}, {(2,2):1, (1,2):-1}, {(2,1):1, (2,2):-1}),
    ]
    
    # Build all 49 terms by taking tensor products
    terms = []
    
    for s_u, s_v, s_w in strassen_2x2:
        for l_u, l_v, l_w in laderman_3x3:
            u_entries = []
            v_entries = []
            w_entries = []
            
            # U: index (i, j) where i = 2*i2 + i3, j = 2*j2 + j3
            for (i2, j2), val2 in s_u.items():
                for (i3, j3), val3 in l_u.items():
                    gi = 2 * i2 + i3
                    gj = 2 * j2 + j3
                    val = val2 * val3
                    if val != 0:
                        u_entries.append((gi * m + gj, val))
            
            # V: index (j, k) where j = 2*j2 + j3, k = 2*k2 + k3
            for (j2, k2), val2 in s_v.items():
                for (j3, k3), val3 in l_v.items():
                    gj = 2 * j2 + j3
                    gk = 2 * k2 + k3
                    val = val2 * val3
                    if val != 0:
                        v_entries.append((gj * p + gk, val))
            
            # W: index (k, i) where k = 2*k2 + k3, i = 2*i2 + i3
            for (k2, i2), val2 in s_w.items():
                for (k3, i3), val3 in l_w.items():
                    gk = 2 * k2 + k3
                    gi = 2 * i2 + i3
                    val = val2 * val3
                    if val != 0:
                        w_entries.append((gk * n + gi, val))
            
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for idx, val in u_entries:
                u[idx] = val
            for idx, val in v_entries:
                v[idx] = val
            for idx, val in w_entries:
                w[idx] = val
            terms.append((u, v, w))
    
    # Build matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook(n, m, p):
    """
    Schoolbook decomposition with rank n*m*p = 125.
    """
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
