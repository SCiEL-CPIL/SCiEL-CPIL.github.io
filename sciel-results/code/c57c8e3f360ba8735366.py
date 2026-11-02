import time
import numpy as np

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _construct_rank26(n, m, p):
    """
    Constructs a rank 26 decomposition for 3x3 matrix multiplication
    using Strassen's algorithm for the top-left 2x2 block.
    """
    if (n, m, p) != (3, 3, 3):
        return None

    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U[idx, col] += val
        for idx, val in v_vals.items():
            V[idx, col] += val
        for idx, val in w_vals.items():
            W[idx, col] += val
        col += 1

    # Indices for A (9x1 flattened 3x3)
    # A[i, j] -> i * 3 + j
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    a20, a21, a22 = 6, 7, 8
    
    # Indices for B
    b00, b01, b02 = 0, 1, 2
    b10, b11, b12 = 3, 4, 5
    b20, b21, b22 = 6, 7, 8
    
    # Indices for C
    c00, c01, c02 = 0, 1, 2
    c10, c11, c12 = 3, 4, 5
    c20, c21, c22 = 6, 7, 8

    # 1. Strassen terms for A11 * B11
    # A11: a00, a01, a10, a11
    # B11: b00, b01, b10, b11
    # C11: c00, c01, c10, c11

    # M1 = (a00 + a11) * (b00 + b11)
    # C00 += M1, C11 += M1
    add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
    
    # M2 = (a10 + a11) * b00
    # C10 += M2
    add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1})
    
    # M3 = a00 * (b10 + b11)
    # C01 += M3, C11 -= M3
    add_term({a00: 1}, {b10: 1, b11: 1}, {c01: 1, c11: -1})
    
    # M4 = a11 * (b01 - b11)
    # C00 += M4, C10 += M4
    add_term({a11: 1}, {b01: 1, b11: -1}, {c00: 1, c10: 1})
    
    # M5 = (a00 + a10) * b11
    # C01 += M5, C11 += M5
    add_term({a00: 1, a10: 1}, {b11: 1}, {c01: 1, c11: 1})
    
    # M6 = (a11 - a00) * (b00 + b01)
    # C00 -= M6, C11 += M6
    add_term({a11: 1, a00: -1}, {b00: 1, b01: 1}, {c00: -1, c11: 1})
    
    # M7 = (a01 - a11) * (b10 + b11)
    # C00 += M7
    add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1})

    # 2. Terms for A12 * B21 (contributes to C11 block? No, C11 is 2x2 top left. A12 is 2x1, B21 is 1x2. Product is 2x2. This is the OFF-diagonal block in the 2x2 partition of C? 
    # Wait, my partition was:
    # A = [A11 A12; A21 A22]
    # C = [C11 C12; C21 C22]
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # So A12 * B21 contributes to C11 (the top-left 2x2 block of C).
    # A12 indices: a02, a12
    # B21 indices: b20, b21
    # C11 indices: c00, c01, c10, c11
    
    # C00 += a02 * b20
    add_term({a02: 1}, {b20: 1}, {c00: 1})
    # C01 += a02 * b21
    add_term({a02: 1}, {b21: 1}, {c01: 1})
    # C10 += a12 * b20
    add_term({a12: 1}, {b20: 1}, {c10: 1})
    # C11 += a12 * b21
    add_term({a12: 1}, {b21: 1}, {c11: 1})

    # 3. Terms for A11 * B12 (contributes to C12: top-right 2x1 block of C)
    # A11 indices: a00, a01, a10, a11
    # B12 indices: b02, b12
    # C12 indices: c02, c12
    
    # C02 += a00*b02 + a01*b12
    # C12 += a10*b02 + a11*b12
    
    # Term for a00*b02
    add_term({a00: 1}, {b02: 1}, {c02: 1})
    # Term for a01*b12
    add_term({a01: 1}, {b12: 1}, {c02: 1})
    # Term for a10*b02
    add_term({a10: 1}, {b02: 1}, {c12: 1})
    # Term for a11*b12
    add_term({a11: 1}, {b12: 1}, {c12: 1})

    # 4. Terms for A12 * B22 (contributes to C12)
    # A12 indices: a02, a12
    # B22 index: b22
    # C12 indices: c02, c12
    
    # C02 += a02 * b22
    add_term({a02: 1}, {b22: 1}, {c02: 1})
    # C12 += a12 * b22
    add_term({a12: 1}, {b22: 1}, {c12: 1})

    # 5. Terms for A21 * B11 (contributes to C21: bottom-left 1x2 block of C)
    # A21 indices: a20, a21
    # B11 indices: b00, b01, b10, b11
    # C21 indices: c20, c21
    
    # C20 += a20*b00 + a21*b10
    # C21 += a20*b01 + a21*b11
    
    # Term for a20*b00
    add_term({a20: 1}, {b00: 1}, {c20: 1})
    # Term for a21*b10
    add_term({a21: 1}, {b10: 1}, {c20: 1})
    # Term for a20*b01
    add_term({a20: 1}, {b01: 1}, {c21: 1})
    # Term for a21*b11
    add_term({a21: 1}, {b11: 1}, {c21: 1})

    # 6. Terms for A22 * B21 (contributes to C21)
    # A22 index: a22
    # B21 indices: b20, b21
    # C21 indices: c20, c21
    
    # C20 += a22 * b20
    add_term({a22: 1}, {b20: 1}, {c20: 1})
    # C21 += a22 * b21
    add_term({a22: 1}, {b21: 1}, {c21: 1})

    # 7. Terms for A21 * B12 (contributes to C22: bottom-right 1x1 block of C)
    # A21 indices: a20, a21
    # B12 indices: b02, b12
    # C22 index: c22
    
    # C22 += a20*b02 + a21*b12
    
    # Term for a20*b02
    add_term({a20: 1}, {b02: 1}, {c22: 1})
    # Term for a21*b12
    add_term({a21: 1}, {b12: 1}, {c22: 1})

    # 8. Terms for A22 * B22 (contributes to C22)
    # A22 index: a22
    # B22 index: b22
    # C22 index: c22
    
    # C22 += a22 * b22
    add_term({a22: 1}, {b22: 1}, {c22: 1})

    return U, V, W

def _schoolbook(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    
    # Try rank 26
    result = _construct_rank26(n, m, p)
    if result is not None:
        U, V, W = result
        if _verify_fast(U, V, W, n, m, p, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
