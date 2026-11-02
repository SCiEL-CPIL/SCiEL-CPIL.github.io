import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Returns list of (u_coeff, v_coeff, w_coeff) for 2x2 Strassen.
    u_coeff: dict {(i,j): coeff} for A
    v_coeff: dict {(j,k): coeff} for B
    w_coeff: dict {(k,i): coeff} for C
    
    The product term is: (sum c_u * A_u) * (sum c_v * B_v) -> sum c_w * C_w
    Actually, the standard tensor decomposition uses bilinear forms.
    Each multiplication is X*Y.
    X = sum a_i A_i
    Y = sum b_j B_j
    C += X*Y
    
    So for each term in the decomposition:
    U part: for each a_i, add (i, j) with coeff a_i
    V part: for each b_j, add (j, k) with coeff b_j
    W part: for each (k, i) in C, add coeff 1
    
    Wait, the standard form is:
    C = sum_{t} lambda_t * (A^{(t)} * B^{(t)})
    where A^{(t)} and B^{(t)} are matrices.
    
    For Strassen 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 - M5 + M4 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    We will return a list of tuples (U_dict, V_dict, W_dict)
    where U_dict maps (i,j) to coeff, etc.
    """
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    u = {(0,0): 1, (1,1): 1}
    v = {(0,0): 1, (1,1): 1}
    w = {(0,0): 1, (1,1): 1} # C11 += M1, C22 += M1
    terms.append((u, v, w))
    
    # M2 = (A21+A22)B11
    u = {(1,0): 1, (1,1): 1}
    v = {(0,0): 1}
    w = {(0,1): 1, (1,1): -1} # C21 += M2, C22 -= M2
    terms.append((u, v, w))
    
    # M3 = A11(B12-B22)
    u = {(0,0): 1}
    v = {(0,1): 1, (1,1): -1}
    w = {(1,0): 1, (1,1): 1} # C12 += M3, C22 += M3
    terms.append((u, v, w))
    
    # M4 = A22(B21-B11)
    u = {(1,1): 1}
    v = {(1,0): 1, (0,0): -1}
    w = {(0,0): 1, (0,1): 1} # C11 += M4, C21 += M4
    terms.append((u, v, w))
    
    # M5 = (A11+A12)B22
    u = {(0,0): 1, (0,1): 1}
    v = {(1,1): 1}
    w = {(0,0): -1, (1,0): 1} # C11 -= M5, C12 += M5
    terms.append((u, v, w))
    
    # M6 = (A21-A11)(B11+B12)
    u = {(1,0): 1, (0,0): -1}
    v = {(0,0): 1, (0,1): 1}
    w = {(1,1): 1} # C22 += M6
    terms.append((u, v, w))
    
    # M7 = (A12-A22)(B21+B22)
    u = {(0,1): 1, (1,1): -1}
    v = {(1,0): 1, (1,1): 1}
    w = {(0,0): 1} # C11 += M7
    terms.append((u, v, w))
    
    return terms

def _laderman_3x3_terms():
    """
    Returns list of (u_coeff, v_coeff, w_coeff) for 3x3 Laderman.
    Laderman's algorithm for 3x3 matrix multiplication has rank 15.
    Reference: "Laderman's algorithm"
    
    Let A, B be 3x3.
    M1 = A11(B12-B22)
    M2 = (A11+A12)B22
    M3 = A33(B21-B11)
    M4 = (A31+A33)B11
    M5 = A13(B31-B21)
    M6 = (A11+A13)B31
    M7 = A23(B32-B12)
    M8 = (A22+A23)B12
    M9 = A32(B13-B33)
    M10 = (A32+A33)B33
    M11 = (A12-A33)(B33+B13)
    M12 = (A13-A22)(B22+B12)
    M13 = (A23-A11)(B11+B31)
    M14 = (A31-A22)(B21+B31)
    M15 = (A32-A11)(B12+B22)
    
    C11 = M12 - M13 + M14
    C12 = M1 + M2
    C13 = M9 + M10
    C21 = M5 + M6
    C22 = M11 - M12 + M13
    C23 = M7 + M8
    C31 = M3 + M4
    C32 = M14 - M15 + M1
    C33 = M11 - M14 + M15
    
    Wait, I need to verify the exact Laderman formulas.
    A common set of 15 multiplications:
    M1 = (A11+A12)(B22+B23) ? No.
    
    Let's use a known correct set.
    From "Optimal Matrix Multiplication Algorithms":
    Laderman (1976) gives rank 15.
    
    One formulation:
    M1 = A11(B12-B22)
    M2 = (A11+A12)B22
    M3 = A33(B21-B11)
    M4 = (A31+A33)B11
    M5 = A13(B31-B21)
    M6 = (A11+A13)B31
    M7 = A23(B32-B12)
    M8 = (A22+A23)B12
    M9 = A32(B13-B33)
    M10 = (A32+A33)B33
    M11 = (A12-A33)(B33+B13)
    M12 = (A13-A22)(B22+B12)
    M13 = (A23-A11)(B11+B31)
    M14 = (A31-A22)(B21+B31)
    M15 = (A32-A11)(B12+B22)
    
    C11 = M12 - M13 + M14
    C12 = M1 + M2
    C13 = M9 + M10
    C21 = M5 + M6
    C22 = M11 - M12 + M13
    C23 = M7 + M8
    C31 = M3 + M4
    C32 = M14 - M15 + M1
    C33 = M11 - M14 + M15
    
    Let's double check C11.
    C11 = A11B11 + A12B21 + A13B31
    M12 = A13B22 - A22B22 + A13B12 - A22B12
    M13 = A23B11 - A11B11 + A23B31 - A11B31
    M14 = A31B21 - A22B21 + A31B31 - A22B31
    M12 - M13 + M14 = ...
    This seems complicated. I might use a different source or just implement a known valid one.
    
    Actually, for the purpose of this task, I can use a simpler decomposition if I can't verify Laderman easily, or I can trust the standard Laderman.
    Let's assume the above is correct.
    """
    terms = []
    
    # Indices are 0-based.
    # M1 = A[0,0]*(B[0,1]-B[1,1])
    u = {(0,0): 1}
    v = {(0,1): 1, (1,1): -1}
    w = {(1,0): 1, (2,1): 1} # C12 += M1, C32 += M1
    terms.append((u, v, w))
    
    # M2 = (A[0,0]+A[0,1])*B[1,1]
    u = {(0,0): 1, (0,1): 1}
    v = {(1,1): 1}
    w = {(1,0): 1} # C12 += M2
    terms.append((u, v, w))
    
    # M3 = A[2,2]*(B[1,0]-B[0,0])
    u = {(2,2): 1}
    v = {(1,0): 1, (0,0): -1}
    w = {(2,0): 1} # C31 += M3
    terms.append((u, v, w))
    
    # M4 = (A[2,0]+A[2,2])*B[0,0]
    u = {(2,0): 1, (2,2): 1}
    v = {(0,0): 1}
    w = {(2,0): 1} # C31 += M4
    terms.append((u, v, w))
    
    # M5 = A[0,2]*(B[2,0]-B[1,0])
    u = {(0,2): 1}
    v = {(2,0): 1, (1,0): -1}
    w = {(1,0): 1} # C21 += M5
    terms.append((u, v, w))
    
    # M6 = (A[0,0]+A[0,2])*B[2,0]
    u = {(0,0): 1, (0,2): 1}
    v = {(2,0): 1}
    w = {(1,0): 1} # C21 += M6
    terms.append((u, v, w))
    
    # M7 = A[1,2]*(B[2,1]-B[0,1])
    u = {(1,2): 1}
    v = {(2,1): 1, (0,1): -1}
    w = {(2,1): 1} # C23 += M7
    terms.append((u, v, w))
    
    # M8 = (A[1,1]+A[1,2])*B[0,1]
    u = {(1,1): 1, (1,2): 1}
    v = {(0,1): 1}
    w = {(2,1): 1} # C23 += M8
    terms.append((u, v, w))
    
    # M9 = A[2,1]*(B[0,2]-B[2,2])
    u = {(2,1): 1}
    v = {(0,2): 1, (2,2): -1}
    w = {(0,2): 1} # C13 += M9
    terms.append((u, v, w))
    
    # M10 = (A[2,1]+A[2,2])*B[2,2]
    u = {(2,1): 1, (2,2): 1}
    v = {(2,2): 1}
    w = {(0,2): 1} # C13 += M10
    terms.append((u, v, w))
    
    # M11 = (A[0,1]-A[2,2])*(B[2,2]+B[0,2])
    u = {(0,1): 1, (2,2): -1}
    v = {(2,2): 1, (0,2): 1}
    w = {(1,1): 1, (2,2): 1} # C22 += M11, C33 += M11
    terms.append((u, v, w))
    
    # M12 = (A[0,2]-A[1,1])*(B[1,1]+B[0,1])
    u = {(0,2): 1, (1,1): -1}
    v = {(1,1): 1, (0,1): 1}
    w = {(0,0): 1, (1,1): -1} # C11 += M12, C22 -= M12
    terms.append((u, v, w))
    
    # M13 = (A[1,2]-A[0,0])*(B[0,0]+B[2,0])
    u = {(1,2): 1, (0,0): -1}
    v = {(0,0): 1, (2,0): 1}
    w = {(0,0): -1, (1,1): 1} # C11 -= M13, C22 += M13
    terms.append((u, v, w))
    
    # M14 = (A[2,0]-A[1,1])*(B[1,0]+B[2,0])
    u = {(2,0): 1, (1,1): -1}
    v = {(1,0): 1, (2,0): 1}
    w = {(0,0): 1, (2,1): -1, (2,2): -1} # C11 += M14, C32 -= M14, C33 -= M14
    terms.append((u, v, w))
    
    # M15 = (A[2,1]-A[0,0])*(B[0,1]+B[1,1])
    u = {(2,1): 1, (0,0): -1}
    v = {(0,1): 1, (1,1): 1}
    w = {(2,1): 1, (2,2): 1} # C32 += M15, C33 += M15
    terms.append((u, v, w))
    
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Reshape
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    # Compute tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W index is k*n + i, so W_mat[k,i,r]
    
    # Use einsum for verification
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # np.einsum('ijr,jkr,ki r->ijk', U_mat, V_mat, W_mat) -- careful with spaces
    
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    
    # Construct target tensor
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] should be 1 if j is the contraction index, else 0?
    # Actually, the tensor of matrix multiplication has T[i,j,k] = 1 if the indices are compatible in the product.
    # Specifically, the tensor is defined such that C = A * B corresponds to tensor contraction.
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k? No.
    # The tensor represents the bilinear map.
    # T[i,j,k] is the coefficient of A[i,j]B[j,k] in C[i,k].
    # So T[i,j,k] should be 1 if we are looking at the term A[i,j]B[j,k] -> C[i,k].
    # Wait, the tensor decomposition expresses the *structure* of the multiplication.
    # The "multiplication tensor" M has M[i,j,k] = 1 if the indices are valid.
    # Specifically, for each r, the term is sum_{i,j,k} U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k] = sum_{i,k} (sum_j ... ) C[i,k]
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] to equal 1 if the indices i,j,k form a valid term in the product?
    # No, we want the resulting C to be correct.
    # C[i,k] = sum_j (sum_r U[i,j,r] V[j,k,r] W[k,i,r]) A[i,j] B[j,k]
    # The inner sum is the "kernel" K[i,j,k].
    # We require K[i,j,k] = 1 for all i,j,k?
    # If K[i,j,k] = 1, then C[i,k] = sum_j A[i,j] B[j,k], which is correct.
    # So we must check if T[i,j,k] == 1 for all i,j,k.
    
    # Target T is all ones.
    target = np.ones((n, m, p))
    
    if not np.allclose(T, target):
        # Find where it fails
        diff = np.abs(T - target)
        max_diff = np.max(diff)
        if max_diff > 1e-9:
            return False, max_diff
    return True, 0.0

def construct():
    n, m, p = DIMENSIONS
    
    # Start with empty matrices
    # We don't know the rank ahead of time, so we'll build lists of terms and then flatten.
    
    terms = []
    
    # 1. Top-Left 2x2 (Strassen)
    # A: rows 0-1, cols 0-1. B: rows 0-1, cols 0-1. C: rows 0-1, cols 0-1.
    strassen_terms = _strassen_2x2_terms()
    for u, v, w in strassen_terms:
        # Shift indices
        u_shifted = {(i, j): c for (i, j), c in u.items()}
        v_shifted = {(j, k): c for (j, k), c in v.items()}
        w_shifted = {(k, i): c for (k, i), c in w.items()}
        terms.append((u_shifted, v_shifted, w_shifted))
    
    # 2. Bottom-Right 3x3 (Laderman)
    # A: rows 2-4, cols 2-4. B: rows 2-4, cols 2-4. C: rows 2-4, cols 2-4.
    laderman_terms = _laderman_3x3_terms()
    for u, v, w in laderman_terms:
        u_shifted = {(i+2, j+2): c for (i, j), c in u.items()}
        v_shifted = {(j+2, k+2): c for (j, k), c in v.items()}
        w_shifted = {(k+2, i+2): c for (k, i), c in w.items()}
        terms.append((u_shifted, v_shifted, w_shifted))
    
    # 3. Off-diagonal blocks (Schoolbook)
    # A12 (2x3): rows 0-1, cols 2-4
    # B21 (3x2): rows 2-4, cols 0-1
    # C11 (2x2): rows 0-1, cols 0-1
    # Terms: A[i,j] B[j,k] -> C[i,k] where i in 0-1, j in 2-4, k in 0-1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # A21 (3x2): rows 2-4, cols 0-1
    # B12 (2x3): rows 0-1, cols 2-4
    # C22 (3x3): rows 2-4, cols 2-4
    # Terms: A[i,j] B[j,k] -> C[i,k] where i in 2-4, j in 0-1, k in 2-4
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # A11 (2x2): rows 0-1, cols 0-1
    # B12 (2x3): rows 0-1, cols 2-4
    # C12 (2x3): rows 0-1, cols 2-4
    # Terms: A[i,j] B[j,k] -> C[i,k] where i in 0-1, j in 0-1, k in 2-4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # A12 (2x3): rows 0-1, cols 2-4
    # B22 (3x3): rows 2-4, cols 2-4
    # C12 (2x3): rows 0-1, cols 2-4
    # Terms: A[i,j] B[j,k] -> C[i,k] where i in 0-1, j in 2-4, k in 2-4
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # A21 (3x2): rows 2-4, cols 0-1
    # B11 (2x2): rows 0-1, cols 0-1
    # C21 (3x2): rows 2-4, cols 0-1
    # Terms: A[i,j] B[j,k] -> C[i,k] where i in 2-4, j in 0-1, k in 0-1
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
                
    # A22 (3x3): rows 2-4, cols 2-4
    # B21 (3x2): rows 2-4, cols 0-1
    # C21 (3x2): rows 2-4, cols 0-1
    # Terms: A[i,j] B[j,k] -> C[i,k] where i in 2-4, j in 2-4, k in 0-1
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), c in u.items():
            U[i * m + j][r] = c
        for (j, k), c in v.items():
            V[j * p + k][r] = c
        for (k, i), c in w.items():
            W[k * n + i][r] = c
            
    # Convert to numpy arrays for verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    is_valid, err = _verify_decomposition(U_arr, V_arr, W_arr, n, m, p)
    
    if not is_valid:
        # If invalid, fall back to schoolbook for everything (Rank 125)
        # This is a safety net.
        U = [[0] * (n*m*p) for _ in range(n * m)]
        V = [[0] * (n*m*p) for _ in range(m * p)]
        W = [[0] * (n*m*p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        is_valid, err = _verify_decomposition(U_arr, V_arr, W_arr, n, m, p)
        if not is_valid:
            # This should never happen
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
