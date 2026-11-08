import time
import numpy as np

DIMENSIONS = [3, 3, 6]
_LADERMAN_W_CACHE = None

def _get_laderman_w():
    global _LADERMAN_W_CACHE
    if _LADERMAN_W_CACHE is not None:
        return _LADERMAN_W_CACHE
    
    # Define Laderman's 23 products for 3x3x3
    # a_terms: list of (i, j, coeff)
    # b_terms: list of (j, k, coeff)
    products = []
    def add_product(a_terms, b_terms):
        products.append((a_terms, b_terms))

    add_product([(0,0,1)], [(0,0,1)]) # P1
    add_product([(0,1,1)], [(1,1,1)]) # P2
    add_product([(1,0,1)], [(0,0,1)]) # P3
    add_product([(1,1,1)], [(1,1,1)]) # P4
    add_product([(2,2,1)], [(2,2,1)]) # P5
    add_product([(0,0,1), (1,1,1), (2,2,-1)], [(2,2,1)]) # P6
    add_product([(0,0,1), (1,1,1), (2,2,-1)], [(0,1,1)]) # P7
    add_product([(0,1,1), (1,0,1)], [(0,1,1)]) # P8
    add_product([(0,1,1), (1,0,1)], [(1,0,1)]) # P9
    add_product([(0,0,1), (1,1,1), (2,2,1), (0,1,-1), (1,0,-1)], [(1,0,1)]) # P10
    add_product([(0,0,1), (1,1,1), (2,2,1), (0,1,1), (1,0,1)], [(0,0,1)]) # P11
    add_product([(0,0,1), (1,1,-1)], [(0,1,1)]) # P12
    add_product([(0,0,1), (1,1,-1)], [(1,0,1)]) # P13
    add_product([(0,1,1), (1,0,-1)], [(0,1,1)]) # P14
    add_product([(0,1,1), (1,0,-1)], [(1,0,1)]) # P15
    add_product([(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [(1,1,1)]) # P16
    add_product([(0,0,1), (0,1,1), (1,0,1), (1,1,1)], [(0,0,1)]) # P17
    add_product([(0,0,1), (1,0,1), (2,2,1)], [(2,2,1)]) # P18
    add_product([(0,1,1), (1,1,1), (2,2,1)], [(2,2,1)]) # P19
    add_product([(0,0,1), (1,0,1), (2,2,1)], [(0,1,1)]) # P20
    add_product([(0,1,1), (1,1,1), (2,2,1)], [(1,0,1)]) # P21
    add_product([(0,0,1), (0,1,1), (1,0,1), (1,1,1), (2,2,1)], [(0,0,1)]) # P22
    add_product([(0,0,1), (0,1,1), (1,0,1), (1,1,1), (2,2,1)], [(1,1,1)]) # P23

    # Build U_23 and V_23
    U_23 = np.zeros((9, 23), dtype=int)
    V_23 = np.zeros((9, 23), dtype=int)
    for l in range(23):
        a_terms, b_terms = products[l]
        for i, j, coeff in a_terms:
            U_23[i*3+j, l] += coeff
        for j, k, coeff in b_terms:
            V_23[j*3+k, l] += coeff

    # Solve for W_23
    # We need 23 samples to solve for 23 unknowns per (i,k)
    rng = np.random.default_rng(42)
    num_samples = 23
    A_samples = rng.integers(-10, 10, size=(num_samples, 3, 3))
    B_samples = rng.integers(-10, 10, size=(num_samples, 3, 3))
    
    P_samples = np.zeros((num_samples, 23))
    for s in range(num_samples):
        A_flat = A_samples[s].flatten()
        B_flat = B_samples[s].flatten()
        # P_l = (U_23[:, l] @ A_flat) * (V_23[:, l] @ B_flat)
        # Vectorize:
        # U_23 is (9, 23). A_flat is (9,).
        # ua = U_23.T @ A_flat -> (23,)
        ua = U_23.T @ A_flat
        vb = V_23.T @ B_flat
        P_samples[s, :] = ua * vb

    C_samples = np.matmul(A_samples, B_samples) # (23, 3, 3)
    
    W_23 = np.zeros((9, 23), dtype=int)
    for i in range(3):
        for k in range(3):
            c_vec = C_samples[:, i, k]
            # Solve P_samples.T @ W = c_vec
            # P_samples.T is (23, 23)
            # Use exact solve if possible
            try:
                W_sol = np.linalg.solve(P_samples.T, c_vec)
                # Check if close to integers
                W_int = np.round(W_sol).astype(int)
                # Verify
                if np.allclose(P_samples.T @ W_int, c_vec, atol=1e-9):
                    W_23[k*3+i, :] = W_int
                else:
                    # Fallback to lstsq if solve fails or is not integer
                    W_sol, _, _, _ = np.linalg.lstsq(P_samples.T, c_vec, rcond=None)
                    W_int = np.round(W_sol).astype(int)
                    if not np.allclose(P_samples.T @ W_int, c_vec, atol=1e-9):
                         raise ValueError("W solution not integer")
                    W_23[k*3+i, :] = W_int
            except Exception:
                raise ValueError("Failed to solve W")

    _LADERMAN_W_CACHE = (U_23, V_23, W_23)
    return _LADERMAN_W_CACHE

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def get_schoolbook():
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

    try:
        if time.time() - start_time > 100:
            return get_schoolbook()
            
        U_23, V_23, W_23 = _get_laderman_w()
        
        rank = 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 1: k in 0..2
        for l in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*3+j][l] = U_23[i*3+j, l]
            for j in range(3):
                for k in range(3):
                    V[j*3+k][l] = V_23[j*3+k, l]
            for k in range(3):
                for i in range(3):
                    W[k*3+i][l] = W_23[k*3+i, l]
                    
        # Block 2: k in 3..5
        for l in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*3+j][l+23] = U_23[i*3+j, l]
            for j in range(3):
                for k in range(3):
                    V[j*3+(k+3)][l+23] = V_23[j*3+k, l]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i][l+23] = W_23[k*3+i, l]
                    
        # Verify
        if time.time() - start_time > 110:
            return get_schoolbook()
            
        rng = np.random.default_rng(42)
        A_test = rng.integers(-5, 5, size=(3, 3))
        B_test = rng.integers(-5, 5, size=(3, 6))
        C_test = np.matmul(A_test, B_test)
        
        C_calc = np.zeros((3, 6))
        for r in range(rank):
            ua = 0
            for i in range(3):
                for j in range(3):
                    ua += U[i*3+j][r] * A_test[i,j]
                    
            vb = 0
            for j in range(3):
                for k in range(6):
                    vb += V[j*3+k][r] * B_test[j,k]
                    
            prod = ua * vb
            
            for k in range(6):
                for i in range(3):
                    C_calc[i,k] += prod * W[k*3+i][r]
                    
        if not np.allclose(C_test, C_calc, atol=1e-6):
            return get_schoolbook()
            
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        return get_schoolbook()
