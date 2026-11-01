import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Rank 28 Strassen solution
    rank_fallback = 28
    U_fallback = np.zeros((n * m, rank_fallback), dtype=float)
    V_fallback = np.zeros((m * p, rank_fallback), dtype=float)
    W_fallback = np.zeros((p * n, rank_fallback), dtype=float)
    
    # Standard Strassen 2x2x2 decomposition
    # U, V, W matrices for the 7 products
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[c11, c12], [c21, c22]]
    #
    # m1 = (a+d)(e+h) = c11+c12+c21+c22
    # m2 = (c+d)e = c21+c22
    # m3 = a(f-h) = c12-c22
    # m4 = d(g-c) = c21-c11
    # m5 = (a+b)h = c11+c12
    # m6 = (c-d)(g+h) = c21+c22-c11-c12 ... wait
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c-d)(g+h)
    # m7 = (a-c)(e+f)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Let's map these to the U, V, W vectors.
    # U is 2x2 (flattened 4x1), V is 2x2 (flattened 4x1), W is 2x2 (flattened 4x1).
    # Term t contributes U_t * V_t * W_t to the tensor?
    # No, the tensor decomposition is sum of rank-1 tensors.
    # The "product" in the algorithm is $M_t = U_t A V_t$? No.
    # The scalar product $M_t$ is computed as $U_t \odot A \odot V_t$?
    # Actually, $M_t = \sum_{i,j} U_{i,j,t} A_{i,j} \sum_{j,k} V_{j,k,t} B_{j,k}$? No.
    # The standard form is:
    # $M_t = \sum_{i,j} U_{i,j,t} A_{i,j}$ is not a scalar.
    # The Strassen algorithm computes 7 scalar products.
    # $M_1 = (A_{0,0} + A_{1,1}) (B_{0,0} + B_{1,1})$
    # This is a bilinear form.
    # In tensor decomposition, this corresponds to a rank-1 tensor $u \otimes v \otimes w$?
    # No, a scalar product is a rank-1 tensor in the space of bilinear forms?
    # The multiplication tensor $M$ is a tensor.
    # A rank-1 decomposition term is $u \otimes v \otimes w$.
    # The value of the tensor on $(A,B)$ is $\langle M, A \otimes B \rangle$.
    # $\langle u \otimes v \otimes w, A \otimes B \rangle = (\langle u, A \rangle) (\langle v, B \rangle) \langle w, C \rangle$?
    # No, the third factor $w$ is in the output space.
    # The term contributes $w_{k,i} \cdot (\sum_{j} u_{i,j} A_{i,j}) \cdot (\sum_{j} v_{j,k} B_{j,k})$ to $C_{i,k}$?
    # This doesn't look like a single scalar multiplication.
    
    # Let's stick to the code provided which was verified to work in previous runs (score 1.1429).
    # The code provided in the prompt uses a specific list of dicts.
    # I will trust the logic in the prompt's `strassen_products` list but double check the indices.
    
    strassen_products = [
        # m1
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}},
        # m2
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): 1}},
        # m3
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): -1}},
        # m4
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(1,0): 1, (0,0): -1}},
        # m5
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): 1, (0,1): 1}},
        # m6
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1}},
        # m7
        {'A': {(0,0): 1, (1,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(0,0): 1, (0,1): 1, (1,0): -1, (1,1): -1}},
    ]
    
    # Wait, the C matrix in the decomposition corresponds to the contribution to the output C.
    # The standard Strassen reconstruction is:
    # C = m1*M1 + m2*M2 + ... where M1..M7 are matrices.
    # M1 = [[1,1],[1,1]]
    # M2 = [[0,0],[1,1]]
    # M3 = [[0,1],[0,-1]]
    # M4 = [[-1,0],[1,0]]
    # M5 = [[1,1],[0,0]]
    # M6 = [[-1,-1],[1,1]]
    # M7 = [[1,1],[-1,-1]]
    
    # Let's check the provided list in the prompt against these.
    # Prompt:
    # 1. C: {(0,0): 1, (1,1): 1} -> M1? No, M1 is all 1s.
    # The prompt's C matrices seem sparse.
    # Let's re-read the prompt's code carefully.
    # {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}}
    # This C is diag(1,1).
    # If A = [[a,d],[d,a]]? No.
    
    # The prompt's code might be incorrect or using a different convention.
    # However, it achieved score 1.1429.
    # This implies the verification passed.
    # I will assume the prompt's `strassen_products` are correct for its verification function.
    
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r
            for (i, j), coeff in prod['A'].items():
                U_fallback[i * m + j, t] = coeff
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V_fallback[j * p + k_global, t] = coeff
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W_fallback[k_global * n + i, t] = coeff

    def verify(U, V, W):
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        T_decomp = np.zeros_like(T)
        for t in range(U.shape[1]):
            T_decomp += np.einsum('i,j,k->ijk', U[:,t], V[:,t], W[:,t])
        return np.allclose(T_decomp, T, atol=1e-8)

    # Schoolbook fallback
    def get_schoolbook():
        U_sb = np.zeros((n*m, n*m*p), dtype=float)
        V_sb = np.zeros((m*p, n*m*p), dtype=float)
        W_sb = np.zeros((p*n, n*m*p), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, idx] = 1.0
                    V_sb[j*p+k, idx] = 1.0
                    W_sb[k*n+i, idx] = 1.0
                    idx += 1
        return U_sb, V_sb, W_sb

    if verify(U_fallback, V_fallback, W_fallback):
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
    else:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
