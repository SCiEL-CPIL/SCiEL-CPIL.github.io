import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 135.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_strassen_2x2_factors():
    """
    Returns U, V, W factors for 2x2 matrix multiplication using Strassen's algorithm.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indices for 2x2:
    0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A00 + A11)(B00 + B11)
    # A part: [1, 0, 0, 1]
    # B part: [1, 0, 0, 1]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += M1, C11 += M1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    # A part: [0, 0, 1, 1]
    # B part: [1, 0, 0, 0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 += M2, C11 -= M2
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    # A part: [1, 0, 0, 0]
    # B part: [0, 1, 0, -1]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C01 += M3, C11 += M3
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    # A part: [0, 0, 0, 1]
    # B part: [0, 0, 1, -1] -> B10 is index 2, B00 is index 0.
    # Wait, B10 is (1,0) -> index 2. B00 is (0,0) -> index 0.
    # M4 = A11(B10 - B00).
    # B part: [ -1, 0, 1, 0]
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    # C00 += M4, C10 += M4
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    # A part: [1, 1, 0, 0]
    # B part: [0, 0, 0, 1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 -= M5, C01 += M5
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # A part: [-1, 0, 1, 0]
    # B part: [1, 1, 0, 0]
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 += M6
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # A part: [0, 1, 0, -1]
    # B part: [0, 0, 1, 1]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 += M7
    W[0, 6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """
    Builds the 189-rank decomposition for 6x6 matrix multiplication
    using 3x3 blocks of 2x2 matrices with Strassen.
    """
    # 6x6 matrix -> 3x3 blocks of 2x2
    block_size = 2
    num_blocks = 3
    rank_per_block = 7
    total_rank = num_blocks * num_blocks * num_blocks * rank_per_block
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U2, V2, W2 = _build_strassen_2x2_factors()
    
    r_idx = 0
    for i in range(num_blocks):
        for k in range(num_blocks):
            for j in range(num_blocks):
                for s in range(rank_per_block):
                    # Map local 2x2 indices to global 6x6 indices
                    # A block (i, k)
                    # U index: row = 2*i + r, col = 2*k + c
                    # Global U index: (2*i + r)*6 + (2*k + c)
                    for r_local in range(2):
                        for c_local in range(2):
                            a_global = (2 * i + r_local) * 6 + (2 * k + c_local)
                            U[a_global, r_idx] = U2[r_local * 2 + c_local, s]
                            
                    # B block (k, j)
                    # V index: row = 2*k + r, col = 2*j + c
                    # Global V index: (2*k + r)*6 + (2*j + c)
                    for r_local in range(2):
                        for c_local in range(2):
                            b_global = (2 * k + r_local) * 6 + (2 * j + c_local)
                            V[b_global, r_idx] = V2[r_local * 2 + c_local, s]
                            
                    # C block (i, j)
                    # W index: col = 2*j + c, row = 2*i + r
                    # Global W index: (2*j + c)*6 + (2*i + r)
                    for r_local in range(2):
                        for c_local in range(2):
                            c_global = (2 * j + c_local) * 6 + (2 * i + r_local)
                            W[c_global, r_idx] = W2[r_local * 2 + c_local, s]
                            
                    r_idx += 1
                    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, num_tests=4, atol=1e-8):
    rank = U.shape[1]
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            C_computed += (Ur @ Vr) * Wr
        if not np.allclose(C_computed, C_expected, atol=atol):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Build the 189-rank decomposition
    result = _build_block_decomposition(n, m, p)
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, num_tests=5):
        return {
            "U": result["U"].tolist(),
            "V": result["V"].tolist(),
            "W": result["W"].tolist(),
        }
    else:
        # Fallback to naive if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist(),
        }

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    valid = _verify_decomposition(U, V, W, 6, 6, 6, num_tests=5)
    print(f"Valid: {valid}")
