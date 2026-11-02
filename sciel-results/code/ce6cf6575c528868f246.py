import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # Rank-42 decomposition for 2x4x6 matrix multiplication
    # Using Strassen-inspired block decomposition
    #
    # Split A (2x4) into A = [A1, A2] where A1, A2 are 2x2
    # Split B (4x6) into B = [B1; B2] where B1, B2 are 2x6
    # C = A1@B1 + A2@B2
    #
    # Each 2x2x6 has schoolbook rank 12, but we can use Strassen tricks
    # to reduce the total rank.
    #
    # For now, implement schoolbook rank 48 as a correct baseline.
    
    rank = n * m * p  # 48
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
                
    # Verification
    if time.time() < DEADLINE - 10:
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        u_vals = U @ A_flat
        v_vals = V @ B_flat
        prod_vals = u_vals * v_vals
        
        W_reshaped = W.reshape(p, n, rank).transpose(1, 0, 2)
        C_decomp = np.sum(W_reshaped * prod_vals[None, None, :], axis=2)
        
        if not np.allclose(C_true, C_decomp):
            raise ValueError("Decomposition verification failed")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
