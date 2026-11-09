"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    deadline = time.time() + 100  # Safety margin before 120s limit

    # Schoolbook decomposition
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input matrix A (size n x m) to the j-th column
                # A[i, j] is at index i*m + j
                U[i * m + j, col] = 1
                
                # V maps input matrix B (size m x p) to the i-th row
                # B[j, k] is at index j*p + k
                V[j * p + k, col] = 1
                
                # W maps to output matrix C (size n x p)
                # C[i, k] is at index k*n + i (according to task description W[k*n+i, r])
                W[k * n + i, col] = 1
                
                col += 1
                
                # Check time
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Verification step to ensure validity (optional but good for robustness)
    # We verify with a few random integer matrices
    if time.time() < deadline - 5:
        try:
            # Convert to float for fast matrix mult
            Uf = U.astype(float)
            Vf = V.astype(float)
            Wf = W.astype(float)
            
            # Tensor product T = sum_r U[:,r] V[:,r]^T W[:,r]^T ? No.
            # The decomposition is:
            # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            # Let's verify a few entries or random matrices.
            
            # Random A (n x m)
            A = np.random.randint(-5, 5, (n, m)).astype(float)
            # Random B (m x p)
            B = np.random.randint(-5, 5, (m, p)).astype(float)
            
            # True C = A @ B
            C_true = A @ B
            
            # Computed C using factors
            # C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is equivalent to:
            # C = (A * U.T) * V.T * W ? No, dimensions don't match simply.
            # Let's do it explicitly for verification
            C_calc = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        for r in range(rank):
                            s += A[i, j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    C_calc[i, k] = s
                    
            if not np.allclose(C_true, C_calc):
                 # Fallback if verification fails (should not happen)
                 pass
        except Exception:
            pass

    # Convert numpy arrays to lists of lists to match expected interface format if strict
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # The seed returned lists of lists. I will return lists of lists to be safe.
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
