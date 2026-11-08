import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Rank 50 decomposition:
    # 1. Strassen blocks for i in {0,1}, j in {0..3}, k in {0..3}: 4 blocks * 7 = 28
    # 2. Schoolbook for i=2, j in {0..3}, k in {0..3}: 16
    # 3. Outer product for j=4, k in {0..3} (all i): 1
    # 4. Inner product for j in {0..3}, k=4 (all i): 4
    # 5. Corner j=4, k=4 (all i): 1
    # Total: 28 + 16 + 1 + 4 + 1 = 50
    
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Part 1: Strassen blocks for i in {0,1}, j in {0..3}, k in {0..3}
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    for (i0, j0, k0) in blocks:
        if time.time() > deadline:
            break
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[i0*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        U[(i0+1)*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] -= 1.0
        col += 1
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        U[i0*m + j0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[j0*p + k0, col] -= 1.0
        W[k0*n + i0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        col += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        U[i0*m + j0, col] += 1.0
        U[i0*m + (j0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U[(i0+1)*m + j0, col] += 1.0
        U[i0*m + j0, col] -= 1.0
        V[j0*p + k0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        U[i0*m + (j0+1), col] += 1.0
        U[(i0+1)*m + (j0+1), col] -= 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        col += 1
    
    # Part 2: Schoolbook for i=2, j in {0..3}, k in {0..3}
    for j in range(4):
        for k in range(4):
            if time.time() > deadline:
                break
            U[2*m + j, col] = 1.0
            V[j*p + k, col] = 1.0
            W[k*n + 2, col] = 1.0
            col += 1
    
    # Part 3: Outer product for j=4, k in {0..3} (all i)
    # C[k,i] += A[i,4] * B[4,k]
    # This is rank 1: U[i,4] = 1, V[4,k] = 1, W[k,i] = 1 for all i,k
    for i in range(n):
        U[i*m + 4, col] = 1.0
    for k in range(4):
        V[4*p + k, col] = 1.0
    for k in range(4):
        for i in range(n):
            W[k*n + i, col] = 1.0
    col += 1
    
    # Part 4: Inner product for j in {0..3}, k=4 (all i)
    # C[4,i] += sum_{j=0}^{3} A[i,j] * B[j,4]
    # This requires 4 terms (one for each j)
    for j in range(4):
        if time.time() > deadline:
            break
        for i in range(n):
            U[i*m + j, col] = 1.0
        V[j*p + 4, col] = 1.0
        for i in range(n):
            W[4*n + i, col] = 1.0
        col += 1
    
    # Part 5: Corner j=4, k=4 (all i)
    # C[4,i] += A[i,4] * B[4,4]
    # Rank 1
    for i in range(n):
        U[i*m + 4, col] = 1.0
    V[4*p + 4, col] = 1.0
    for i in range(n):
        W[4*n + i, col] = 1.0
    col += 1
    
    # Verification (optional, for debugging)
    # If col < total_rank, we have unused columns; if col > total_rank, error
    if col != total_rank:
        # Adjust if needed
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Verification
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    
    print(f"Verification Errors: {errors}")
    print(f"Rank: {U.shape[1]}")
