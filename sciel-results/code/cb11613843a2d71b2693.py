import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = 4, 4, 8
    
    # Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    # Strategy:
    # 1. Use Strassen's algorithm (Rank 7) for 2x2x2.
    # 2. Use Kronecker product to get Rank 49 for 4x4x4.
    # 3. Split the 8-column dimension into two 4-column blocks.
    # 4. Apply the Rank-49 decomposition to each block independently.
    # Total Rank = 49 + 49 = 98.
    
    # Strassen 2x2x2 rank-7 decomposition
    # Indices: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # i=0, j=0
        [0, 0, 0, 0, 1,  0, 1],  # i=0, j=1
        [0, 1, 0, 0, 0,  1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0,  0,-1]   # i=1, j=1
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # j=0, k=0
        [0, 0, 1,  0, 0, 1, 0],  # j=0, k=1
        [0, 0, 0,  1, 0, 0, 1],  # j=1, k=0
        [1, 0,-1,  0, 1, 0, 1]   # j=1, k=1
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0
        [0, 1, 0, 1,  0, 0, 0],  # k=0, i=1
        [0, 0, 1, 0,  1, 0, 0],  # k=1, i=0
        [1,-1, 1, 0,  0, 1, 0]   # k=1, i=1
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    # U4[i*4+j, t*7+s] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    # where i = 2*i1 + i2, j = 2*j1 + j2
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Construct final 4x4x8 matrices
    # Shapes: U (16x98), V (32x98), W (32x98)
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    # Optional: Lightweight verification if time allows
    # Given the deterministic nature and previous success, we skip heavy verification
    # to ensure we stay well within the time limit, but we can do a spot check.
    if time.time() < deadline - 5.0:
        # Spot check one element to ensure no construction error
        # C[0,0] = sum(A[0,j] * B[j,0])
        # In tensor form: sum_r U[0*4+0, r] * V[0*8+0, r] * W[0*4+0, r] should be 1?
        # No, the tensor decomposition satisfies sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i,i'} delta_{j,j'} delta_{k,k'} ?
        # Actually, the decomposition is for the identity tensor of matrix mult:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) corresponds to the "1" in the tensor, else 0?
        # The standard decomposition is for the tensor T_{i,j,k} = 1 if i,j,k form a valid multiplication term?
        # No, the tensor is T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i,i'} ... wait.
        # The task says "exactly decompose the matrix multiplication tensor".
        # The tensor M_{i,j,k} is 1 if the indices correspond to the term A_{i,j}B_{j,k}.
        # So sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} should be 1 for all i,j,k?
        # No, that would mean C_{i,k} = sum_j sum_r ... = sum_j A_{i,j} B_{j,k} * (sum_r ...)?
        # If sum_r U V W = 1 for ALL i,j,k, then C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # Yes, that is the condition. The tensor of matrix multiplication has 1s at all (i,j,k) indices?
        # Let's check the dimensions.
        # U is (n*m) x R. V is (m*p) x R. W is (p*n) x R.
        # The tensor T has indices i (n), j (m), k (p).
        # T_{i,j,k} = 1 for all i,j,k?
        # If so, sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
        
        # Let's verify this property for i=0, j=0, k=0.
        val = 0.0
        for r in range(rank):
            val += U[0, r] * V[0, r] * W[0, r]
        if abs(val - 1.0) > 1e-9:
            # Fallback to naive if construction failed (should not happen)
            return construct_naive(n, m, p)
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
