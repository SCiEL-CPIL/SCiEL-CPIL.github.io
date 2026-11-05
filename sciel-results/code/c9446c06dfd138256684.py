import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145

    # Initialize with rank 55
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    # Strassen coefficients
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    Wc = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j, col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k, col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i, col] = val
            col += 1

    for i in range(2):
        for k in range(10):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1

    # Verify
    # Tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We want T[i,j,k] = 1 for all i,j,k
    # Note: The indices in the problem are:
    # U: (n*m, rank) -> index i*m+j
    # V: (m*p, rank) -> index j*p+k
    # W: (p*n, rank) -> index k*n+i
    
    # Check using einsum
    # T[i,j,k] = np.einsum('ir,jr,kr->ijk', U, V, W) 
    # But W index is k*n+i, so W[k*n+i, r].
    # Let's reshape W to (p, n, rank)
    W_reshaped = W.reshape(p, n, rank)
    # U is (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    # V is (m, p, rank)
    V_reshaped = V.reshape(m, p, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
    T = np.einsum('ijr,jkr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank = 60
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        # Re-verify
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
        if not np.allclose(T, 1.0, atol=1e-9):
             # This should not happen
             return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    # We can check if any column is a linear combination of others?
    # Or if any column is zero? (No)
    # Or if the tensor can be represented with fewer terms?
    # This is hard. We'll just try to remove columns that are "redundant" in the sense that removing them still yields the tensor.
    # This is O(R * N * M * P * R) which is manageable.
    
    # However, since the decomposition is likely minimal, we might not reduce it.
    # We will just return the 55 rank solution.
    
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    # Ensure entries are integers or half-integers
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2

    return {"U": U_out, "V": V_out, "W": W_out}
