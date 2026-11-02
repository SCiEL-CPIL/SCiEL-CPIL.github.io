import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 2, 10

    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    # Vectors are length-4 (row-major 2x2 matrices).
    # U, V, W are the factor vectors for the 7 terms.
    strassen_terms = [
        # (U_vec, V_vec, W_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    rank = 35
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    # We have p=10 columns in B. We split them into p//2 = 5 blocks of 2 columns.
    # For each block k (0..4), we apply the Strassen decomposition.
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            # Fill U (A part): same for all blocks, A is 2x2
            # A indices: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
            U[0, col] = a_vec[0]
            U[1, col] = a_vec[1]
            U[2, col] = a_vec[2]
            U[3, col] = a_vec[3]
            
            # Fill V (B part): B is 2x10.
            # Block k uses columns 2k and 2k+1.
            # B indices: row 0: 0, 1, ..., 9. row 1: 10, 11, ..., 19.
            # V index for B[i, j] is i*p + j.
            # b_vec corresponds to a 2x2 matrix [[b00, b01], [b10, b11]]
            # b00 -> B[0, 2k] -> index 0*p + 2k
            # b01 -> B[0, 2k+1] -> index 0*p + 2k+1
            # b10 -> B[1, 2k] -> index 1*p + 2k
            # b11 -> B[1, 2k+1] -> index 1*p + 2k+1
            
            V[0 * p + 2 * k, col] = b_vec[0]
            V[0 * p + 2 * k + 1, col] = b_vec[1]
            V[1 * p + 2 * k, col] = b_vec[2]
            V[1 * p + 2 * k + 1, col] = b_vec[3]
            
            # Fill W (C part): C is 2x10.
            # C indices follow trace convention W[k*n+i, r] ? 
            # Task says: "output factor follows the official trace convention W[k*n+i,r]"
            # Usually C[i, j] is index i*p + j in row-major.
            # Let's check the standard mapping for the tensor T[a,b,c].
            # T[i*m + k, k*p + j, j*n + i] = 1.
            # Here:
            # a_idx = i*m + k (A index)
            # b_idx = k*p + j (B index)
            # c_idx = j*n + i (C index)
            
            # So C[i, j] is at index j*n + i.
            # Block k produces C columns 2k and 2k+1.
            # c_vec corresponds to a 2x2 matrix [[c00, c01], [c10, c11]]
            # c00 -> C[0, 2k] -> index (2k)*n + 0
            # c01 -> C[0, 2k+1] -> index (2k+1)*n + 0
            # c10 -> C[1, 2k] -> index (2k)*n + 1
            # c11 -> C[1, 2k+1] -> index (2k+1)*n + 1
            
            W[2 * k * n + 0, col] = c_vec[0]
            W[(2 * k + 1) * n + 0, col] = c_vec[1]
            W[2 * k * n + 1, col] = c_vec[2]
            W[(2 * k + 1) * n + 1, col] = c_vec[3]
            
            col += 1

    # Verification
    # Build target tensor T
    # T[a,b,c] = 1 if a = i*m+k, b = k*p+j, c = j*n+i
    # a in [0, 3], b in [0, 19], c in [0, 19]
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Compute decomposition result
    # Result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We can do this efficiently with einsum or loops.
    # Loop is fine for small dimensions.
    Result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # Outer product of u, v, w
        Result += np.einsum('a,b,c->abc', u, v, w)

    # Check validity
    if not np.allclose(Result, T, atol=1e-6):
        # Fallback to naive rank 40 if Strassen fails (should not happen)
        rank = n * m * p # 40
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a_idx = i * m + k
                    b_idx = k * p + j
                    c_idx = j * n + i
                    U[a_idx, col] = 1.0
                    V[b_idx, col] = 1.0
                    W[c_idx, col] = 1.0
                    col += 1
        # Re-verify fallback (trivially correct)
        if not np.allclose(np.einsum('ar,br,cr->abc', U, V, W), T, atol=1e-6):
             raise Exception("Decomposition failed")

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
