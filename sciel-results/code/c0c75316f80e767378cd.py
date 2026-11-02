import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    # Verify a decomposition is correct
    def verify(U_mat, V_mat, W_mat):
        # Check matching entries (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U_mat[a_idx, :], V_mat[b_idx, :] * W_mat[c_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check non-matching entries (should be 0)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for j2 in range(m):
                        if j2 != j:
                            a_idx = i * m + j
                            b_idx = j2 * p + k
                            c_idx = k * n + i
                            val = np.dot(U_mat[a_idx, :], V_mat[b_idx, :] * W_mat[c_idx, :])
                            if abs(val) > 1e-9:
                                return False
        return True

    # Trivial rank-28 schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p  # 28
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
        return U, V, W

    # Build the rank-25 solution using Strassen blocks for column pairs and direct terms for the last column
    def build_rank25():
        rank = 25
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        def add_strassen_block(col_start, k_start):
            # Term 0: (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1]) -> C[0,k0], C[1,k1]
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_start+1) + 1, col_start+0] += 1

            # Term 1: (A[1,0]+A[1,1])*(B[0,k0]) -> C[1,k0], C[1,k1]
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_start+1) + 1, col_start+1] -= 1

            # Term 2: A[0,0]*(B[0,k1] - B[1,k1]) -> C[0,k1], C[1,k1]
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1
            V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1
            W[2*(k_start+1) + 1, col_start+2] += 1

            # Term 3: A[1,1]*(B[1,k0] - B[0,k0]) -> C[0,k0], C[1,k0]
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            # Term 4: (A[0,0]+A[0,1])*(B[1,k1]) -> -C[0,k0], C[0,k1]
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_start+1) + 0, col_start+4] += 1

            # Term 5: (A[1,0]-A[0,0])*(B[0,k0]+B[0,k1]) -> C[1,k1]
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1

            # Term 6: (A[0,1]-A[1,1])*(B[1,k0]+B[1,k1]) -> C[0,k0]
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1
            V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        # Apply Strassen blocks for column pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)

        # Handle column 6 with 4 separate terms
        col_start = 21
        k = 6
        U[0, col_start+0] += 1
        V[7*0 + k, col_start+0] += 1
        W[2*k + 0, col_start+0] += 1

        U[1, col_start+1] += 1
        V[7*1 + k, col_start+1] += 1
        W[2*k + 0, col_start+1] += 1

        U[2, col_start+2] += 1
        V[7*0 + k, col_start+2] += 1
        W[2*k + 1, col_start+2] += 1

        U[3, col_start+3] += 1
        V[7*1 + k, col_start+3] += 1
        W[2*k + 1, col_start+3] += 1

        return U, V, W
    
    # Build and verify rank-25
    U, V, W = build_rank25()
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
