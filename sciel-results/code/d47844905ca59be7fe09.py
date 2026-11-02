import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    def verify(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U_mat[a_idx, :], V_mat[b_idx, :] * W_mat[c_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
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

    def schoolbook():
        rank = n * m * p
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

    # Build rank-25: three Strassen blocks (7 each) + 4 schoolbook terms for last column
    def build_rank25():
        rank = 25
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def add_strassen_block(col_start, k_start):
            # Term 0: (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1])
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_start+1) + 1, col_start+0] += 1

            # Term 1: (A[1,0]+A[1,1])*(B[0,k0])
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_start+1) + 1, col_start+1] -= 1

            # Term 2: A[0,0]*(B[0,k1] - B[1,k1])
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1
            V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1
            W[2*(k_start+1) + 1, col_start+2] += 1

            # Term 3: A[1,1]*(B[1,k0] - B[0,k0])
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            # Term 4: (A[0,0]+A[0,1])*(B[1,k1])
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_start+1) + 0, col_start+4] += 1

            # Term 5: (A[1,0]-A[0,0])*(B[0,k0]+B[0,k1])
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1

            # Term 6: (A[0,1]-A[1,1])*(B[1,k0]+B[1,k1])
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

    # Structural change: Try a rank-24 decomposition using a "merged" last block.
    # Instead of Strassen(4,5) + Schoolbook(6), we attempt a custom rank-10 decomposition
    # for the 2x2x3 sub-problem (cols 4,5,6).
    # If this fails verification, we fall back to the known valid rank-25.
    
    def build_rank24_attempt():
        rank = 24
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def add_strassen_block(col_start, k_start):
            # Standard Strassen for cols k_start, k_start+1
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_start+1) + 1, col_start+0] += 1

            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_start+1) + 1, col_start+1] -= 1

            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1
            V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1
            W[2*(k_start+1) + 1, col_start+2] += 1

            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_start+1) + 0, col_start+4] += 1

            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1

            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1
            V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        # Strassen for cols (0,1) and (2,3)
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)

        # Custom Rank-10 block for cols (4,5,6)
        # We need to decompose T_{2x2x3} in rank 10.
        # This is a known open problem if rank < 10, but let's try a specific construction.
        # Actually, 2x2x3 has rank 7? No, 2x2x2 is 7. 2x2x3 is at least 7+? 
        # Lower bound for 2x2x3 is 7 (from 2x2x2) + 3? No.
        # Let's just try to fit 10 terms.
        
        col_start = 14
        # Terms 0-9 for cols 4,5,6
        
        # We will use a heuristic: take the Strassen for (4,5) and modify/merge with col 6.
        # Or just use 4 schoolbook terms for col 6 and 6 terms for (4,5)? 
        # 6 terms for 2x2x2 is impossible (min rank 7).
        # So we must use 7 for (4,5) and 3 for col 6? 
        # Can we do col 6 in 3 terms? 
        # C0,6 = A00 B0,6 + A01 B1,6
        # C1,6 = A10 B0,6 + A11 B1,6
        # This requires 4 terms in schoolbook. Can we do it in 3?
        # (A00+A11)(B0,6+B1,6) -> C0,6 + C1,6 + cross terms.
        # (A00-A11)(B0,6-B1,6) -> C0,6 + C1,6 - cross terms.
        # Sum: 2(C0,6+C1,6). Diff: 2(A00 B0,6 - A01 B1,6 + A11 B1,6 - A10 B0,6).
        # This doesn't isolate C0,6 and C1,6 easily without more terms.
        # So 3 terms for col 6 is likely impossible.
        
        # Therefore, a simple block diagonal 7+7+10 is unlikely to work if 10 isn't enough for 2x2x3.
        # What is the rank of 2x2x3? It is 7. (Strassen applies to 2x2x2, but 2x2x3 is just 2x2x2 + 2x2x1.
        # Rank(2x2x1) = 2. So Rank(2x2x3) <= 7+2 = 9? 
        # Wait, Strassen is for 2x2x2. 
        # If we use Strassen for cols 4,5 (rank 7), and schoolbook for col 6 (rank 4), total 11.
        # If we can do col 6 in rank 3, total 10.
        # If we can do col 6 in rank 2, total 9.
        
        # Let's try to construct a rank-9 block for cols 4,5,6.
        # 7 (Strassen for 4,5) + 2 (for col 6)?
        # Can we do 2x2x1 in rank 2?
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # This is rank 2 if we allow linear combinations?
        # T = A00 B0 e0 + A01 B1 e0 + A10 B0 e1 + A11 B1 e1
        # = B0 (A00 e0 + A10 e1) + B1 (A01 e0 + A11 e1)
        # Yes! Rank 2.
        # So 7 + 2 = 9 terms for cols 4,5,6.
        # Total rank: 7 (cols 0,1) + 7 (cols 2,3) + 9 (cols 4,5,6) = 23.
        
        # Let's implement this Rank-23 attempt!
        
        # Terms 0-6: Strassen for cols 4,5
        # We reuse the Strassen logic but offset.
        
        # Term 0: (A[0,0]+A[1,1])*(B[0,4]+B[1,5])
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + 4, col_start+0] += 1
        V[7*1 + 5, col_start+0] += 1
        W[2*4 + 0, col_start+0] += 1
        W[2*5 + 1, col_start+0] += 1

        # Term 1: (A[1,0]+A[1,1])*(B[0,4])
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + 4, col_start+1] += 1
        W[2*4 + 1, col_start+1] += 1
        W[2*5 + 1, col_start+1] -= 1

        # Term 2: A[0,0]*(B[0,5] - B[1,5])
        U[0, col_start+2] += 1
        V[7*0 + 5, col_start+2] += 1
        V[7*1 + 5, col_start+2] -= 1
        W[2*5 + 0, col_start+2] += 1
        W[2*5 + 1, col_start+2] += 1

        # Term 3: A[1,1]*(B[1,4] - B[0,4])
        U[3, col_start+3] += 1
        V[7*1 + 4, col_start+3] += 1
        V[7*0 + 4, col_start+3] -= 1
        W[2*4 + 0, col_start+3] += 1
        W[2*4 + 1, col_start+3] += 1

        # Term 4: (A[0,0]+A[0,1])*(B[1,5])
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + 5, col_start+4] += 1
        W[2*4 + 0, col_start+4] -= 1
        W[2*5 + 0, col_start+4] += 1

        # Term 5: (A[1,0]-A[0,0])*(B[0,4]+B[0,5])
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + 4, col_start+5] += 1
        V[7*0 + 5, col_start+5] += 1
        W[2*5 + 1, col_start+5] += 1

        # Term 6: (A[0,1]-A[1,1])*(B[1,4]+B[1,5])
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + 4, col_start+6] += 1
        V[7*1 + 5, col_start+6] += 1
        W[2*4 + 0, col_start+6] += 1

        # Terms 7-8: Rank-2 decomposition for col 6
        # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
        # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
        
        # Term 7: B[0,6] * (A[0,0]*e0 + A[1,0]*e1)
        # U[0,7] = 1, U[2,7] = 1
        # V[0,7] = 1 (B[0,6])
        # W[6*2+0, 7] = 1 (C[0,6])
        # W[6*2+1, 7] = 1 (C[1,6])
        U[0, col_start+7] += 1
        U[2, col_start+7] += 1
        V[7*0 + 6, col_start+7] += 1
        W[2*6 + 0, col_start+7] += 1
        W[2*6 + 1, col_start+7] += 1

        # Term 8: B[1,6] * (A[0,1]*e0 + A[1,1]*e1)
        # U[1,8] = 1, U[3,8] = 1
        # V[1,8] = 1 (B[1,6])
        # W[6*2+0, 8] = 1 (C[0,6])
        # W[6*2+1, 8] = 1 (C[1,6])
        U[1, col_start+8] += 1
        U[3, col_start+8] += 1
        V[7*1 + 6, col_start+8] += 1
        W[2*6 + 0, col_start+8] += 1
        W[2*6 + 1, col_start+8] += 1

        return U, V, W

    # Try rank-23 first
    if time.time() < deadline:
        U, V, W = build_rank24_attempt() # Named 24 but actually 23 if valid
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank-25
    U, V, W = build_rank25()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
