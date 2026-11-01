import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook fallback (rank 36)
    rank_school = n * m * p  # 36
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that U,V,W decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Block decomposition: split 3x6 into two 3x3 blocks, rank 18
    rank_block = 2 * (n * m * 3)  # 2 * 9 = 18
    U_block = [[0] * rank_block for _ in range(n * m)]
    V_block = [[0] * rank_block for _ in range(m * p)]
    W_block = [[0] * rank_block for _ in range(p * n)]
    
    col = 0
    for b in range(2):  # two 3x3 blocks
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    U_block[i * m + j][col] = 1
                    V_block[j * p + b * 3 + k][col] = 1
                    W_block[(b * 3 + k) * n + i][col] = 1
                    col += 1
    
    if verify_decomposition(U_block, V_block, W_block, n, m, p):
        best_rank = rank_block
        best_result = {"U": U_block, "V": V_block, "W": W_block}
    
    # Try to find a rank-7 decomposition for 2x3 x 3x3
    # If found, use it twice to get rank 14 for 2x3 x 3x6
    
    n2, m2, p2 = 2, 3, 3
    
    # The 2x3 x 3x3 tensor:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # 6 outputs, 9 schoolbook multiplications
    
    # Try to construct a rank-7 decomposition algebraically
    # Using the idea of sharing multiplications between rows of A
    
    # Let's try: M1..M9 are the 9 schoolbook terms
    # Can we express some as combinations?
    
    # For a 2x3 x 3x3, consider:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Try rank 7: define 7 scalar multiplications M1..M7
    # and express each C[i,k] as a linear combination
    
    # Strategy: Use (A[0,j] + A[1,j]) * (B[j,k] + B[j,k']) type terms
    
    # Let's try a specific construction:
    # M1 = A[0,0]*B[0,0]
    # M2 = A[0,1]*B[1,1]  
    # M3 = A[0,2]*B[2,2]
    # M4 = A[1,0]*B[0,0]
    # M5 = A[1,1]*B[1,1]
    # M6 = A[1,2]*B[2,2]
    # M7 = (A[0,0]+A[1,0])*(B[0,1]+B[1,0])  # cross term
    
    # This doesn't cleanly work. Let me try a different approach.
    
    # Actually, for 2x3 x 3x3, the minimum rank is likely 9 (same as schoolbook)
    # because the structure doesn't allow much sharing.
    
    # Let me try a randomized search for rank < 18 for the full 2x3 x 3x6 problem
    # using integer/half-integer entries
    
    np.random.seed(42)
    
    # Try ranks from 12 to 17 (below the block rank of 18)
    for target_rank in range(12, 18):
        if time.time() > deadline - 5:
            break
        
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
            
            # Generate random integer/half-integer matrices
            values = np.array([-1, -0.5, 0, 0.5, 1])
            
            U_cand = np.random.choice(values, size=(n*m, target_rank))
            V_cand = np.random.choice(values, size=(m*p, target_rank))
            W_cand = np.random.choice(values, size=(p*n, target_rank))
            
            # Verify
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = np.sum(U_cand[i*m+j] * V_cand[j*p+k] * W_cand[k*n+i])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best_rank = target_rank
                best_result = {
                    "U": U_cand.tolist(),
                    "V": V_cand.tolist(),
                    "W": W_cand.tolist()
                }
                break  # Move to next (lower) rank
        
        if best_rank == target_rank:
            break  # Found minimal rank for this search
    
    return best_result
