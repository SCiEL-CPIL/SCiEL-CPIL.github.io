import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.ones((n, m, p))
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        return np.max(np.abs(T_computed - T_expected)) < tol
    
    def strassen_2x2_2():
        """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # P1 = (a+d)(e+h)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # P2 = (c+d)e
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1
        
        # P3 = a(f-h)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[2,2] = 1; W[3,2] = 1
        
        # P4 = d(g-e)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1
        
        # P5 = (a+b)h
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[2,4] = 1
        
        # P6 = (c-a)(e+f)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # P7 = (b-d)(g+h)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W
    
    def build_single_strassen():
        """
        Single Strassen 2x2x2 block at i∈{0,1}, j∈{0,1}, k∈{0,1}.
        Schoolbook for everything else.
        Rank = 7 + (24 - 4) = 7 + 20 = 27? No...
        
        Total entries: 3*3*8 = 72
        Strassen covers: 2*2*2 = 8 entries with 7 terms
        Remaining: 72 - 8 = 64 entries with 64 schoolbook terms
        Total rank: 7 + 64 = 71. That's worse!
        
        Wait, I need to think about this differently. The Strassen block handles
        a 2x2 sub-matrix multiplication. Let me reconsider.
        
        For 3x3 * 3x8:
        - A is 3x3, B is 3x8
        - Strassen on A[0:2,0:2] * B[0:2,0:2] gives C[0:2,0:2] with 7 terms
        - A[0:2,0:2] * B[0:2,2:8] -> 2*2*6 = 24 terms
        - A[0:2,2:3] * B[2:3,0:8] -> 2*1*8 = 16 terms
        - A[2:3,0:3] * B[0:3,0:8] -> 1*3*8 = 24 terms
        Total: 7 + 24 + 16 + 24 = 71. Still worse than 68.
        """
        pass
    
    def build_lacing_35():
        """
        Lacing technique: 
        - Use Strassen for 2x2x2 blocks on (i∈{0,1}, j∈{0,1}) for each k-pair
        - But share the i=2 row terms across k-pairs using a cross-term structure
        
        Actually, let's try a different lacing:
        Decompose the 3x3 A matrix as:
        A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
        
        Similarly B = [[B11, B12], [B21, B22]] where B11 is 2x4, B12 is 2x4, B21 is 1x4, B22 is 1x4
        
        C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
                   [A21*B11 + A22*B21, A21*B12 + A22*B22]]
        
        This doesn't directly help. Let me just use the 4-block Strassen approach
        but with a better local decomposition.
        
        For 3x3x2, can we do better than 17?
        Schoolbook: 18 terms
        Strassen on 2x2x2 sub-block: 7 terms + 10 schoolbook = 17 terms
        
        What if we use a 3x2x2 or 3x3x1 decomposition?
        
        Let me try: for 3x3x2, use Strassen on rows 0,1 of A and cols 0,1 of B (2x2x2),
        then handle row 2 of A and col 2 of B.
        
        Row 2 of A * B: 3*2 = 6 terms
        Rows 0,1 of A * col 2 of B: 2*2 = 4 terms
        Total: 7 + 6 + 4 = 17. Same.
        
        Alternative: use a different 2x2x2 sub-block. What if we use rows 0,1 and cols 1,2?
        Same count: 7 + 6 + 4 = 17.
        
        What about using Strassen on a 2x3x2 sub-block? Not standard.
        
        Let me try to find if 3x3x2 has rank < 17.
        Lower bound: max(9, 6, 6) = 9. Upper bound: 18 (schoolbook).
        Known: 3x3x2 rank is at most 17 (Strassen + schoolbook).
        
        I believe 17 is optimal for 3x3x2. So 4*17 = 68 is likely optimal for this block approach.
        
        Let me try a completely different approach: use the full 3x3x8 tensor structure.
        
        Actually, let me just implement the clean 4-block Strassen (rank 68) and return it.
        The current program already does this. Let me verify it works and optimize the code.
        """
        pass
    
    def build_4block_strassen():
        """
        4 independent 3x3x2 blocks, each with rank 17 (Strassen 2x2x2 + schoolbook).
        Total rank: 68.
        """
        U_block, V_block, W_block = _get_17_rank_block()
        rank_block = 17
        rank_total = 4 * rank_block
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U[:, col_start:col_end] = U_block
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
            
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def _get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part (7 terms) for the 2x2x2 sub-block (i=0,1; j=0,1; k=0,1)
        U_coeffs = [
            {0: 1, 3: 1},
            {2: 1, 3: 1},
            {0: 1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 0: -1},
            {1: 1, 3: -1}
        ]
        V_coeffs = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        W_coeffs = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local*3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local*2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_local, col] += coeff
            col += 1
        
        # Schoolbook part for remaining (i,j,k) in 3x3x2
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    if i in (0,1) and j in (0,1) and k in (0,1):
                        continue
                    U[i*3 + j, col] = 1.0
                    V[j*2 + k, col] = 1.0
                    W[k*3 + i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    # Build the 4-block Strassen decomposition (rank 68)
    result = build_4block_strassen()
    
    if validate(result["U"], result["V"], result["W"]):
        return result
    else:
        return schoolbook()
