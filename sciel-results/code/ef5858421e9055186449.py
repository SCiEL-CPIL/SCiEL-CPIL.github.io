import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def sanitize_matrix(M):
        """
        Ensures all entries are exactly integers or half-integers.
        Multiplies by 2, rounds to nearest integer, divides by 2.
        """
        M_2 = M * 2.0
        M_rounded = np.round(M_2)
        M_clean = M_rounded / 2.0
        return M_clean

    def verify(U, V, W, tol=1e-9):
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    u_idx = i*2 + j
                    v_idx = j*9 + k
                    w_idx = k*2 + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def build_rank32():
        """
        Build a rank-32 decomposition.
        Uses Strassen (rank 7) for 4 blocks of 2 columns each (columns 0-1, 2-3, 4-5, 6-7)
        and schoolbook (rank 4) for the last column (column 8).
        Total Rank = 7*4 + 4 = 32.
        """
        terms = []
        
        def add_strassen_block(k0, k1):
            # Standard Strassen decomposition for M_{2,2,2}
            # U indices: 0,1,2,3 (i*2+j)
            # V indices: j*2+k_local (0..3) -> mapped to global j*9 + k
            # W indices: k_local*2+i (0..3) -> mapped to global k*2 + i
            
            # Term 1
            U1 = {0: 1, 3: 1}
            V1 = {k0: 1, 9+k1: 1} # j=0,k0 and j=1,k1
            W1 = {2*k0: 1, 2*k1+1: 1} # k0,i=0 and k1,i=1
            terms.append((U1, V1, W1))
            
            # Term 2
            U2 = {2: 1, 3: 1}
            V2 = {k0: 1} # j=0,k0
            W2 = {2*k0+1: 1, 2*k1+1: -1} # k0,i=1 and k1,i=1
            terms.append((U2, V2, W2))
            
            # Term 3
            U3 = {0: 1}
            V3 = {k1: 1, 9+k1: -1} # j=0,k1 and j=1,k1
            W3 = {2*k1: 1, 2*k1+1: 1} # k1,i=0 and k1,i=1
            terms.append((U3, V3, W3))
            
            # Term 4
            U4 = {3: 1}
            V4 = {9+k0: 1, k0: -1} # j=1,k0 and j=0,k0
            W4 = {2*k0: 1, 2*k0+1: 1} # k0,i=0 and k0,i=1
            terms.append((U4, V4, W4))
            
            # Term 5
            U5 = {0: 1, 1: 1}
            V5 = {9+k1: 1} # j=1,k1
            W5 = {2*k0: -1, 2*k1: 1} # k0,i=0 and k1,i=0
            terms.append((U5, V5, W5))
            
            # Term 6
            U6 = {2: 1, 0: -1}
            V6 = {k0: 1, k1: 1} # j=0,k0 and j=0,k1
            W6 = {2*k1+1: 1} # k1,i=1
            terms.append((U6, V6, W6))
            
            # Term 7
            U7 = {1: 1, 3: -1}
            V7 = {9+k0: 1, 9+k1: 1} # j=1,k0 and j=1,k1
            W7 = {2*k0: 1} # k0,i=0
            terms.append((U7, V7, W7))

        # Apply Strassen to column pairs (0,1), (2,3), (4,5), (6,7)
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            add_strassen_block(k0, k1)
        
        # Handle last column (k=8) with schoolbook
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))

        rank = len(terms)
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W

    # Try to find a better solution? 
    # The prompt suggests trying to reduce rank below 32.
    # However, the current best known is 32. 
    # Let's stick to the robust rank-32 solution to ensure validity.
    # Any attempt to go lower (e.g. 31) requires a valid rank-10 for 2x2x3, which is complex and error-prone to derive on the fly.
    # The "inspiration" programs also settle on rank 32 or fail to verify lower ranks.
    
    U, V, W = build_rank32()
    
    # Sanitize to ensure integer/half-integer constraint is met exactly
    U = sanitize_matrix(U)
    V = sanitize_matrix(V)
    W = sanitize_matrix(W)
    
    # Final verification
    if not verify(U, V, W):
        raise ValueError("Internal verification failed for rank 32 solution")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
