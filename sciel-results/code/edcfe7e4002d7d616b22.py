"""Program for matrix multiplication 2x2 by 2x13 with rank-minimal decomposition."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Trivial decomposition: rank = n*m*p = 52
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
                r += 1

    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid exact decomposition of matmul tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T4 = T.reshape(n, m, m, p, p, n)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1 if (j == jp and i == ip and k == kp) else 0
                                if abs(T4[i, j, jp, k, kp, ip] - expected) > 1e-6:
                                    return False
        return True

    def try_structured_rank28():
        """
        Try to construct a rank-28 decomposition for 2x2 * 2x13.
        
        Strategy: For each column k of B (k=0..12), we need to compute:
          C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
          C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        
        We use 2 terms per column (total 26) plus 2 global correction terms.
        
        For column k, use:
          Term 1: U[0,k]=1, U[2,k]=1; V[k,k]=1, V[13+k,k]=1; W[k,k]=1, W[13+k,k]=1
          Term 2: U[1,k+13]=1, U[3,k+13]=1; V[k,k+13]=-1, V[13+k,k+13]=1; W[k,k+13]=1, W[13+k,k+13]=1
        
        Then add 2 correction terms at indices 26 and 27.
        
        Let's verify this works.
        """
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # For each column k (0..12):
        for k in range(p):
            # Term k: contributes to C[0,k] and C[1,k] via A[0,0]*B[0,k] and A[1,0]*B[0,k]
            U[0, k] = 1      # A[0,0]
            U[2, k] = 1      # A[1,0]
            V[k, k] = 1      # B[0,k]
            V[13 + k, k] = 1 # B[1,k]
            W[k, k] = 1      # C[0,k]
            W[13 + k, k] = 1 # C[1,k]
            
            # Term k+13: contributes via A[0,1]*B[1,k] and A[1,1]*B[1,k]
            U[1, k + 13] = 1   # A[0,1]
            U[3, k + 13] = 1   # A[1,1]
            V[k, k + 13] = -1  # -B[0,k] (correction)
            V[13 + k, k + 13] = 1  # B[1,k]
            W[k, k + 13] = 1   # C[0,k]
            W[13 + k, k + 13] = 1  # C[1,k]
        
        # Correction terms 26 and 27
        # We need to fix the -B[0,k] terms that were introduced
        # Term 26: add back A[0,0]*B[0,k] for all k? No, that's not right.
        # 
        # Let me reconsider. The above construction gives:
        # C[0,k] = A[0,0]*B[0,k] + A[1,0]*B[0,k] - A[0,1]*B[0,k] + A[0,1]*B[1,k]
        #        = (A[0,0] + A[1,0] - A[0,1])*B[0,k] + A[0,1]*B[1,k]
        # This is NOT correct.
        
        # Let me try a different approach.
        # 
        # For each column k, use the identity:
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        #
        # Rank-3 per column would be 39 total. We want 28.
        #
        # Let me try: for each pair of columns, use a shared structure.
        #
        # Actually, let me just return None and let the random search handle it.
        return None

    best_rank = rank_trivial
    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}

    # Try structured rank-28 first
    if time.time() < deadline - 5:
        result_28 = try_structured_rank28()
        if result_28 is not None:
            U, V, W = result_28
            if check_decomposition(U, V, W):
                best_rank = 28
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # If rank 28 not found, try random search for ranks 29, 30, ..., 35
    if best_rank > 28:
        for target_rank in range(35, 27, -1):
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(30):
                if time.time() > deadline:
                    break
                
                np.random.seed(attempt * 1000 + target_rank * 7)
                U = np.random.randint(-1, 2, size=(n * m, target_rank))
                V = np.random.randint(-1, 2, size=(m * p, target_rank))
                W = np.random.randint(-1, 2, size=(p * n, target_rank))
                
                if check_decomposition(U, V, W):
                    found = True
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            
            if found:
                break

    return best_result
# EVOLVE-BLOCK-END
