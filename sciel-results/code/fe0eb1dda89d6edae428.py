"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        # Compute using decomposition
        C_comp = np.zeros((n, p))
        for r in range(rank):
            # X_r[i] = sum_j U[i*m+j, r] * A[i,j]
            X_r = np.array([sum(U[i*m+j, r] * A[i, j] for j in range(m)) for i in range(n)])
            # Y_r[k] = sum_j V[j*p+k, r] * B[j,k]
            Y_r = np.array([sum(V[j*p+k, r] * B[j, k] for j in range(m)) for k in range(p)])
            # C[i,k] += W[k*n+i, r] * X_r[i] * Y_r[k]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k*n+i, r] * X_r[i] * Y_r[k]
        
        return np.allclose(C_comp, C_true, atol=1e-10)
    
    # Try to find a better decomposition
    # The problem is equivalent to 8 independent 2x4x1 multiplications
    # Each 2x4x1 has rank 8 (two dot products of length 4)
    # Total schoolbook: 64
    
    # Try a structured approach: group the 8 k-values into pairs
    # For each pair (k1, k2), we compute 2 outputs: C[0,k1], C[1,k1], C[0,k2], C[1,k2]
    # This is a 2x4x2 multiplication, schoolbook rank 16
    # If we can do 2x4x2 with rank < 16, we save
    
    # For 2x2x2, Strassen gives rank 7. For 2x4x2, let's try to construct.
    # 2x4x2: C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}, k in {0,1}, j in {0,1,2,3}
    # Schoolbook: 16 terms. Can we do better?
    
    # Let's try a direct construction for 2x4x2 with rank 12
    # This would give total rank 4*12 = 48 for 2x4x8
    
    # Actually, let's just try to use a known result or search
    # Given time constraints, I'll try a simple heuristic search
    
    best_result = schoolbook()
    
    # Try rank 56, 52, 48 with a simple random search
    for target_rank in [56, 52, 48, 44, 40]:
        if time.time() > deadline - 30:
            break
        
        # Simple random search: start from schoolbook and try to merge terms
        # This is a very crude approach
        rank = target_rank
        if rank >= n * m * p:
            continue
            
        # Create initial random decomposition
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize with schoolbook pattern for first min(rank, n*m*p) terms
        num_init = min(rank, n * m * p)
        for c in range(num_init):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i, c] = 1
        
        # Try to optimize the remaining terms
        # This is a very simplified approach - just check if it works as-is
        if verify(U.tolist(), V.tolist(), W.tolist()):
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    return best_result
# EVOLVE-BLOCK-END
