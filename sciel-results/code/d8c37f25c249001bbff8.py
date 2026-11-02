import numpy as np
import time

DIMENSIONS = [3, 3, 8]
DEADLINE_SECONDS = 140  # Safety margin before 150s kill

def _validate_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rng = np.random.RandomState(42)
    for trial in range(20):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            A_part = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    A_part[i, j] = u_r[i * m + j]
            
            B_part = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    B_part[j, k] = v_r[j * p + k]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += w_r[k * n + i] * (A_part * B_part).sum()
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build a rank-68 decomposition using Strassen for the 2x2 block
    # A is 3x3, B is 3x8
    # Split A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Split B = [[B11, B12], [B21, B22]] where B11 is 2x4, B12 is 2x4, B21 is 1x4, B22 is 1x4
    # C = A*B is 3x8:
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Use Strassen for A11*B11 (2x2 * 2x4): rank 14
    # Use Strassen for A11*B12 (2x2 * 2x4): rank 14
    # Cross terms:
    # A12*B21: 2x1 * 1x4 -> rank 2 (just multiply each row of A12 by B21)
    # A12*B22: 2x1 * 1x4 -> rank 2
    # A21*B11: 1x2 * 2x4 -> rank 4
    # A21*B12: 1x2 * 2x4 -> rank 4
    # A22*B21: 1x1 * 1x4 -> rank 1
    # A22*B22: 1x1 * 1x4 -> rank 1
    # Total: 14 + 14 + 2 + 2 + 4 + 4 + 1 + 1 = 42... wait that's too low.
    
    # Let me reconsider. The Strassen decomposition for 2x2 * 2x2 uses 7 multiplications.
    # For 2x2 * 2x4, we can apply Strassen to each 2x2 column block, so 2 * 7 = 14.
    # But we need to be careful about how the factors combine.
    
    # Actually, let's use a direct construction.
    # The standard matmul tensor for 3x3 * 3x8 has rank at most 72 (schoolbook).
    # Using Strassen on the top-left 2x2 block of A and corresponding 2x8 block of B:
    # A11 (2x2) * B1 (2x8) where B1 = [B11, B12]
    # This can be done with rank 7*4 = 28 (Strassen for 2x2*2x2 is 7, applied to 4 column blocks)
    # Wait, 2x2 * 2x8: treat as 2x2 * [2x2, 2x2, 2x2, 2x2], each 2x2*2x2 is rank 7, total 28.
    
    # Let me try a different split.
    # A = [A_top; A_bot] where A_top is 2x3, A_bot is 1x3
    # B = [B_left, B_right] where B_left is 3x4, B_right is 3x4
    # C = A*B is 3x8
    # C_top = A_top * B (2x8)
    # C_bot = A_bot * B (1x8)
    
    # For C_top = A_top * B: A_top is 2x3, B is 3x8
    # This is a 2x3 * 3x8 matmul, schoolbook rank = 2*3*8 = 48
    # Can we do better? Split A_top = [A11, A12] where A11 is 2x2, A12 is 2x1
    # B = [B1; B2] where B1 is 2x8, B2 is 1x8
    # C_top = A11*B1 + A12*B2
    # A11*B1: 2x2 * 2x8, Strassen rank = 7*4 = 28
    # A12*B2: 2x1 * 1x8, rank = 2
    # Total for C_top: 30
    
    # For C_bot = A_bot * B: A_bot is 1x3, B is 3x8
    # A_bot = [a1, a2, a3] (1x3)
    # C_bot = a1*B1 + a2*B2 + a3*B3 where B1,B2,B3 are 1x8 rows of B
    # This is rank 3 (each term is outer product of scalar and row)
    # Wait, C_bot is 1x8, so each term a_i * B[i,:] gives a 1x8 vector
    # Total rank for C_bot: 3
    
    # But we need to combine these. The issue is that the U, V, W factors must be consistent.
    # Let me just construct it directly.
    
    # Actually, let me use a known result: R(3,3,8) <= 68
    # Construction: Use Strassen for the 2x2*2x2 block, handle the rest with schoolbook
    
    # Let's build the rank-68 decomposition:
    # A is 3x3, B is 3x8
    # Indices: A[i,j] for i in 0..2, j in 0..2
    #           B[j,k] for j in 0..2, k in 0..7
    #           C[i,k] for i in 0..2, k in 0..7
    
    # Strassen on A[0:2, 0:2] * B[0:2, 0:8] (2x2 * 2x8)
    # Let A11 = A[0:2, 0:2], A12 = A[0:2, 2] (2x1)
    #     A21 = A[2, 0:2] (1x2), A22 = A[2, 2] (1x1)
    # Let B1 = B[0:2, :] (2x8), B2 = B[2, :] (1x8)
    # C[0:2, :] = A11*B1 + A12*B2
    # C[2, :] = A21*B1 + A22*B2
    
    # For A11*B1 (2x2 * 2x8):
    # Split B1 = [B1a, B1b] where B1a is 2x4, B1b is 2x4
    # A11*B1a: Strassen rank 7
    # A11*B1b: Strassen rank 7
    # Total: 14
    
    # A12*B2: 2x1 * 1x8, rank 2
    # A21*B1: 1x2 * 2x8, rank 8 (each of 2 rows of A21 times 8 columns... wait)
    # A21 is 1x2, B1 is 2x8, so A21*B1 is 1x8
    # This is sum of 2 terms: A21[0]*B1[0,:] + A21[1]*B1[1,:], rank 2
    # A22*B2: 1x1 * 1x8, rank 1
    
    # Total rank: 14 + 2 + 2 + 1 = 19? That seems too low.
    
    # I think I'm confusing myself. Let me just use the schoolbook for now
    # and return rank 72. The inspiration says rank 68 was achieved, but I need
    # to construct it carefully.
    
    # For safety, return schoolbook rank 72
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
