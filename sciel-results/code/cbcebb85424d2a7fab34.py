import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 3, 5, 7
    
    # Winograd scheme for M_{1, 8, 8}
    # Rank 8
    # We construct U, V, W for M_{1, 8, 8}
    # Then we take 3 copies for M_{3, 8, 8}
    # Then we extract sub-matrices for M_{3, 5, 7}
    
    def get_winograd_1_8_8():
        # Returns U, V, W for M_{1, 8, 8} with rank 8
        # U: 8x8, V: 64x8, W: 8x8
        # This is the standard Winograd decomposition
        # The matrices are derived from the DFT-like structure
        # For now, I will use a known valid decomposition
        # If I cannot generate it, I will use schoolbook for M_{1,8,8} (rank 64)
        # But rank 8 is possible.
        
        # Construction of M_{1, 2^k, 2^k} rank 2^k
        # U is the matrix such that A' = A U
        # W is the matrix such that C = C' W
        # V is constructed such that B' = V B
        # For M_{1, d, d}, U and W are often the same (DFT matrix)
        # V is the "middle" matrix
        
        # For d=8, we can use the Kronecker product of d=2 matrices
        # Base case d=2:
        # U_2 = [[1, 0], [1, 1]] ? No.
        # Let's use the explicit matrices for d=4 and d=8 from literature
        # Or simply return schoolbook if generation fails
        pass

    # Given the complexity of generating the exact Winograd matrices on the fly
    # and the risk of error, I will stick to the schoolbook decomposition
    # but I will try to implement the rank 24 logic if I can verify it quickly.
    # However, without the explicit matrices, I cannot return rank 24.
    # I will return the schoolbook decomposition which is guaranteed correct.
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    return schoolbook()
