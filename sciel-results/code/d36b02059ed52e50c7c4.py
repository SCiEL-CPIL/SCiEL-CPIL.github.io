import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the schoolbook rank-128 decomposition as a safe fallback
    def build_schoolbook():
        rank = 128
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition by testing with random matrices
    def verify_decomp(result):
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        rank = U.shape[1]
        
        # Test with a few random matrices
        for seed in [42, 123, 999]:
            np.random.seed(seed)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            
            C_true = A @ B
            
            # Compute decomposed product
            # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank):
                            C_decomp[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            if not np.allclose(C_true, C_decomp, atol=1e-10):
                return False
        return True
    
    # Build the rank-112 block Strassen decomposition
    def build_rank112():
        rank = 112
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # We split B into two 4x4 halves: B_left = B[:,0:4], B_right = B[:,4:8]
        # Each half uses a 4x4x4 Strassen decomposition (rank 56)
        
        # Strassen 7 products for 4x4x4:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00-A10)(B10+B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01-A11)(B00+B01)
        # M7 = (A00+A01)(B10-B00)
        
        # A coefficients for each product (2x2 block structure)
        A_coefs = [
            [[1, 0], [0, 1]],   # M1: A00 + A11
            [[0, 0], [1, 1]],   # M2: A10 + A11
            [[1, 0], [-1, 0]],  # M3: A00 - A10
            [[1, 0], [0, 0]],   # M4: A00
            [[0, 0], [0, 1]],   # M5: A11
            [[0, 1], [0, -1]],  # M6: A01 - A11
            [[1, 1], [0, 0]],   # M7: A00 + A01
        ]
        
        B_coefs = [
            [[1, 0], [0, 1]],   # M1: B00 + B11
            [[1, 0], [0, 0]],   # M2: B00
            [[0, 0], [1, 1]],   # M3: B10 + B11
            [[1, 0], [0, 0]],   # M4: B00
            [[0, 0], [0, 1]],   # M5: B11
            [[1, 1], [0, 0]],   # M6: B00 + B01
            [[0, -1], [1, 0]],  # M7: B10 - B00
        ]
        
        # C reconstruction: which C blocks each product contributes to, with sign
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        C_recon = {
            0: [(0, 0, 1), (0, 1, 1)],    # M1 -> C00 (+1), C11 (+1)
            1: [(1, 0, 1), (1, 1, -1)],   # M2 -> C10 (+1), C11 (-1)
            2: [(1, 0, 1), (1, 1, 1)],    # M3 -> C10 (+1), C11 (+1)
            3: [(0, 0, 1), (0, 1, 1)],    # M4 -> C00 (+1), C01 (+1)
            4: [(0, 0, -1), (0, 1, 1), (1, 1, 1)],  # M5 -> C00 (-1), C01 (+1), C11 (+1)
            5: [(0, 0, 1)],               # M6 -> C00 (+1)
            6: [(0, 1, 1), (1, 0, -1)],   # M7 -> C01 (+1), C10 (-1)
        }
        
        # Wait, let me re-derive Strassen correctly.
        # Standard Strassen for 2x2 block matrices:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = (A11-A21)(B21+B22)
        # M4 = A11B11
        # M5 = A22B22
        # M6 = (A12-A22)(B11+B12)
        # M7 = (A11+A12)(B21-B11)
        #
        # C11 = M1 + M4 - M5 + M6
        # C12 = M4 + M5
        # C21 = M2 + M3
        # C22 = M1 - M2 + M3 + M5
        
        # Let me redo with 0-indexed blocks:
        # A00, A01, A10, A11
        # B00, B01, B10, B11
        #
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00-A10)(B10+B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01-A11)(B00+B01)
        # M7 = (A00+A01)(B10-B00)
        #
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # But wait, let me verify C01:
        # C01 = A00*B01 + A01*B11
        # M4 + M5 = A00*B00 + A11*B11 -- this is NOT C01!
        
        # I'm using the wrong Strassen variant. Let me use the correct one.
        # The standard Strassen is:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = (A11-A21)(B21+B22)
        # M4 = A11B11
        # M5 = A22B22
        # M6 = (A12-A22)(B11+B12)
        # M7 = (A11+A12)(B21-B11)
        #
        # C11 = M1 + M4 - M5 + M6
        # C12 = M4 + M5
        # C21 = M2 + M3
        # C22 = M1 - M2 + M3 + M5
        
        # Hmm, C12 = M4 + M5 = A11B11 + A22B22. But C12 should be A11B12 + A12B22.
        # This doesn't match either. Let me look up the correct Strassen formulas.
        
        # Actually, I think I have the block indexing confused. Let me use a different approach.
        # I'll just use the schoolbook rank 128 to be safe, since the Strassen tensor
        # decomposition is error-prone to implement manually.
        
        # For now, return None to indicate this function is not ready
        return None
    
    # Try to build rank 112, verify, and fall back to 128 if needed
    result_112 = build_rank112()
    if result_112 is not None:
        if verify_decomp(result_112):
            return result_112
    
    # Fall back to schoolbook rank 128
    return build_schoolbook()
# EVOLVE-BLOCK-END
