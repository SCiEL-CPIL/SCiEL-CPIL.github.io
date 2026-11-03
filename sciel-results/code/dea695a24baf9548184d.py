import time
import numpy as np

DIMENSIONS = [2, 3, 4]
n, m, p = DIMENSIONS

def construct():
    deadline = time.time() + 145.0
    
    # Try to construct rank-12 decomposition
    # Strategy: partition B (3x4) into two 3x2 blocks
    # For each 2x3x2 sub-problem, try to use 6 terms
    # Total: 12 terms
    
    # Actually, let's first verify the rank-16 solution works,
    # then try to find rank 12 via structural search
    
    def _build_rank16():
        """Construct the known rank-16 decomposition."""
        U = np.zeros((n * m, 16))
        V = np.zeros((m * p, 16))
        W = np.zeros((p * n, 16))
        
        # Block 1: k=0,1
        # Term 0: m1 = A00(B00+B11)
        U[0, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[2, 0] = 1
        # Term 1: m2 = A10(B00+B01)
        U[3, 1] = 1; V[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1; W[3, 1] = 1
        # Term 2: m3 = A00(B10+B11)
        U[0, 2] = 1; V[4, 2] = 1; V[5, 2] = 1; W[0, 2] = 1; W[2, 2] = 1
        # Term 3: m4 = A11(B00+B10)
        U[4, 3] = 1; V[0, 3] = 1; V[4, 3] = 1; W[1, 3] = 1; W[3, 3] = 1
        # Term 4: m5 = (A00+A11)(B00+B11)
        U[0, 4] = 1; U[4, 4] = 1; V[0, 4] = 1; V[5, 4] = 1; W[0, 4] = 1; W[3, 4] = 1
        # Term 5: m6 = (A10+A11)(B01+B11)
        U[3, 5] = 1; U[4, 5] = 1; V[1, 5] = 1; V[5, 5] = 1; W[1, 5] = 1; W[3, 5] = 1
        # Term 6: m7 = (A00+A01)(B10+B11)
        U[0, 6] = 1; U[1, 6] = 1; V[4, 6] = 1; V[5, 6] = 1; W[0, 6] = 1; W[3, 6] = 1
        # Term 7: m8 = A_{i,2} B_{2,k}
        U[2, 7] = 1; U[5, 7] = 1; V[8, 7] = 1; V[9, 7] = 1
        W[0, 7] = 1; W[1, 7] = 1; W[2, 7] = 1; W[3, 7] = 1
        
        # Block 2: k=2,3
        # Term 8: m1
        U[0, 8] = 1; V[2, 8] = 1; V[6, 8] = 1; W[4, 8] = 1; W[6, 8] = 1
        # Term 9: m2
        U[3, 9] = 1; V[2, 9] = 1; V[3, 9] = 1; W[5, 9] = 1; W[7, 9] = 1
        # Term 10: m3
        U[0, 10] = 1; V[6, 10] = 1; V[7, 10] = 1; W[4, 10] = 1; W[6, 10] = 1
        # Term 11: m4
        U[4, 11] = 1; V[2, 11] = 1; V[6, 11] = 1; W[5, 11] = 1; W[7, 11] = 1
        # Term 12: m5
        U[0, 12] = 1; U[4, 12] = 1; V[2, 12] = 1; V[7, 12] = 1; W[4, 12] = 1; W[7, 12] = 1
        # Term 13: m6
        U[3, 13] = 1; U[4, 13] = 1; V[3, 13] = 1; V[7, 13] = 1; W[5, 13] = 1; W[7, 13] = 1
        # Term 14: m7
        U[0, 14] = 1; U[1, 14] = 1; V[6, 14] = 1; V[7, 14] = 1; W[4, 14] = 1; W[7, 14] = 1
        # Term 15: m8
        U[2, 15] = 1; U[5, 15] = 1; V[10, 15] = 1; V[11, 15] = 1
        W[4, 15] = 1; W[5, 15] = 1; W[6, 15] = 1; W[7, 15] = 1
        
        return U, V, W
    
    def _build_schoolbook():
        """Construct the schoolbook decomposition of rank 24."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def _verify(U, V, W, n, m, p, tol=1e-9):
        """Verify the decomposition is exact."""
        # Build target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Reconstruct
        recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        if np.linalg.norm(recon - T) > tol:
            return False
        
        # Also verify with random matrices
        np.random.seed(12345)
        for _ in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(U.shape[1]):
                Ar = np.zeros((n, m))
                Br = np.zeros((m, p))
                for i in range(n):
                    for j in range(m):
                        Ar[i, j] = U[i * m + j, r]
                for j in range(m):
                    for k in range(p):
                        Br[j, k] = V[j * p + k, r]
                Mr = Ar @ Br
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += W[k * n + i, r] * Mr[i, k]
            
            if not np.allclose(C_true, C_calc, atol=tol):
                return False
        
        return True
    
    # Try rank 16 first (known to work)
    U16, V16, W16 = _build_rank16()
    
    if _verify(U16, V16, W16, n, m, p):
        return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Fallback to schoolbook
    U24, V24, W24 = _build_schoolbook()
    return {"U": U24.tolist(), "V": V24.tolist(), "W": W24.tolist()}
