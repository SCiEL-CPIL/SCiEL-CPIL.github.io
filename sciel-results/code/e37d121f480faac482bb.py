"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Build the target tensor: T[i,j,k] = 1 if (i,j) in [0,n) x [0,m) and (j,k) in [0,m) x [0,p)
    # W is indexed as W[k*n + i, r], V as V[j*p + k, r], U as U[i*m + j, r]
    
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def build_strassen_2x2x2(i0, j0, k0, U, V, W, col):
        """Add 7 Strassen mults for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        # m1 = (a00+a11)(b00+b11) -> C[i0,k0] += m1, C[i0+1,k0+1] += m1
        U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # m2 = (a10+a11)b00 -> C[i0+1,k0] += m2, C[i0+1,k0+1] -= m2
        U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] += -1
        col += 1
        # m3 = a00(b10-b11) -> C[i0,k0+1] += m3, C[i0+1,k0+1] += m3
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + k0, col] += -1
        W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # m4 = a11(b01-b00) -> C[i0,k0] += m4, C[i0+1,k0] += m4
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] += -1
        W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
        col += 1
        # m5 = (a00+a01)b11 -> C[i0,k0] -= m5, C[i0,k0+1] += m5
        U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += -1; W[(k0+1)*n + i0, col] += 1
        col += 1
        # m6 = (a10-a00)(b00+b01) -> C[i0+1,k0+1] += m6
        U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] += -1
        V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # m7 = (a01-a11)(b10+b11) -> C[i0,k0] += m7
        U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] += -1
        V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1
        return col
    
    def build_strassen_2x2x4(i0, j0, k0, U, V, W, col):
        """A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+4] using two 2x2x2 Strassen"""
        col = build_strassen_2x2x2(i0, j0, k0, U, V, W, col)
        col = build_strassen_2x2x2(i0, j0, k0+2, U, V, W, col)
        return col
    
    def build_1x2x2(i0, j0, k0, U, V, W, col):
        """A[i0, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0, k0:k0+2]
        4 mults: a*b00, a*b10 -> C[i0,k0], a*b01, a*b11 -> C[i0,k0+1]
        Actually schoolbook: 2*2 = 4 mults
        """
        # a * b00
        U[i0*m + j0, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + i0, col] = 1; col += 1
        # a * b01
        U[i0*m + j0, col] = 1; V[j0*p + (k0+1), col] = 1; W[(k0+1)*n + i0, col] = 1; col += 1
        # a * b10
        U[i0*m + (j0+1), col] = 1; V[(j0+1)*p + k0, col] = 1; W[k0*n + i0, col] = 1; col += 1
        # a * b11
        U[i0*m + (j0+1), col] = 1; V[(j0+1)*p + (k0+1), col] = 1; W[(k0+1)*n + i0, col] = 1; col += 1
        return col
    
    def build_1x2x4(i0, j0, k0, U, V, W, col):
        """A[i0, j0:j0+2] * B[j0:j0+2, k0:k0+4] -> 8 mults"""
        col = build_1x2x2(i0, j0, k0, U, V, W, col)
        col = build_1x2x2(i0, j0, k0+2, U, V, W, col)
        return col
    
    def build_2x1x2(i0, j0, k0, U, V, W, col):
        """A[i0:i0+2, j0] * B[j0, k0:k0+2] -> 4 mults"""
        # a00 * b
        U[i0*m + j0, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + i0, col] = 1; col += 1
        U[i0*m + j0, col] = 1; V[j0*p + (k0+1), col] = 1; W[(k0+1)*n + i0, col] = 1; col += 1
        # a10 * b
        U[(i0+1)*m + j0, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + (i0+1), col] = 1; col += 1
        U[(i0+1)*m + j0, col] = 1; V[j0*p + (k0+1), col] = 1; W[(k0+1)*n + (i0+1), col] = 1; col += 1
        return col
    
    def build_2x1x4(i0, j0, k0, U, V, W, col):
        """A[i0:i0+2, j0] * B[j0, k0:k0+4] -> 8 mults"""
        col = build_2x1x2(i0, j0, k0, U, V, W, col)
        col = build_2x1x2(i0, j0, k0+2, U, V, W, col)
        return col
    
    def build_1x1x4(i, j, k0, U, V, W, col):
        """A[i,j] * B[j, k0:k0+4] -> 4 mults"""
        for kk in range(4):
            U[i*m + j, col] = 1; V[j*p + (k0+kk), col] = 1; W[(k0+kk)*n + i, col] = 1; col += 1
        return col
    
    # Strategy 1: Rank 34 decomposition
    # Block 1: A[0:2, 0:2] * B[0:2, 0:4] -> 2 Strassen 2x2x2 = 14
    # Block 2: A[0:2, 2] * B[2, 0:4] -> 8
    # Block 3: A[2, 0:2] * B[0:2, 0:4] -> 8
    # Block 4: A[2, 2] * B[2, 0:4] -> 4
    # Total: 34
    rank34 = 34
    U34 = np.zeros((n*m, rank34))
    V34 = np.zeros((m*p, rank34))
    W34 = np.zeros((p*n, rank34))
    col = 0
    col = build_strassen_2x2x4(0, 0, 0, U34, V34, W34, col)  # 14
    col = build_2x1x4(0, 2, 0, U34, V34, W34, col)           # 8
    col = build_1x2x4(2, 0, 0, U34, V34, W34, col)           # 8
    col = build_1x1x4(2, 2, 0, U34, V34, W34, col)           # 4
    
    # Verify
    def verify(U, V, W):
        """Check if decomposition is exact. T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]"""
        target = np.zeros((n*m*p,))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m*p + j*p + k] = 1.0
        computed = np.zeros((n*m*p,))
        for r in range(U.shape[1]):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        computed[i*m*p + j*p + k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        return np.allclose(computed, target, atol=1e-9)
    
    if verify(U34, V34, W34):
        return {"U": U34.tolist(), "V": V34.tolist(), "W": W34.tolist()}
    
    # Fallback: schoolbook rank 36
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
