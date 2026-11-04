"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    def add_strassen(U, V, W, i0, j0, k0, col):
        # Standard Strassen 2x2x2
        # M1 = (A11+A22)(B11+B22)
        U[i0*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
        V[j0*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M2 = (A21+A22)B11
        U[(i0+1)*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+(i0+1), col] += 1; W[(k0+1)*n+(i0+1), col] -= 1
        col += 1
        # M3 = A11(B12-B22)
        U[i0*m+j0, col] += 1
        V[j0*p+(k0+1), col] += 1; V[(j0+1)*p+(k0+1), col] -= 1
        W[(k0+1)*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M4 = A22(B21-B11)
        U[(i0+1)*m+(j0+1), col] += 1
        V[(j0+1)*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+(i0+1), col] += 1; W[k0*n+i0, col] += 1
        col += 1
        # M5 = (A11+A12)B22
        U[i0*m+j0, col] += 1; U[i0*m+(j0+1), col] += 1
        V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] -= 1; W[(k0+1)*n+i0, col] += 1
        col += 1
        # M6 = (A21-A11)(B11+B12)
        U[(i0+1)*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+(k0+1), col] += 1
        W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[i0*m+(j0+1), col] += 1; U[(i0+1)*m+(j0+1), col] -= 1
        V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col
    
    def build_rank50():
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        col = add_strassen(U, V, W, 0, 0, 0, col)
        col = add_strassen(U, V, W, 0, 0, 2, col)
        col = add_strassen(U, V, W, 0, 2, 0, col)
        col = add_strassen(U, V, W, 0, 2, 2, col)
        col = add_strassen(U, V, W, 0, 0, 4, col)
        col = add_strassen(U, V, W, 0, 2, 4, col)
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1
        return U, V, W
    
    def build_rank49():
        # Try rank 49: 6 Strassen blocks (42) + 7 terms for last column
        rank = 49
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        col = add_strassen(U, V, W, 0, 0, 0, col)
        col = add_strassen(U, V, W, 0, 0, 2, col)
        col = add_strassen(U, V, W, 0, 2, 0, col)
        col = add_strassen(U, V, W, 0, 2, 2, col)
        col = add_strassen(U, V, W, 0, 0, 4, col)
        col = add_strassen(U, V, W, 0, 2, 4, col)
        # 7 terms for column 6 (2x4x1)
        # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6] + A[0,2]*B[2,6] + A[0,3]*B[3,6]
        # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6] + A[1,2]*B[2,6] + A[1,3]*B[3,6]
        # Term 1: (A[0,0]+A[1,1]) * B[0,6] -> contributes to C[0,6] and C[1,6]
        U[0*m+0, col] += 1; U[1*m+1, col] += 1
        V[0*p+6, col] += 1
        W[6*n+0, col] += 1; W[6*n+1, col] += 1
        col += 1
        # Term 2: (A[1,0]-A[0,0]) * B[0,6] -> corrects C[1,6] and C[0,6]
        U[1*m+0, col] += 1; U[0*m+0, col] -= 1
        V[0*p+6, col] += 1
        W[6*n+1, col] += 1; W[6*n+0, col] -= 1
        col += 1
        # Term 3: A[0,1]*B[1,6]
        U[0*m+1, col] += 1
        V[1*p+6, col] += 1
        W[6*n+0, col] += 1
        col += 1
        # Term 4: A[1,1]*B[1,6]
        U[1*m+1, col] += 1
        V[1*p+6, col] += 1
        W[6*n+1, col] += 1
        col += 1
        # Term 5: A[0,2]*B[2,6]
        U[0*m+2, col] += 1
        V[2*p+6, col] += 1
        W[6*n+0, col] += 1
        col += 1
        # Term 6: A[1,2]*B[2,6]
        U[1*m+2, col] += 1
        V[2*p+6, col] += 1
        W[6*n+1, col] += 1
        col += 1
        # Term 7: (A[0,3]+A[1,3])*B[3,6]
        U[0*m+3, col] += 1; U[1*m+3, col] += 1
        V[3*p+6, col] += 1
        W[6*n+0, col] += 1; W[6*n+1, col] += 1
        col += 1
        return U, V, W
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Target: T[i,j,k] = 1 if all indices valid (matrix multiplication tensor)
        # Actually, the matrix multiplication tensor is:
        # T[i,j,k] = 1 for the "multiplication" pattern
        # For M_{n,m,p}: T[i,j,k] = 1 where i in [0,n), j in [0,m), k in [0,p)
        # The tensor represents: C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity-like tensor for matmul)
        
        # Actually, the correct check: for random A and B, compute C = A@B
        # and verify that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = C[i,k]
        
        # Let's do a numerical check
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Try rank 49 first
    try:
        U, V, W = build_rank49()
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fall back to rank 50
    U, V, W = build_rank50()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
