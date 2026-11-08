"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W, rank):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block. Returns new col."""
        # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+i1, col] += 1; W[k1*n+i1, col] -= 1
        col += 1
        # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+i0, col] += 1; W[k0*n+i1, col] += 1
        col += 1
        # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] -= 1; W[k1*n+i0, col] += 1
        col += 1
        # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] += 1; U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col
    
    def schoolbook_block(U, V, W, col, i_range, j_range, k_range):
        """Add schoolbook terms for a block. Returns new col."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col
    
    def try_strassen_split():
        rank = 54
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        # Partition m=5 into {0,1}, {2,3}, {4}
        # Partition p=6 into {0,1}, {2,3}, {4,5}
        for j0, j1 in [(0,1), (2,3)]:
            for k0, k1 in [(0,1), (2,3), (4,5)]:
                col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
        # Remaining j=4, all k
        col = schoolbook_block(U, V, W, col, [0,1], [4], range(p))
        return U, V, W, rank
    
    U, V, W, rank = try_strassen_split()
    if not validate(U, V, W, rank):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    # Convert to lists for the return interface
    U_list = [[int(U[i, c]) if abs(U[i, c] - round(U[i, c])) < 1e-9 else float(U[i, c]) for c in range(rank)] for i in range(n*m)]
    V_list = [[int(V[i, c]) if abs(V[i, c] - round(V[i, c])) < 1e-9 else float(V[i, c]) for c in range(rank)] for i in range(m*p)]
    W_list = [[int(W[i, c]) if abs(W[i, c] - round(W[i, c])) < 1e-9 else float(W[i, c]) for c in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
