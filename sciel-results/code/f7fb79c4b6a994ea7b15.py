import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Construct a rank-88 tensor decomposition for 3x4x8 matrix multiplication.
    
    Strategy:
    - Split 3x4x8 into 1x4x8 (rank 32, schoolbook) + 2x4x8
    - For 2x4x8: split into 2x2x4 (x2 blocks) 
    - Each 2x2x4: split into 2x2x2 (x2 blocks)
    - Each 2x2x2: use Strassen (rank 7)
    - Total: 32 + 2*(2*(2*7)) = 32 + 56 = 88
    
    Index conventions:
    - U: (n*m, rank), index (i*m + j)
    - V: (m*p, rank), index (j*p + k)  
    - W: (p*n, rank), index (k*n + i)  [trace convention]
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: 1x4x8 subproblem (rows 0 of A, all 4 cols, all 8 cols of B)
    # Schoolbook: 1*4*8 = 32 terms
    for i in range(1):  # i=0
        for j in range(4):
            for k in range(8):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 2: 2x4x8 subproblem (rows 1,2 of A)
    # Split 2x4x8 into two 2x2x4 blocks (cols 0-3 and 4-7 of B)
    for j_block in range(2):  # j blocks: 0-1, 2-3
        j_start = j_block * 2
        for k_block in range(2):  # k blocks: 0-3, 4-7
            k_start = k_block * 4
            # 2x2x4 subproblem: split into two 2x2x2 blocks
            for k2 in range(2):
                k2_start = k_start + k2 * 2
                # 2x2x2 subproblem: use Strassen
                # Strassen indices:
                # M1 = (A11+A22)(B11+B22)
                # M2 = (A21+A22)B11
                # M3 = A11(B21-B22)
                # M4 = A22(B12-B11)
                # M5 = (A11+A12)B22
                # M6 = (A21-A11)(B11+B12)
                # M7 = (A12-A22)(B21+B22)
                # C11 = M1+M4-M5-M7
                # C12 = M3+M5
                # C21 = M2+M4
                # C22 = M1-M2+M3+M7
                
                for s in range(7):
                    # Determine which Strassen term
                    if s == 0:  # M1 = (A11+A22)(B11+B22)
                        # A indices: (0,0) and (1,1) relative to block
                        # B indices: (0,0) and (1,1) relative to block
                        for (ai, aj, bi, bk) in [(0,0,0,0), (0,0,1,1), (1,1,0,0), (1,1,1,1)]:
                            U[(1 + ai) * m + (j_start + aj), col] = 1
                            V[(j_start + aj) * p + (k2_start + bi), col] = 1
                            W[(k2_start + bi) * n + (1 + ai), col] = 1
                    elif s == 1:  # M2 = (A21+A22)B11
                        for (ai, aj) in [(1,0), (1,1)]:
                            U[(1 + ai) * m + (j_start + aj), col] = 1
                            V[(j_start + 0) * p + (k2_start + 0), col] = 1
                            W[(k2_start + 0) * n + (1 + ai), col] = 1
                    elif s == 2:  # M3 = A11(B21-B22)
                        U[(1 + 0) * m + (j_start + 0), col] = 1
                        V[(j_start + 0) * p + (k2_start + 1), col] = 1
                        W[(k2_start + 1) * n + (1 + 0), col] = 1
                        V[(j_start + 0) * p + (k2_start + 1), col] -= 1
                        W[(k2_start + 1) * n + (1 + 0), col] -= 0  # This is wrong, let me redo
                    # Actually, Strassen is more subtle. Let me use a cleaner formulation.
                    break
                break
            break
        break
    
    # The above approach is getting messy. Let me use a cleaner Strassen implementation.
    # Reset and do it properly.
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # Part 1: 1x4x8 schoolbook (32 terms)
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
    
    # Part 2: 2x4x8 using Strassen recursively
    # 2x4x8 = two 2x2x4 blocks
    for jb in range(2):
        js = jb * 2
        # 2x2x4 = two 2x2x2 blocks
        for kb in range(2):
            ks = kb * 4
            for k2b in range(2):
                k2s = ks + k2b * 2
                # 2x2x2 Strassen: 7 terms
                # Block A indices: rows 1,2; cols js,js+1
                # Block B indices: rows js,js+1; cols k2s,k2s+1
                # Block C indices: rows k2s,k2s+1; cols 1,2
                
                # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
                U[1*m + js, col] = 1
                V[js*p + k2s, col] = 1
                W[k2s*n + 1, col] = 1
                U[1*m + (js+1), col] = 1
                V[(js+1)*p + (k2s+1), col] = 1
                W[(k2s+1)*n + 2, col] = 1
                col += 1
                
                # M2 = (A[1,0]+A[1,1])B[0,0]
                U[2*m + js, col] = 1
                V[js*p + k2s, col] = 1
                W[k2s*n + 2, col] = 1
                U[2*m + (js+1), col] = 1
                V[(js+1)*p + k2s, col] = 1
                W[k2s*n + 2, col] += 1
                col += 1
                
                # M3 = A[0,0](B[1,0]-B[1,1])
                U[1*m + js, col] = 1
                V[(js+1)*p + k2s, col] = 1
                W[k2s*n + 1, col] = 1
                V[(js+1)*p + (k2s+1), col] = -1
                W[(k2s+1)*n + 1, col] = -1
                col += 1
                
                # M4 = A[1,1](B[0,1]-B[0,0])
                U[2*m + (js+1), col] = 1
                V[js*p + (k2s+1), col] = 1
                W[(k2s+1)*n + 2, col] = 1
                V[js*p + k2s, col] = -1
                W[k2s*n + 2, col] = -1
                col += 1
                
                # M5 = (A[0,0]+A[0,1])B[1,1]
                U[1*m + js, col] = 1
                V[(js+1)*p + (k2s+1), col] = 1
                W[(k2s+1)*n + 1, col] = 1
                U[1*m + (js+1), col] = 1
                V[(js+1)*p + (k2s+1), col] += 1
                W[(k2s+1)*n + 1, col] += 1
                col += 1
                
                # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
                U[2*m + js, col] = 1
                V[js*p + k2s, col] = 1
                W[k2s*n + 2, col] = 1
                U[1*m + js, col] = -1
                V[js*p + k2s, col] += 1
                W[k2s*n + 1, col] = 1
                U[2*m + js, col] += 1
                V[js*p + (k2s+1), col] = 1
                W[(k2s+1)*n + 2, col] = 1
                U[1*m + js, col] -= 1
                V[js*p + (k2s+1), col] += 1
                W[(k2s+1)*n + 1, col] += 1
                col += 1
                
                # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
                U[1*m + (js+1), col] = 1
                V[(js+1)*p + k2s, col] = 1
                W[k2s*n + 1, col] = 1
                U[2*m + (js+1), col] = -1
                V[(js+1)*p + k2s, col] += 1
                W[k2s*n + 2, col] = 1
                U[1*m + (js+1), col] += 1
                V[(js+1)*p + (k2s+1), col] = 1
                W[(k2s+1)*n + 1, col] += 1
                U[2*m + (js+1), col] -= 1
                V[(js+1)*p + (k2s+1), col] += 1
                W[(k2s+1)*n + 2, col] += 1
                col += 1
    
    # Convert to lists of integers/half-integers
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
