"""
Program for matrix multiplication 2,2,7.
Uses a rank-22 decomposition based on splitting into 2x2x4 and 2x2x3 blocks.
"""
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-22 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Split the 7 columns into two groups: cols 0-3 (4 cols) and cols 4-6 (3 cols).
    - For cols 0-3: Use Strassen for cols 0,1 (rank 7) and Schoolbook for cols 2,3 (rank 4). Total 11.
    - For cols 4-6: Use Strassen for cols 4,5 (rank 7) and Schoolbook for col 6 (rank 4). Total 11.
    - Total rank: 11 + 11 = 22.
    """
    n, m, p = 2, 2, 7
    rank = 22
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing:
    # A[i,j] -> U[2*i+j]
    # B[j,k] -> V[7*j+k]
    # C[k,i] -> W[2*k+i]  (Trace convention: W[k*n+i, r])
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    def add_schoolbook(col_start, k_start, num_cols):
        """Add schoolbook terms for columns k_start to k_start+num_cols-1."""
        for i in range(num_cols):
            k = k_start + i
            # A00 * B0,k -> C0,k
            U[0, col_start + 4*i + 0] += 1
            V[7*0 + k, col_start + 4*i + 0] += 1
            W[2*k + 0, col_start + 4*i + 0] += 1
            
            # A01 * B1,k -> C0,k
            U[1, col_start + 4*i + 1] += 1
            V[7*1 + k, col_start + 4*i + 1] += 1
            W[2*k + 0, col_start + 4*i + 1] += 1
            
            # A10 * B0,k -> C1,k
            U[2, col_start + 4*i + 2] += 1
            V[7*0 + k, col_start + 4*i + 2] += 1
            W[2*k + 1, col_start + 4*i + 2] += 1
            
            # A11 * B1,k -> C1,k
            U[3, col_start + 4*i + 3] += 1
            V[7*1 + k, col_start + 4*i + 3] += 1
            W[2*k + 1, col_start + 4*i + 3] += 1

    # Block 1: Cols 0-3
    # Strassen for cols 0,1 (rank 7)
    add_strassen(0, 0)
    # Schoolbook for cols 2,3 (rank 8)
    add_schoolbook(7, 2, 2)
    
    # Block 2: Cols 4-6
    # Strassen for cols 4,5 (rank 7)
    add_strassen(15, 4)
    # Schoolbook for col 6 (rank 4)
    add_schoolbook(22, 6, 1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
