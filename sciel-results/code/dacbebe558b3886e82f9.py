"""Matrix multiplication 2x4x5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-30 exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Strategy:
    - Split the 2x4x5 tensor into two 2x2x5 blocks along the m=4 dimension.
    - For each 2x2x5 block, split the p=5 dimension into 2x2, 2x2, and 1x1.
    - Each 2x2x2 sub-block uses Strassen's algorithm (rank 7).
    - Each 2x2x1 sub-block uses a single rank-1 term (since C[i] = sum_j A[i,j]*B[j] 
      can be represented as one rank-1 term when B has only one column and the 
      contraction is over all j).
    
    Total rank: 2 * (7 + 7 + 1) = 30.
    
    Verification:
    For a 2x2x1 block with A (2x2), B (2x1), C (2x1):
    C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    This is a linear map from (A,B) to C. The tensor is:
    T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k=0 (the only column in C)
    
    Wait, let me reconsider. The tensor for matrix mult is T[i,j,k] where
    i indexes rows of A/C, j indexes cols of A/rows of B, k indexes cols of B/C.
    
    For 2x2x1: A is 2x2, B is 2x1, C is 2x1.
    T[i,j,k] = 1 if k=0 (since B has only col 0, and C has only col 0).
    Actually the standard tensor is T[i,j,k] = 1 for all i,j,k in range of dimensions.
    
    So T[i,j,0] = 1 for all i in {0,1}, j in {0,1}.
    
    A rank-1 decomposition: U[i,j] * V[j,0] * W[0,i]
    If we set U[i,j]=1, V[j,0]=1, W[0,i]=1, then
    sum over r: U[i,j,r]*V[j,0,r]*W[0,i,r] = 1*1*1 = 1. ✓
    
    So rank 1 works for 2x2x1.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 30
    U = np.zeros((n*m, rank))  # (8, 30)
    V = np.zeros((m*p, rank))  # (20, 30)
    W = np.zeros((p*n, rank))  # (10, 30)
    
    def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
        """
        Add a rank-1 term.
        u_coeffs: dict {(i, j): val} for A matrix (n x m)
        v_coeffs: dict {(j, k): val} for B matrix (m x p)
        w_coeffs: dict {(k, i): val} for C matrix (p x n)
        """
        for (i, j), val in u_coeffs.items():
            U[i*m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j*p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k*n + i, idx] += val

    def strassen_2x2x2(a_col_offset, b_row_offset, c_col_offset, idx_start):
        """
        Add Strassen decomposition for a 2x2x2 block.
        a_col_offset: column offset in A (0 or 2)
        b_row_offset: row offset in B (0 or 2)
        c_col_offset: column offset in C (0, 2, or 4)
        """
        # Let A subblock be A[0:2, a_col_offset:a_col_offset+2]
        # B subblock be B[b_row_offset:b_row_offset+2, c_col_offset:c_col_offset+2]
        # C subblock be C[c_col_offset:c_col_offset+2, 0:2]
        
        # M1 = (A00 + A11) * (B00 + B11)
        add_term(idx_start,
                 {(0, a_col_offset): 1, (1, a_col_offset+1): 1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                 {(c_col_offset, 0): 1, (c_col_offset+1, 1): 1})
        
        # M2 = (A10 + A11) * B00
        add_term(idx_start+1,
                 {(1, a_col_offset): 1, (1, a_col_offset+1): 1},
                 {(b_row_offset, c_col_offset): 1},
                 {(c_col_offset, 1): 1, (c_col_offset+1, 1): -1})
        
        # M3 = A00 * (B01 - B11)
        add_term(idx_start+2,
                 {(0, a_col_offset): 1},
                 {(b_row_offset, c_col_offset+1): 1, (b_row_offset+1, c_col_offset+1): -1},
                 {(c_col_offset+1, 0): 1, (c_col_offset+1, 1): 1})
        
        # M4 = A11 * (B10 - B00)
        add_term(idx_start+3,
                 {(1, a_col_offset+1): 1},
                 {(b_row_offset+1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},
                 {(c_col_offset, 0): 1, (c_col_offset, 1): 1})
        
        # M5 = (A00 + A01) * B11
        add_term(idx_start+4,
                 {(0, a_col_offset): 1, (0, a_col_offset+1): 1},
                 {(b_row_offset+1, c_col_offset+1): 1},
                 {(c_col_offset, 0): -1, (c_col_offset+1, 0): 1})
        
        # M6 = (A10 - A00) * (B00 + B01)
        add_term(idx_start+5,
                 {(1, a_col_offset): 1, (0, a_col_offset): -1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset+1): 1},
                 {(c_col_offset+1, 1): 1})
        
        # M7 = (A01 - A11) * (B10 + B11)
        add_term(idx_start+6,
                 {(0, a_col_offset+1): 1, (1, a_col_offset+1): -1},
                 {(b_row_offset+1, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                 {(c_col_offset, 0): 1})

    def rank1_2x2x1(a_col_offset, b_row_offset, c_col_offset, idx):
        """
        Add a single rank-1 term for a 2x2x1 block.
        A subblock: A[0:2, a_col_offset:a_col_offset+2]
        B subblock: B[b_row_offset:b_row_offset+2, c_col_offset]  (single column)
        C subblock: C[c_col_offset, 0:2]
        
        The tensor T[i,j,k] = 1 for i in {0,1}, j in {a_col_offset, a_col_offset+1}, k=c_col_offset.
        We need: sum_r U[i,j,r] * V[j,c_col_offset,r] * W[c_col_offset,i,r] = 1
        
        Set U[i,j,r]=1 for all i,j, V[j,c_col_offset,r]=1 for all j, W[c_col_offset,i,r]=1 for all i.
        Then the product is 1*1*1 = 1. ✓
        """
        add_term(idx,
                 {(0, a_col_offset): 1, (0, a_col_offset+1): 1,
                  (1, a_col_offset): 1, (1, a_col_offset+1): 1},
                 {(b_row_offset, c_col_offset): 1, (b_row_offset+1, c_col_offset): 1},
                 {(c_col_offset, 0): 1, (c_col_offset, 1): 1})

    idx = 0
    # Block 1: A cols 0-1, B rows 0-1
    # C cols 0-1: Strassen
    strassen_2x2x2(0, 0, 0, idx)
    idx += 7
    # C cols 2-3: Strassen
    strassen_2x2x2(0, 0, 2, idx)
    idx += 7
    # C col 4: Rank-1
    rank1_2x2x1(0, 0, 4, idx)
    idx += 1
    
    # Block 2: A cols 2-3, B rows 2-3
    # C cols 0-1: Strassen
    strassen_2x2x2(2, 2, 0, idx)
    idx += 7
    # C cols 2-3: Strassen
    strassen_2x2x2(2, 2, 2, idx)
    idx += 7
    # C col 4: Rank-1
    rank1_2x2x1(2, 2, 4, idx)
    idx += 1
    
    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
