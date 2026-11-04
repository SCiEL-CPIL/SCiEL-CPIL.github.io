"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2*2x2 Strassen decomposition.
    Each term is (U_coeff, V_coeff, W_coeff) where each is a 4-element vector
    for the 2x2 block (flattened row-major: [a,b,c,d] for [[a,b],[c,d]]).
    
    Strassen's 7 products for C = A*B where A,B,C are 2x2:
    A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]], C = [[c11,c12],[c21,c22]]
    
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6
    """
    # For each of the 7 products, we need U (coefficients for A elements),
    # V (coefficients for B elements), W (coefficients for C elements)
    # A elements indexed: 0=a11, 1=a12, 2=a21, 3=a22
    # B elements indexed: 0=b11, 1=b12, 2=b21, 3=b22
    # C elements indexed: 0=c11, 1=c12, 2=c21, 3=c22
    
    terms = []
    # M1 = (a11+a22)(b11+b22), contributes to c11(+1), c22(+1)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (a21+a22)b11, contributes to c21(+1), c22(-1)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # M3 = a11(b12-b22), contributes to c12(+1), c22(+1)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    # M4 = a22(b21-b11), contributes to c11(+1), c21(+1)
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
    # M5 = (a11+a12)b22, contributes to c11(-1), c12(+1)
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # M6 = (a21-a11)(b11+b12), contributes to c22(+1)
    terms.append([(-1,1,0,0), (1,1,0,0), (0,0,0,1)])
    # M7 = (a12-a22)(b21+b22), contributes to c11(+1)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses block-structured decomposition with Strassen's algorithm for 2x2x2 sub-blocks.
    Falls back to schoolbook for non-2x2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Block structure:
    # A (n x m) = (4 x 5): split rows into [2, 2]
    # B (m x p) = (5 x 6): split cols into [3, 3]
    # Middle dim j: split into [2, 2, 1]
    
    # Row blocks of A: sizes [2, 2]
    row_blocks = [2, 2]
    # Col blocks of B: sizes [3, 3]  
    col_blocks = [3, 3]
    # Middle blocks: sizes [2, 2, 1]
    mid_blocks = [2, 2, 1]
    
    # Verify dimensions
    assert sum(row_blocks) == n, f"Row blocks {row_blocks} don't sum to {n}"
    assert sum(col_blocks) == p, f"Col blocks {col_blocks} don't sum to {p}"
    assert sum(mid_blocks) == m, f"Mid blocks {mid_blocks} don't sum to {m}"
    
    # Build the decomposition
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Index conventions:
    # U[i*m + j, r] - row i, col j of A
    # V[j*p + k, r] - row j, col k of B
    # W[k*n + i, r] - output at position (k*n + i) per trace convention
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Row block offsets in A
    row_offsets = []
    offset = 0
    for rb in row_blocks:
        row_offsets.append(offset)
        offset += rb
    
    # Col block offsets in B (and in C)
    col_offsets = []
    offset = 0
    for cb in col_blocks:
        col_offsets.append(offset)
        offset += cb
    
    # Mid block offsets
    mid_offsets = []
    offset = 0
    for mb in mid_blocks:
        mid_offsets.append(offset)
        offset += mb
    
    # For each combination of (row_block, mid_block, col_block), add terms
    for ri, rb_size in enumerate(row_blocks):
        for mi, mb_size in enumerate(mid_blocks):
            for ci, cb_size in enumerate(col_blocks):
                # Block dimensions: rb_size x mb_size * mb_size x cb_size -> rb_size x cb_size
                # If all are 2, use Strassen (rank 7)
                # Otherwise use schoolbook (rank = rb_size * mb_size * cb_size)
                
                if rb_size == 2 and mb_size == 2 and cb_size == 2:
                    # Use Strassen
                    strassen_terms = _strassen_2x2x2()
                    for (u_coeff, v_coeff, w_coeff) in strassen_terms:
                        # u_coeff: 4 elements for A block [rb_size x mb_size] = [2x2]
                        # v_coeff: 4 elements for B block [mb_size x cb_size] = [2x2]
                        # w_coeff: 4 elements for C block [rb_size x cb_size] = [2x2]
                        
                        u_row = [0.0] * (n * m)
                        v_row = [0.0] * (m * p)
                        w_row = [0.0] * (p * n)
                        
                        # Fill U for this block
                        # A block starts at row (row_offsets[ri]), col (mid_offsets[mi])
                        for idx, coeff in enumerate(u_coeff):
                            if coeff != 0:
                                # idx maps to position in 2x2 block: idx = local_i * mb_size + local_j
                                local_i = idx // mb_size
                                local_j = idx % mb_size
                                global_i = row_offsets[ri] + local_i
                                global_j = mid_offsets[mi] + local_j
                                u_row[global_i * m + global_j] = coeff
                        
                        # Fill V for this block
                        # B block starts at row (mid_offsets[mi]), col (col_offsets[ci])
                        for idx, coeff in enumerate(v_coeff):
                            if coeff != 0:
                                local_j = idx // cb_size
                                local_k = idx % cb_size
                                global_j = mid_offsets[mi] + local_j
                                global_k = col_offsets[ci] + local_k
                                v_row[global_j * p + global_k] = coeff
                        
                        # Fill W for this block
                        # C block: output C[i,k] where i is in row block, k is in col block
                        # W index is k*n + i
                        for idx, coeff in enumerate(w_coeff):
                            if coeff != 0:
                                local_i = idx // cb_size
                                local_k = idx % cb_size
                                global_i = row_offsets[ri] + local_i
                                global_k = col_offsets[ci] + local_k
                                w_row[global_k * n + global_i] = coeff
                        
                        U_rows.append(u_row)
                        V_rows.append(v_row)
                        W_rows.append(w_row)
                else:
                    # Schoolbook for this block
                    for local_i in range(rb_size):
                        for local_j in range(mb_size):
                            for local_k in range(cb_size):
                                u_row = [0.0] * (n * m)
                                v_row = [0.0] * (m * p)
                                w_row = [0.0] * (p * n)
                                
                                global_i = row_offsets[ri] + local_i
                                global_j = mid_offsets[mi] + local_j
                                global_k = col_offsets[ci] + local_k
                                
                                u_row[global_i * m + global_j] = 1
                                v_row[global_j * p + global_k] = 1
                                w_row[global_k * n + global_i] = 1
                                
                                U_rows.append(u_row)
                                V_rows.append(v_row)
                                W_rows.append(w_row)
                
                # Check deadline
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    rank = len(U_rows)
    
    # Convert to lists of integers/half-integers where possible
    U = U_rows
    V = V_rows
    W = W_rows
    
    return {"U": U, "V": V, "W": W}
