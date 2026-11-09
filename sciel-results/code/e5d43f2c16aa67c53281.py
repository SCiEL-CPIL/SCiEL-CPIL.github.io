"""Matrix multiplication 3x4x4 tensor decomposition."""
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    # We'll build the decomposition using 4 Strassen blocks for i in {1,2} (0-indexed: i in {0,1})
    # covering j in {0,1,2,3} and k in {0,1,2,3} in a 2x2 grid of blocks
    # Each block is a 2x2x2 tensor decomposed with rank 7
    # Remaining: i=2 (0-indexed), all j,k -> 4*4=16 schoolbook terms
    # Total rank: 4*7 + 16 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    # Strassen 2x2x2 decomposition for a block with i-offset di, j-offset dj, k-offset dk
    # The block covers i in {di, di+1}, j in {dj, dj+1}, k in {dk, dk+1}
    # 7 Strassen factors
    strassen_coeffs = [
        # (u_coeff_i0, u_coeff_i1, v_coeff_j0, v_coeff_j1, w_coeff_k0_i0, w_coeff_k0_i1, w_coeff_k1_i0, w_coeff_k1_i1)
        # m1 = (a00+a11)(b00+b11)
        (1, 1, 1, 1, 1, 0, 0, 1),
        # m2 = (a10+a11)b00
        (0, 0, 1, 1, 1, 0, 0, 0),
        # m3 = a00(b01-b11)
        (1, 0, 1, -1, 0, 0, 1, 0),
        # m4 = a11(b10-b00)
        (0, 1, -1, 1, 0, 0, 1, 0),
        # m5 = (a00+a01)b11
        (1, 1, 0, 1, 0, 0, 0, 1),
        # m6 = (a10-a00)(b00+b01)
        (-1, 1, 1, 1, 1, 0, 0, 0),
        # m7 = (a01-a11)(b10+b11)
        (0, 0, 1, 1, 0, 0, 1, 0),
    ]
    
    # Wait, I need to be more careful. Let me re-derive.
    # For a 2x2x2 block: A[i,j] for i in {0,1}, j in {0,1}; B[j,k] for j in {0,1}, k in {0,1}; C[k,i] for k in {0,1}, i in {0,1}
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # 
    # Strassen:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # U indices: i*m + j -> for block with di, dj: (di+di_offset)*m + (dj+dj_offset)
    # V indices: j*p + k -> for block with dj, dk: (dj+dj_offset)*p + (dk+dk_offset)
    # W indices: k*n + i -> for block with dk, di: (dk+dk_offset)*n + (di+di_offset)
    
    strassen_factors = [
        # Each: (U_i0_coeff, U_i1_coeff, V_j0_coeff, V_j1_coeff, W_k0_i0_coeff, W_k0_i1_coeff, W_k1_i0_coeff, W_k1_i1_coeff)
        # m1 = (A00+A11)(B00+B11)
        (1, 1, 1, 1, 1, 0, 0, 1),
        # m2 = (A10+A11)B00
        (0, 0, 1, 1, 1, 0, 0, 0),
        # m3 = A00(B01-B11)
        (1, 0, 1, -1, 0, 0, 1, 0),
        # m4 = A11(B10-B00)
        (0, 1, -1, 1, 0, 0, 1, 0),
        # m5 = (A00+A01)B11
        (1, 1, 0, 1, 0, 0, 0, 1),
        # m6 = (A10-A00)(B00+B01)
        (-1, 1, 1, 1, 1, 0, 0, 0),
        # m7 = (A01-A11)(B10+B11)
        (0, 0, 1, 1, 0, 0, 1, 0),
    ]
    
    # Hmm wait, I'm confusing myself. Let me redo this more carefully.
    
    # For a 2x2x2 block at offsets (di, dj, dk):
    # U row for A[i,j] where i in {di, di+1}, j in {dj, dj+1}: row = (di+i_off)*m + (dj+j_off)
    # V row for B[j,k] where j in {dj, dj+1}, k in {dk, dk+1}: row = (dj+j_off)*p + (dk+k_off)
    # W row for C[k,i] where k in {dk, dk+1}, i in {di, di+1}: row = (dk+k_off)*n + (di+i_off)
    
    # Strassen m1 = (A00+A11)(B00+B11):
    #   U: row(di* m + dj) += 1, row((di+1)*m + (dj+1)) += 1
    #   V: row(dj*p + dk) += 1, row((dj+1)*p + (dk+1)) += 1
    #   W: row(dk*n + di) += 1, row((dk+1)*n + (di+1)) += 1
    
    # m2 = (A10+A11)B00:
    #   U: row((di+1)*m + dj) += 1, row((di+1)*m + (dj+1)) += 1
    #   V: row(dj*p + dk) += 1
    #   W: row(dk*n + (di+1)) += 1
    
    # m3 = A00(B01-B11):
    #   U: row(di*m + dj) += 1
    #   V: row((dj)*p + (dk+1)) += 1, row((dj+1)*p + (dk+1)) -= 1
    #   W: row((dk+1)*n + di) += 1
    
    # m4 = A11(B10-B00):
    #   U: row((di+1)*m + (dj+1)) += 1
    #   V: row((dj+1)*p + dk) += 1, row(dj*p + dk) -= 1
    #   W: row((dk+1)*n + (di+1)) += 1
    
    # m5 = (A00+A01)B11:
    #   U: row(di*m + dj) += 1, row(di*m + (dj+1)) += 1
    #   V: row((dj+1)*p + (dk+1)) += 1
    #   W: row((dk+1)*n + di) += 1
    
    # m6 = (A10-A00)(B00+B01):
    #   U: row((di+1)*m + dj) += 1, row(di*m + dj) -= 1
    #   V: row(dj*p + dk) += 1, row(dj*p + (dk+1)) += 1
    #   W: row(dk*n + (di+1)) += 1
    
    # m7 = (A01-A11)(B10+B11):
    #   U: row(di*m + (dj+1)) += 1, row((di+1)*m + (dj+1)) -= 1
    #   V: row((dj+1)*p + dk) += 1, row((dj+1)*p + (dk+1)) += 1
    #   W: row((dk+1)*n + di) += 1
    
    # Now let's define the 4 blocks for i in {0,1}:
    # Block (0,0): di=0, dj=0, dk=0
    # Block (0,1): di=0, dj=0, dk=2
    # Block (1,0): di=0, dj=2, dk=0
    # Block (1,1): di=0, dj=2, dk=2
    # Wait, I want to cover i in {0,1} fully with j in {0,1,2,3} and k in {0,1,2,3}
    # So 4 blocks: (di=0, dj=0, dk=0), (di=0, dj=0, dk=2), (di=0, dj=2, dk=0), (di=0, dj=2, dk=2)
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2),
    ]
    
    for (di, dj, dk) in blocks:
        # 7 Strassen factors
        # m1
        U[di*m + dj][col] = 1
        U[(di+1)*m + (dj+1)][col] = 1
        V[dj*p + dk][col] = 1
        V[(dj+1)*p + (dk+1)][col] = 1
        W[dk*n + di][col] = 1
        W[(dk+1)*n + (di+1)][col] = 1
        col += 1
        
        # m2
        U[(di+1)*m + dj][col] = 1
        U[(di+1)*m + (dj+1)][col] = 1
        V[dj*p + dk][col] = 1
        W[dk*n + (di+1)][col] = 1
        col += 1
        
        # m3
        U[di*m + dj][col] = 1
        V[dj*p + (dk+1)][col] = 1
        V[(dj+1)*p + (dk+1)][col] = -1
        W[(dk+1)*n + di][col] = 1
        col += 1
        
        # m4
        U[(di+1)*m + (dj+1)][col] = 1
        V[(dj+1)*p + dk][col] = 1
        V[dj*p + dk][col] = -1
        W[(dk+1)*n + (di+1)][col] = 1
        col += 1
        
        # m5
        U[di*m + dj][col] = 1
        U[di*m + (dj+1)][col] = 1
        V[(dj+1)*p + (dk+1)][col] = 1
        W[(dk+1)*n + di][col] = 1
        col += 1
        
        # m6
        U[(di+1)*m + dj][col] = 1
        U[di*m + dj][col] = -1
        V[dj*p + dk][col] = 1
        V[dj*p + (dk+1)][col] = 1
        W[dk*n + (di+1)][col] = 1
        col += 1
        
        # m7
        U[di*m + (dj+1)][col] = 1
        U[(di+1)*m + (dj+1)][col] = -1
        V[(dj+1)*p + dk][col] = 1
        V[(dj+1)*p + (dk+1)][col] = 1
        W[(dk+1)*n + di][col] = 1
        col += 1
    
    # Now handle i=2 (0-indexed), all j in {0,1,2,3}, k in {0,1,2,3}: 16 schoolbook terms
    for j in range(m):
        for k in range(p):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

