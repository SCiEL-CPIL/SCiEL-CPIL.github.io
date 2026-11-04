import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 40 decomposition
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx):
        nonlocal col
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen_2x2x2(i_offset, j_offset, k_offset):
        # i_offset: 0-based row index of A block (size 2)
        # j_offset: 0-based col index of A block (size 2) / row index of B block
        # k_offset: 0-based col index of B block (size 2)
        # A indices: i_offset, i_offset+1
        # B indices: j_offset, j_offset+1 for rows; k_offset, k_offset+1 for cols
        # Strassen 7 products
        # M1 = (A11+A22)(B11+B22)
        add_term(i_offset * m + j_offset, j_offset * p + k_offset, k_offset * n + i_offset)
        add_term(i_offset * m + j_offset + 1, (j_offset + 1) * p + (k_offset + 1), (k_offset + 1) * n + (i_offset + 1))
        # Wait, Strassen requires sums. We cannot just add terms.
        # We need to construct the linear combinations in U, V, W.
        # This function is a placeholder. We must implement the actual linear combinations.
        pass

    # Correct implementation:
    # We need to explicitly build the U, V, W entries for each Strassen block.
    
    # Helper to add a Strassen block for A[2x2] * B[2x2]
    # A rows: i0, i0+1. A cols: j0, j0+1.
    # B rows: j0, j0+1. B cols: k0, k0+1.
    # Result C rows: i0, i0+1. C cols: k0, k0+1.
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22. V: B11, B22. W: C11, C22.
        # We need to encode the sums.
        # Term 1: A11 * B11 * C11
        # Term 2: A22 * B22 * C22
        # But Strassen uses (A11+A22).
        # This requires setting U entries to 1 for both A11 and A22 in the same column?
        # No, the decomposition is a sum of rank-1 tensors.
        # Rank-1 tensor: U_{i,j} V_{j,k} W_{k,i}.
        # For Strassen M1, we need to compute (A11+A22)(B11+B22).
        # This is not a single rank-1 term.
        # Strassen algorithm computes 7 products of linear combinations.
        # Each product is a bilinear form.
        # (A11+A22) is a linear combination of A entries.
        # (B11+B22) is a linear combination of B entries.
        # The product is a sum of 4 terms: A11B11 + A11B22 + A22B11 + A22B22.
        # We need to represent this sum using rank-1 tensors.
        # Actually, the Strassen decomposition *is* the rank-7 decomposition.
        # The 7 "multiplications" are the rank-1 terms in the tensor decomposition?
        # No. The 7 multiplications are scalar multiplications.
        # The tensor rank is 7.
        # The 7 rank-1 tensors correspond to the 7 Strassen products.
        # For M1 = (A11+A22)(B11+B22), the corresponding rank-1 tensor has:
        # U_{i,j} non-zero for (1,1) and (2,2).
        # V_{j,k} non-zero for (1,1) and (2,2).
        # W_{k,i} non-zero for (1,1) and (2,2).
        # Specifically:
        # U[1,1]=1, U[2,2]=1.
        # V[1,1]=1, V[2,2]=1.
        # W[1,1]=1, W[2,2]=1.
        # This gives A11B11 + A11B22 + A22B11 + A22B22.
        # Which is (A11+A22)(B11+B22).
        
        # So, for each Strassen product, we set the appropriate U, V, W entries to 1 (or -1).
        
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # U: (1,1)=1, (2,2)=1
        # V: (1,1)=1, (2,2)=1
        # W: (1,1)=1, (2,2)=1
        U[i0 * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[j0 * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        
        # M2: (A11+A22)B21 -> C21 += M2
        # U: (1,1)=1, (2,2)=1
        # V: (2,1)=1
        # W: (2,1)=1
        U[i0 * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1
        
        # M3: A11(B12-B22) -> C12 += M3
        # U: (1,1)=1
        # V: (1,2)=1, (2,2)=-1
        # W: (2,1)=1
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = -1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1
        
        # M4: A22(B21-B11) -> C21 += M4
        # U: (2,2)=1
        # V: (2,1)=1, (1,1)=-1
        # W: (2,1)=1
        U[(i0 + 1) * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1
        
        # M5: (A21+A11)B11 -> C11 += M5
        # U: (2,1)=1, (1,1)=1
        # V: (1,1)=1
        # W: (1,1)=1
        U[(i0 + 1) * m + j0][col] = 1
        U[i0 * m + j0][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1
        
        # M6: (A12-A21)(B21+B22) -> C12 += M6
        # U: (1,2)=1, (2,1)=-1
        # V: (2,1)=1, (2,2)=1
        # W: (2,1)=1
        U[i0 * m + (j0 + 1)][col] = 1
        U[(i0 + 1) * m + j0][col] = -1
        V[(j0 + 1) * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1
        
        # M7: (A11-A22)(B12+B11) -> C12 += M7
        # U: (1,1)=1, (2,2)=-1
        # V: (1,2)=1, (1,1)=1
        # W: (2,1)=1
        U[i0 * m + j0][col] = 1
        U[(i0 + 1) * m + (j0 + 1)][col] = -1
        V[j0 * p + (k0 + 1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1

    # Helper to add a 2x2x1 block (A[2x2] * B[2x1])
    # A rows i0, i0+1. A cols j0, j0+1.
    # B rows j0, j0+1. B col k0.
    # C rows i0, i0+1. C col k0.
    # Rank 2.
    def add_2x2x1(i0, j0, k0):
        nonlocal col
        # Term 1: j0
        U[i0 * m + j0][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1
        # Term 2: j0+1
        U[i0 * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        col += 1

    # Top rows (i=0,1)
    # A_L: i0=0, j0=0 (cols 0,1)
    # A_R: i0=0, j0=2 (cols 2,3)
    # B_L: j0=0 (rows 0,1), k0=0,2,4 (cols 0-1, 2-3, 4-5)
    # B_R: j0=2 (rows 2,3), k0=0,2,4
    
    # A_L * B_L (3 blocks of 2x2x2)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    
    # A_R * B_R (3 blocks of 2x2x2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 2, 4)
    
    # Bottom row (i=2)
    # A_Bottom_L: i0=2, j0=0 (cols 0,1). But i is single row.
    # We need 1x2x2 blocks.
    # My add_2x2x1 assumes A is 2x2.
    # For bottom row, A is 1x2.
    # We can treat it as part of a 2x2 block where the top row is zero?
    # Or just write specific terms.
    # C_2, k = A_2,0 B_0,k + A_2,1 B_1,k + A_2,2 B_2,k + A_2,3 B_3,k
    
    # For j=0,1 (B_L rows 0,1)
    # C_2, k = A_2,0 B_0,k + A_2,1 B_1,k
    # This is 2x2x1 logic but with A row 2 only.
    # Term 1: A_2,0 B_0,k
    # U[2*m + 0] = 1
    # V[0*p + k] = 1
    # W[k*n + 2] = 1
    # Term 2: A_2,1 B_1,k
    # U[2*m + 1] = 1
    # V[1*p + k] = 1
    # W[k*n + 2] = 1
    
    # We have 6 columns for B.
    # For each column k (0..5), we have 2 terms.
    # Total 12 terms?
    # No, we can group by k blocks of 2.
    # For k=0,1:
    # Term 1: A_2,0 (B_0,0 + B_0,1)? No.
    
    # Let's just use the 2x2x1 helper by embedding the 1x2 into a 2x2 with zeros.
    # Or simpler:
    # For j=0: U[2*m+0]=1, V[0*p+k]=1, W[k*n+2]=1
    # For j=1: U[2*m+1]=1, V[1*p+k]=1, W[k*n+2]=1
    # This is 2 terms per k. 6 k's -> 12 terms.
    # But we can do better?
    # R(M_{1,2,2}) = 2.
    # So for each 2-column block of B, we use 2 ranks.
    # 3 blocks -> 6 ranks.
    
    # Let's implement add_1x2x2(i0, j0, k0)
    def add_1x2x2(i0, j0, k0):
        nonlocal col
        # A row i0, cols j0, j0+1
        # B rows j0, j0+1, cols k0, k0+1
        # C row i0, cols k0, k0+1
        # Term 1: j0
        U[i0 * m + j0][col] = 1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1
        # Term 2: j0+1
        U[i0 * m + (j0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1

    # Bottom row: i0=2
    # j=0,1 (B rows 0,1)
    add_1x2x2(2, 0, 0)
    add_1x2x2(2, 0, 2)
    add_1x2x2(2, 0, 4)
    
    # j=2,3 (B rows 2,3)
    add_1x2x2(2, 2, 0)
    add_1x2x2(2, 2, 2)
    add_1x2x2(2, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
