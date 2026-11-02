import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Hybrid Strassen/schoolbook decomposition
    # A: 2x3, B: 3x7, C: 2x7
    # Partition: A = [A_left (2x2) | A_right (2x1)]
    #             B = [B_top (2x7); B_bottom (1x7)]
    # C = A_left * B_top + A_right * B_bottom + A_left * B_bottom + A_right * B_top
    # Wait, that's not right. C[i,k] = sum_j A[i,j]*B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Let's partition j into {0,1} and {2}:
    # C = A[:, :2] * B[:2, :] + A[:, 2:3] * B[2:3, :]
    # First term: 2x2 * 2x7 -> 2x7
    # Second term: 2x1 * 1x7 -> 2x7
    
    # For the first term (2x2 * 2x7), we can use Strassen-like decomposition
    # Strassen for 2x2 * 2x2 has rank 7. For 2x2 * 2x7, we can do:
    # B[:2, :] = [B00 B01 B02 B03 B04 B05 B06; B10 B11 B12 B13 B14 B15 B16]
    # We can apply Strassen to each pair of columns, but that gives 7*4 = 28 for 4 pairs
    # Better: think of B[:2,:] as two 2x4 blocks? No.
    
    # Actually, let's use a different partition.
    # Partition the 7 columns of B into 2+2+2+1
    # Then 2x2 * 2x2 (rank 7 via Strassen) for each 2-column block, plus schoolbook for the last column
    # 3 blocks of 2 columns: 3 * 7 = 21
    # 1 block of 1 column: 2x2 * 2x1, schoolbook rank 4
    # Total for A[:, :2] * B[:2, :]: 21 + 4 = 25
    # Plus A[:, 2:3] * B[2:3, :]: 2x1 * 1x7, schoolbook rank 2
    # Total: 25 + 2 = 27
    
    # But the cell finding said 39. Let me just implement the straightforward hybrid.
    
    # Let me use a simpler approach: 
    # Use Strassen for a 2x2 * 2x2 sub-problem and schoolbook for the rest.
    # Partition: take j in {0,1} for Strassen, j=2 for schoolbook.
    # For j in {0,1}: A[:, :2] (2x2) * B[:2, :] (2x7)
    #   Partition B[:2, :] into columns: we can do Strassen on 2x2 * 2x2 for each pair
    #   7 columns -> 3 pairs + 1 single
    #   Each 2x2*2x2: rank 7
    #   Each 2x2*2x1: rank 4 (schoolbook)
    #   Total: 3*7 + 4 = 25
    # For j=2: A[:, 2:3] (2x1) * B[2:3, :] (1x7): rank 2 (each of 2 rows * 7 cols, but it's outer product)
    #   Actually 2x1 * 1x7: the tensor has rank 2 (it's a sum of 2 rank-1 terms)
    # Total: 25 + 2 = 27
    
    # Let me implement this.
    
    rank = 0
    U_rows = n * m  # 6
    V_rows = m * p  # 21
    W_rows = p * n  # 14
    
    U = []
    V = []
    W = []
    
    def add_term(u_vals, v_vals, w_vals):
        """Add a rank-1 term. u_vals: dict (i,j)->val, v_vals: dict (j,k)->val, w_vals: dict (k,i)->val"""
        nonlocal rank
        u_row = [0.0] * U_rows
        v_row = [0.0] * V_rows
        w_row = [0.0] * W_rows
        for (i, j), val in u_vals.items():
            u_row[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_row[j * p + k] = val
        for (k, i), val in w_vals.items():
            w_row[k * n + i] = val
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)
        rank += 1
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2_2x7(col_offset):
        """Apply Strassen to A[:, :2] (2x2) * B[:2, col_offset:col_offset+2] (2x2)
        Returns terms for the contribution to C[:, col_offset:col_offset+2]"""
        # A is rows 0,1; cols 0,1
        # B is rows 0,1; cols col_offset, col_offset+1
        # C is rows 0,1; cols col_offset, col_offset+1
        
        terms = []
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,col_offset]+B[1,col_offset+1])
        # This is a product of (sum of two A entries) * (sum of two B entries)
        # = A[0,0]*B[0,col_offset] + A[0,0]*B[1,col_offset+1] + A[1,1]*B[0,col_offset] + A[1,1]*B[1,col_offset+1]
        # We need to express this as a rank-1 tensor term
        # U: (0,0)->1, (1,1)->1
        # V: (0,col_offset)->1, (1,col_offset+1)->1
        # W: (col_offset,0)->1, (col_offset+1,1)->1
        terms.append((
            {(0,0): 1.0, (1,1): 1.0},
            {(0,col_offset): 1.0, (1,col_offset+1): 1.0},
            {(col_offset,0): 1.0, (col_offset+1,1): 1.0}
        ))
        
        # M2 = (A[1,0]+A[1,1]) * B[0,col_offset]
        # = A[1,0]*B[0,col_offset] + A[1,1]*B[0,col_offset]
        terms.append((
            {(1,0): 1.0, (1,1): 1.0},
            {(0,col_offset): 1.0},
            {(col_offset,1): 1.0}  # Wait, this only contributes to C[1,col_offset]
        ))
        # Hmm, this is getting complicated. The W term needs to be consistent.
        # Let me reconsider.
        
        # Actually, for Strassen, each M is a scalar product of two sums.
        # The tensor decomposition term for M1:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] should equal the contribution
        # For M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1]) where k0=col_offset, k1=col_offset+1
        # This contributes to:
        # C[0,k0] += A[0,0]*B[0,k0]
        # C[0,k1] += A[0,0]*B[1,k1]  -- NO, this is wrong
        # 
        # Actually M1 is just a scalar, and it's used in the combination.
        # The issue is that in the tensor decomposition, each "rank-1" term
        # must have U, V, W such that the product U[i,j]*V[j,k]*W[k,i] gives
        # the right coefficient for A[i,j]*B[j,k] in C[i',k'].
        
        # Let me restart with a cleaner approach.
        # For the schoolbook part, it's straightforward.
        # For Strassen, I need to be more careful.
        
        # Let me just use schoolbook for everything and see if I can find a lower rank
        # through a different partitioning.
        
        return terms
    
    # Let me just implement a clean schoolbook with a different partitioning
    # and see if I can do better.
    
    # Actually, let me implement the hybrid properly.
    # The key insight: for 2x2 * 2x2, Strassen uses 7 multiplications.
    # Each multiplication corresponds to a rank-1 term in the tensor.
    
    # For 2x2 * 2x2 with A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    # M1 = (a+d)(e+h) -> contributes to C00, C01, C10, C11
    # The tensor term: U has a,d; V has e,h; W has e+h contribution
    # This is tricky because W needs to distribute.
    
    # Let me look at this differently. The matmul tensor for 2x2*2x2 has 8 terms.
    # Strassen combines them into 7.
    
    # I'll implement a known rank-7 decomposition for 2x2*2x2 and extend it.
    
    # For now, let me just return the schoolbook rank 42 and try to improve
    # with a numerical search for a lower rank.
    
    # Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
