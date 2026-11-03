import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 hierarchical Strassen decomposition
    # Split the m=4 dimension into two pairs (j=0,1 and j=2,3)
    # For each pair, perform a 2x2x8 product using Strassen-like combination
    # along the p=8 dimension, treating it as 4 blocks of 2x2x2.
    
    # The key insight: 2x4x8 can be decomposed by viewing it as two 2x2x8 products.
    # Each 2x2x8 has rank at most 7*4 = 28 if we use Strassen on the 2x2 part 
    # and 4 separate 2x2x2 Strassen instances for the p blocks... 
    # Actually, let's use a known rank-28 decomposition structure.
    
    # Rank 28 for 2x4x8: Use 4 blocks of Strassen 2x2x2 with sharing.
    # Structure: 
    #   C = A B where A is 2x4, B is 4x8
    #   Split A into A1 (2x2, rows j=0,1) and A2 (2x2, rows j=2,3)
    #   Split B into B1 (2x8, cols j=0,1) and B2 (2x8, cols j=2,3)
    #   C = A1*B1 + A2*B2
    #   
    #   Each 2x2x8 product can be computed with rank 14 using a 
    #   "Strassen in time" approach: split the 8 columns into 4 pairs,
    #   compute 4 2x2x2 Strassen products (each rank 7), but with 
    #   linear combinations that allow sharing? No, 4*7=28 per 2x2x8,
    #   so total would be 56.
    #
    #   To get rank 28 total, we need a different grouping.
    #   Group by i (output rows): C has 2*8 = 16 outputs.
    #   Alternatively, use the fact that rank(2,4,8) = 28 is known.
    #
    #   Let me construct it explicitly using 28 scalar multiplications.
    
    rank = 28
    U = np.zeros((n * m, rank))  # 8 x 28
    V = np.zeros((m * p, rank))  # 32 x 28
    W = np.zeros((p * n, rank))  # 16 x 28
    
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # i in 0..1, j in 0..3, k in 0..7
    
    # We'll define 28 terms. Each term c in 0..27 has:
    #   U[i*m+j][c] = a coefficient
    #   V[j*p+k][c] = a coefficient  
    #   W[k*n+i][c] = a coefficient
    # such that sum_c U[i*m+j][c] * V[j'*p+k][c] = delta(i,i')... 
    # Actually the tensor is: T[i,j,k] = product over c of U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Wait no: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = A[i,j] * B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]? 
    # No. Standard: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] where W is for the output.
    # The convention says W[k*n+i, r]. So the output C[i,k] = sum_j sum_r A[i,j]*B[j,k]*... 
    # Actually: C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[j,r]) (sum_j B[j,k] V[r,j])... 
    # Let me re-read: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]? No that's not right dimensionally.
    # 
    # Standard matrix multiplication tensor decomposition:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}? No.
    # 
    # The matrix multiplication tensor is M_{i,j,k} = A_{i,j} B_{j,k}.
    # A decomposition is: M_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r where W is used for output indexing.
    # Given the interface: U is (n*m, rank) so U[i*m+j, r], V is (m*p, rank) so V[j*p+k, r],
    # W is (p*n, rank) so W[k*n+i, r].
    # Then C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # For this to work: sum_j U[i*m+j, r] * A[i,j] and sum_j V[j*p+k, r] * B[j,k] are the "contracted" parts.
    # Actually the correct form is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) (sum_j B[j,k] V[j*p+k, r]) W[k*n+i, r]?
    # No, that's not standard. Let me think again.
    
    # The bilinear form: C[i,k] = sum_{j} A[i,j] * B[j,k]
    # In tensor decomposition: T[i,j,k] = A[i,j] * B[j,k]
    # = sum_r U[i,j,r] * V[j,k,r]
    # But we have three matrices. The convention with W[k*n+i, r] suggests:
    # C[i,k] = sum_r W[k*n+i, r] * <U_r, A_row_i> * <V_r, B_col_k>
    # where <U_r, A_row_i> = sum_j U[i*m+j, r] * A[i,j]
    # and <V_r, B_col_k> = sum_j V[j*p+k, r] * B[j,k]
    
    # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    # For this to equal sum_j A[i,j] B[j,k] for all A, B:
    # sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r] = delta_{j,j'} ... no.
    # We need: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_{j'} V[j'*p+k, r] B[j',k])
    # = sum_{j,j'} A[i,j] B[j',k] (sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r])
    # This should equal sum_j A[i,j] B[j,k]
    # So: sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r] = delta_{j,j'} (and independent of i,k)
    
    # This is the condition. Let's construct 28 such terms.
    
    # I'll use a known rank-28 construction for 2x4x8.
    # Based on grouping: split j into {0,1} and {2,3}.
    # For the first group (j=0,1): compute S1[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # For the second group (j=2,3): compute S2[i,k] = A[i,2]B[2,k] + A[i,3]B[3,k]
    # C[i,k] = S1[i,k] + S2[i,k]
    
    # Each 2x2x8 product can be done in rank 14 using the following:
    # Split k into 4 pairs: {0,1}, {2,3}, {4,5}, {6,7}
    # For each pair, do a 2x2x2 Strassen (rank 7)? That gives 28 per 2x2x8, total 56.
    # 
    # To get 14 per 2x2x8, we need a rank-14 decomposition of 2x2x8.
    # 2x2x8 has rank at most 7*2=14? No, rank(2,2,8) = 7*4=28? 
    # Actually rank(2,2,m) for m even: use Strassen recursively.
    # 2x2x8 = 2x2x(2*4) = 2x2x(4*2). 
    # rank(2,2,4) = 7*2 = 14? Or is it 14?
    # 2x2x4: split the 4 into two 2s. 2x2x4 = 2x2x2 + 2x2x2 shifted.
    # Rank(2,2,4) = 14 (using two Strassen 2x2x2).
    # So rank(2,2,8) = 28 (using four Strassen 2x2x2).
    # Total for 2x4x8 = 2 * 28 = 56.
    
    # Hmm, so rank 28 might not be achievable with this simple grouping.
    # Let me try a different approach: interleave the groups.
    
    # Alternative: Use all 4 j values together with 7 Strassen-like combinations
    # per "block" of k, with 4 k-blocks = 28.
    
    # Let's define 7 combinations for the j dimension (like Strassen 2x2 but for 4 inputs)
    # and 4 k-blocks.
    
    # Actually, let me just implement the 56-rank solution which is definitely correct,
    # and see if I can optimize.
    
    # 56-rank solution:
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 terms
    strassen = [
        # (U_coeffs for i,j), (V_coeffs for j,k), (W_coeff for k,i)
        # Standard Strassen for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
    ]
    
    # For 2x2x2 Strassen:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B10], [B01, B11]] (note: B[j,k] so B00=B[0,0], B10=B[1,0])
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # Let me encode Strassen terms for 2x2x2:
    # Term t: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    
    def add_strassen_222(U, V, W, i_offset, j_offset, k_offset, t_offset):
        """Add 7 Strassen terms for a 2x2x2 block."""
        # i in {i_offset, i_offset+1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        # A indices: i0=i_offset, i1=i_offset+1; j0=j_offset, j1=j_offset+1
        # B indices: j0, j1 for rows; k0, k1 for cols
        # C indices: i0,k0; i0,k1; i1,k0; i1,k1
        
        # M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        c = t_offset + 0
        U[i_offset * m + j_offset, c] += 1
        U[(i_offset+1) * m + (j_offset+1), c] += 1
        V[j_offset * p + k_offset, c] += 1
        V[(j_offset+1) * p + (k_offset+1), c] += 1
        W[k_offset * n + i_offset, c] += 1
        W[(k_offset+1) * n + (i_offset+1), c] += 1
        W[k_offset * n + (i_offset+1), c] -= 1
        W[(k_offset+1) * n + i_offset, c] -= 1
        
        # M2 = (A[i1,j0]+A[i1,j1]) B[j0,k0]
        c = t_offset + 1
        U[(i_offset+1) * m + j_offset, c] += 1
        U[(i_offset+1) * m + (j_offset+1), c] += 1
        V[j_offset * p + k_offset, c] += 1
        W[k_offset * n + (i_offset+1), c] += 1
        
        # M3 = A[i0,j0] (B[j1,k0]-B[j1,k1])
        c = t_offset + 2
        U[i_offset * m + j_offset, c] += 1
        V[(j_offset+1) * p + k_offset, c] += 1
        V[(j_offset+1) * p + (k_offset+1), c] -= 1
        W[k_offset * n + i_offset, c] += 1
        W[(k_offset+1) * n + i_offset, c] -= 1
        
        # M4 = A[i1,j1] (B[j0,k1]-B[j0,k0])
        c = t_offset + 3
        U[(i_offset+1) * m + (j_offset+1), c] += 1
        V[j_offset * p + (k_offset+1), c] += 1
        V[j_offset * p + k_offset, c] -= 1
        W[(k_offset+1) * n + (i_offset+1), c] += 1
        W[k_offset * n + (i_offset+1), c] -= 1
        
        # M5 = (A[i0,j0]+A[i0,j1]) B[j1,k1]
        c = t_offset + 4
        U[i_offset * m + j_offset, c] += 1
        U[i_offset * m + (j_offset+1), c] += 1
        V[(j_offset+1) * p + (k_offset+1), c] += 1
        W[(k_offset+1) * n + i_offset, c] += 1
        
        # M6 = (A[i1,j0]-A[i0,j0]) (B[j0,k0]+B[j1,k0])
        c = t_offset + 5
        U[(i_offset+1) * m + j_offset, c] += 1
        U[i_offset * m + j_offset, c] -= 1
        V[j_offset * p + k_offset, c] += 1
        V[(j_offset+1) * p + k_offset, c] += 1
        W[k_offset * n + (i_offset+1), c] += 1
        W[k_offset * n + i_offset, c] -= 1
        
        # M7 = (A[i0,j1]-A[i1,j1]) (B[j0,k1]+B[j1,k1])
        c = t_offset + 6
        U[i_offset * m + (j_offset+1), c] += 1
        U[(i_offset+1) * m + (j_offset+1), c] -= 1
        V[j_offset * p + (k_offset+1), c] += 1
        V[(j_offset+1) * p + (k_offset+1), c] += 1
        W[(k_offset+1) * n + i_offset, c] += 1
        W[(k_offset+1) * n + (i_offset+1), c] -= 1
    
    # Now build the 56-rank solution:
    # Split j into {0,1} and {2,3}
    # For j-group 0 (j=0,1): 4 k-blocks of size 2, each Strassen 2x2x2 -> 28 terms
    # For j-group 1 (j=2,3): 4 k-blocks of size 2, each Strassen 2x2x2 -> 28 terms
    
    t = 0
    # j-group 0: j_offset=0
    for kb in range(4):
        k_offset = kb * 2
        add_strassen_222(U, V, W, 0, 0, k_offset, t)
        t += 7
    
    # j-group 1: j_offset=2
    for kb in range(4):
        k_offset = kb * 2
        add_strassen_222(U, V, W, 0, 2, k_offset, t)
        t += 7
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
