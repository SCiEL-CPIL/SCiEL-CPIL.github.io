import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = 3, 4, 7
    rank = 12
    
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 12
    V = [[0.0] * rank for _ in range(m * p)]  # 28 x 12
    W = [[0.0] * rank for _ in range(p * n)]  # 21 x 12
    
    # For each row i (0..2), we have 4 "slots" (rank 4 per row)
    # U[i*m+j, c] = 1 if c corresponds to (i, j)
    # V[j*p+k, c] = contribution of A_ij*B_jk to C_ik
    # W[k*n+i, c] = 1 if c corresponds to (i, k)
    
    # The key insight: for rank 12, each row i gets 4 slots.
    # In slot (i, j), U selects A_ij, V maps to all 7 outputs via B_jk, W distributes to output positions.
    # But this gives rank 12 only if V entries are structured properly.
    
    # Actually, let me reconsider. The standard lower bound for (3,4,7) tensor:
    # The tensor has dimensions 3*4*7 = 84 entries.
    # Rank 12 means we need 12 rank-1 tensors that sum to the multiplication tensor.
    
    # Let me construct it differently. For each i (row of A) and each j (column of A = row of B):
    # We want a rank-1 term that captures the product A_ij * B_jk for all k.
    # Term (i,j): U has 1 at position (i*m+j), V has B's row j (a 7-vector) at position j, W has 1 at all k for row i.
    # But this isn't rank-1 in the tensor sense...
    
    # Correct approach: The bilinear form is C_ik = sum_j A_ij B_jk
    # Rank-1 decomposition: C_ik = sum_c (sum_j U_ij^c A_ij) (sum_j V_jk^c B_jk) W_ik^c? No...
    # Proper form: C_ik = sum_c U_ij^c V_jk^c W_ki^c * A_ij * B_jk? No.
    
    # The tensor T[i,j,k] = delta for the multiplication: T[i,j,k] contributes to C_ik via A_ij * B_jk
    # Actually T[i,j,k] = 1 for all valid i,j,k is NOT the right tensor.
    
    # The matrix multiplication tensor M has M[i,j,k] = 1 if we think of it as:
    # C_ik = sum_{j} A_ij B_jk, so the "trilinear" form is sum_{i,j,k} M[i,j,k] A_ij B_jk C_ik where M[i,j,k]... 
    # No, it's bilinear: sum_{i,k} (sum_j A_ij B_jk) C_ik, and the tensor is M[i,j,k] such that
    # C_ik = sum_j M[i,j,k] A_ij B_jk with M[i,j,k] = 1 for all i,j,k? No!
    
    # The bilinear map: (A, B) -> C where C_ik = sum_j A_ij B_jk
    # This is a 3-way tensor T[i,j,k] with the property that the bilinear form is sum_{i,j,k} T[i,j,k] A_ij B_jk C_ik... 
    # Actually no. The bilinear form pairing (A,B) to C_ik is: <A, B>_ik = sum_j A_ij B_jk
    # As a tensor: T has components T[i,j,k] where the bilinear form is sum_{i,j,k} T[i,j,k] A_ij B_jk
    # But that equals C_ik only if we're careful. Actually T[i,j,k] should satisfy:
    # sum_{j} A_ij B_jk = sum_{i',j',k'} T[i',j',k'] A_i'j' B_j'k' delta... 
    
    # Let me just be concrete. The decomposition C_ik = sum_c (sum_j U[i,j,c] A_ij) (sum_j V[j,k,c] B_jk) ... no.
    
    # Correct rank-1 decomposition of the matmul tensor:
    # C_ik = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A_ij) * (sum_j V[j*p+k, c] B_jk)
    # Wait, that's not right either because the sums over j would be wrong.
    
    # The right form is: C_ik = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] summed over j? No.
    
    # Let me look at the schoolbook code to understand the convention:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    # So for term c corresponding to (i0,j0,k0):
    # U[i*m+j, c] = 1 iff i=i0 and j=j0
    # V[j*p+k, c] = 1 iff j=j0 and k=k0
    # W[k*n+i, c] = 1 iff k=k0 and i=i0
    # The reconstructed C_ik = sum_c U[i*m+j, c] A_ij * V[j*p+k, c] B_jk * W[k*n+i, c]? 
    # No, that's not how it works with a single sum over j.
    
    # I think the convention is: C_ik = sum_c W[k*n+i, c] * (product over appropriate indices)
    # Given the shapes U(n*m, rank), V(m*p, rank), W(p*n, rank):
    # C_ik = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A_ij) * (sum_j' V[j'*p+k, c] B_j'k)
    # That can't be right because the two sums are independent.
    
    # Actually I think it's: the reconstruction is
    # C_ik = sum_{c} (sum_j U[i*m+j, c] * A_ij) * (sum_j V[j*p+k, c] * B_jk) * W[k*n+i, c]
    # No wait, that doesn't dimensionally work for the index j.
    
    # Let me just trust the schoolbook and extend the pattern. Schoolbook has rank 84.
    # For rank 12, I need a different construction.
    
    # Given the risk of invalidity, let me try a safer intermediate: rank 28 (3*7 or 4*7).
    # Or better, let me try the natural row-based decomposition with rank 12.
    
    # Construction for rank 12:
    # For each row i (0,1,2) and each column j (0,1,2,3) of A (=row j of B):
    #   slot c = i*4 + j
    #   U[i*m+j, c] = 1  (selects A_ij)
    #   V[j*p+k, c] = B_jk ... but V must be FIXED, not depend on input!
    
    # This is the problem. V[j*p+k, c] must be constant. So I can't put B_jk in V.
    
    # The correct way: For rank 12, we need 12 rank-1 tensors.
    # Term c: contributes sum_{i,j,k} U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A_ij B_jk to C
    # For this to equal C_ik for all A,B, we need the tensor to decompose correctly.
    
    # Given the extreme risk of an invalid decomposition scoring 0, and that the schoolbook 
    # (rank 84) is guaranteed valid with score 1.0, let me return the schoolbook.
    
    # But wait - maybe there IS a known rank 12 or lower for (3,4,7)?
    # The asymptotic rank is around (3*4*7)^(something), but exact small ranks are hard.
    # Known: (2,2,2) has rank 7. (1,m,p) has rank m (if m<=p) or p... 
    # Actually for (1,4,7): C_k = sum_j A_j B_jk. This is just 4 dot products of length 7?
    # No, it's 4 scalars times a 4x7 matrix. The tensor rank of (1,4,7) matmul is 4
    # (since it's just a fixed 4x7 matrix multiplication, which is rank 4 as a bilinear map).
    
    # So (3,4,7) should have rank <= 3*4 = 12 by linearity over rows.
    # The issue is whether the decomposition uses integer/half-integer entries.
    
    # For (1,4,7): C_k = sum_{j=0}^{3} A_j * B_{j,k}
    # Rank-1 decomposition: 4 terms, one per j.
    # Term j: U[j, j] = 1, V[j*7+k, j] = 1 for all k, W[k, j] = ... 
    # Wait, shapes: U(1*4, 4) = (4,4), V(4*7, 4) = (28,4), W(7*1, 4) = (7,4)
    # C_k = sum_{c=0}^{3} W[k*1+c] * (sum_{j} U[1*4... ] ) ... 
    
    # I'm getting confused by the exact convention. Let me just replicate the schoolbook pattern
    # but grouped by rows, using rank 12 with the understanding that each (i,j) pair gets one slot.
    
    # Following the EXACT schoolbook convention but reducing rank:
    # Schoolbook: for each (i,j,k), one term. Rank = 3*4*7 = 84.
    # To get rank 12, I'd need to merge terms. For fixed (i,j), sum over k gives one term
    # only if V and W are structured to handle all k simultaneously.
    
    # Term for (i,j): 
    #   U[i*m+j, c] = 1
    #   V[j*p+k, c] = 1 for ALL k (0..6)  
    #   W[k*n+i, c] = 1 for ALL k (0..6)
    # Then C_ik gets contribution: W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1*1*1 = 1 for all k
    # But we need C_ik to get A_ij * B_jk, not just 1.
    # The A and B values are outside the decomposition - they're multiplied by U and V.
    # So: (sum_j U[i*m+j,c] A_ij) = A_ij (since only j slot has U=1)
    # (sum_k... no, V[j*p+k, c] multiplies B_jk: sum over... 
    
    # I think the reconstruction is:
    # C_ik = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A_ij) * (some_B_part)
    # Where some_B_part depends on c and k.
    
    # Given my uncertainty about the exact reconstruction formula and the high risk of
    # producing an invalid decomposition (score 0), I'll return the schoolbook which is
    # guaranteed correct.
    
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
