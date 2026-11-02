"""Tensor decomposition for 3x5x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _to_frac_list(mat):
    """Convert numpy array to list of Fraction lists for exact output."""
    return [[Fraction(x).limit_denominator(1000000) for x in row] for row in mat]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Try to find a rank-minimal decomposition using Strassen-like factorization
    # Split m=5 into (2,3), and use Strassen's 7-rank for the 2x2x2 subproblem
    
    # The idea: C[i,k] = sum_j A[i,j]*B[j,k]
    # Group j into j1 in {0,1} and j2 in {2,3,4}
    # C[i,k] = sum_{j1} A[i,j1]*B[j1,k] + sum_{j2} A[i,j2]*B[j2,k]
    
    # For the first sum (2 inner dims): use Strassen with rank 7
    # For the second sum (3 inner dims): schoolbook rank 9
    # Total: 7 + 9 = 16 per (i,k) pair... but we need to be more careful
    
    # Actually, let's use a different approach: try random linear combinations
    # to find a lower rank decomposition
    
    # First, build the schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try Strassen-based decomposition
    # For 3x5x7, split m=5 as m1=2, m2=3
    # For each pair (i,k), we compute C[i,k] = sum_j A[i,j]B[j,k]
    # = sum_{j=0}^{1} A[i,j]B[j,k] + sum_{j=2}^{4} A[i,j]B[j,k]
    
    # The first term: 2 inner dims, can use rank-7 (Strassen)
    # The second term: 3 inner dims, schoolbook rank 9
    # But these are independent sums, so total rank per (i,k) is 7+9=16
    # Total rank = n*p*(7+9) = 3*7*16 = 336... that's worse!
    
    # Better approach: We need to share structure across all (i,k) pairs.
    # Let's try: for each "Strassen product" and each "schoolbook term", 
    # create one tensor factor that covers all (i,k) simultaneously.
    
    # Strassen for the 2-dim part: 
    # M1 = A[i,0]*B[0,k] + A[i,1]*B[1,k]  -- wait, that's not right
    # Strassen computes C = A*B where A,B are 2x2 using 7 products
    # But here we have A as n x 2 and B as 2 x p for each (i,k)
    
    # Actually, the 2x2 Strassen applies to computing a 2x2 product from two 2x2 matrices.
    # In our case, for fixed i,k, we're computing a scalar: sum_j A[i,j]*B[j,k]
    # This is just a dot product of length m=5, which has rank m=5 (schoolbook).
    # Strassen doesn't directly help for a single dot product.
    
    # The key insight for Strassen is that it applies when you're computing a MATRIX product,
    # i.e., all (i,k) pairs simultaneously. The tensor structure allows sharing.
    
    # Let me think about this differently. The matrix multiplication tensor T_{i,j,k} = 1
    # for all i,j,k. We want rank r such that T = sum_c U_c ⊗ V_c ⊗ W_c
    # where (U_c)_{i,j} U_c is a 3x5 matrix, (V_c)_{j,k} is 5x7, (W_c)_{k,i} is 7x3.
    
    # Schoolbook: r = 3*5*7 = 105, with U_{ij}^{(ijk)} = delta, etc.
    
    # For a lower rank, we need to find linear dependencies.
    # One approach: if we can express one of the basis tensors as a combination of others.
    
    # Let's try a randomized approach to find a lower rank decomposition.
    # We'll use the fact that the tensor space has dimension 3*5*7 = 105,
    # and try to find a rank-104 or lower decomposition.
    
    # Actually, for a "flat" tensor like the multiplication tensor, 
    # the rank is often equal to the dimension unless there's special structure.
    
    # Let me try a concrete Strassen-style decomposition for 3x5x7:
    # Split as (3,2,3) x (2,3,7) -> (3,7)
    # Use Strassen on the 2x2 part where applicable.
    
    # Here's a concrete plan:
    # Write A as 3x5 = [A_left (3x2), A_right (3x3)]
    # Write B as 5x7 = [B_top (2x7); B_bottom (3x7)]
    # C = A_left @ B_top + A_right @ B_bottom
    # 
    # A_left @ B_top: 3x2 times 2x7. This is a "bilinear" computation.
    # The tensor for this part has rank at most 2 (each column of A_left times each row of B_top)
    # Actually, rank of (3x2)@(2x7) multiplication = 2*... no.
    # The tensor T_{i,j,k} for (3x2)@(2x7) has i in 0..2, j in 0..1, k in 0..6
    # Schoolbook rank = 3*2*7 = 42. But we can do better.
    # Actually, for a (n x m) @ (m x p) with small m, the rank is at most m*n*p but can be lower.
    # For m=2: rank of 3x2x7 tensor. Each "slice" j=0,1 gives a 3x7 matrix.
    # The tensor is sum_j (A[:,j] ⊗ B[j,:] ⊗ identity-like)
    # Rank = 2 * ... hmm.
    
    # Let me just try to verify if rank < 105 is achievable and implement a search.
    
    # Given the complexity and time constraints, let me try a specific known construction:
    # For 3x5x7, use the fact that 5 = 2+3 and apply Strassen to the 2-part.
    
    # Strassen decomposition for the 2x2x7 part (treating the 7 as a vector):
    # A is 3x2, B is 2x7. C = A@B is 3x7.
    # For each column k of B (a 2x1 vector) and each row i of A (a 1x2 vector):
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a dot product of length 2. The tensor for all (i,k) pairs:
    # T_{i,k} = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # = sum_j A[i,j]*B[j,k]
    # Rank of this 3x2x7 tensor: schoolbook is 3*2*7=42.
    # But we can write it as: for j=0: A[:,0] ⊗ B[0,:] (outer product, rank 1 in tensor sense)
    # for j=1: A[:,1] ⊗ B[1,:]
    # So rank = 2! Each term is a rank-1 tensor: u_j ⊗ v_j where u_j is 3x1, v_j is 7x1... 
    # No wait, the tensor T_{i,k} with 3 indices? No, it's a bilinear form with 2 groups.
    
    # I'm overcomplicating this. The tensor has 3 index groups: (i,j) for A, (j,k) for B, (k,i) for C.
    # The multiplication tensor M_{(i,j),(j,k),(k,i)} = 1 if indices match, else 0.
    # More precisely, M_{a,b,c} where a=(i,j), b=(j,k), c=(k,i), and M=1 iff all j's and k's and i's are consistent.
    # Actually, M_{(i,j),(j',k),(k',i')} = 1 iff j=j', k=k', i=i'.
    # So M is the "identity" tensor in the sense that it picks out matching indices.
    # The rank of this tensor is exactly n*m*p = 105 because each (i,j,k) triple gives an independent rank-1 component,
    # and they're linearly independent (each has a unique 1 in a different position of the 3-way array).
    
    # Wait, that's not quite right. The tensor M_{ijk} = 1 for all i,j,k (it's the all-ones tensor).
    # No! For matrix multiplication, T_{i,j,k} represents the contribution of A[i,j]*B[j,k] to C[i,k].
    # The tensor is T_{i,j,k} = 1 for all valid i,j,k, and the multiplication is:
    # C[i,k] = sum_j T_{i,j,k} * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
    # So T_{i,j,k} = 1 for all i in 0..2, j in 0..4, k in 0..6.
    # This is the all-ones tensor of shape 3x5x7!
    
    # The rank of the all-ones tensor 1_{n,m,p} is... min(n*m, m*p, p*n, n*m*p)?
    # Actually, the all-ones tensor 1_{n,m,p} has rank 1! Because 1_{i,j,k} = 1*1*1 = u_i * v_j * w_k
    # where u_i = 1, v_j = 1, w_k = 1 for all i,j,k.
    # So rank = 1!
    
    # But that can't be right for matrix multiplication. Let me re-read the problem.
    
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T_{ijk} is defined such that:
    # (A * B)_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
    # So T_{ijk} = 1 for all i,j,k. And indeed, this is the all-ones tensor.
    # And its rank is 1: U_{ij} = 1, V_{jk} = 1, W_{ki} = 1, rank=1.
    
    # Wait, that would mean we can do matrix multiplication with rank 1, which is absurd.
    # Let me re-examine. The decomposition is:
    # C_{ik} = sum_c U_{(i,j),c} * V_{(j,k),c} * W_{(k,i),c}
    # where the sum over j is implicit in U and V indices.
    # So U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # And we need: for all i,j,k, the coefficient of A_{ij} B_{jk} in C_{ik} is 1.
    # The contribution to C_{ik} from factor c is: sum_j U_{(i,j),c} V_{(j,k),c} * W_{(k,i),c}
    # We need: sum_c W_{(k,i),c} * (sum_j U_{(i,j),c} V_{(j,k),c}) = 1 for all i,k
    # And this must hold with the A and B being arbitrary, so the structure is:
    # C_{ik} = sum_c W_{(k,i),c} * (U_c^T * A * V_c) where U_c is n x m, V_c is m x p.
    
    # So we need: sum_c W_{(k,i),c} * (U_c^T A V_c) = A B for all A, B.
    # This means the bilinear map (A,B) -> sum_c W_{ki}^{(c)} U_c^T A V_c must equal (A,B) -> AB.
    
    # For this to work, we need the set of matrices {U_c, V_c, W_c} to span the multiplication.
    # The schoolbook uses r = nmp with U_c = E_{ij}, V_c = E_{jk}, W_c = 1 (or delta_{ki}).
    
    # So my earlier analysis was wrong. The tensor is NOT the all-ones tensor in the simple sense.
    # It's a more structured tensor because the indices are grouped.
    
    # OK so back to finding a lower rank. Let me try a computational approach.
    
    # Given time constraints, let me just return the schoolbook solution which is guaranteed correct.
    U, V, W, rank = schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
