import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    Strategy: Dictionary learning / Greedy basis pursuit for exact tensor decomposition.
    We search for a set of rank-1 tensors (atoms) with small integer entries that
    linearly combine to form the multiplication tensor.
    """
    n, m, p = DIMENSIONS
    target_shape = (n, m, p)
    # Flatten the target tensor
    target = np.zeros(target_shape)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0
    target_flat = target.flatten()
    dim_tensor = len(target_flat) # 96

    # The multiplication tensor T has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # Wait, the standard definition is:
    # (AB)_{i,k} = \sum_j A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}? No.
    # The tensor M_{n,m,p} is defined such that <M, A \otimes B \otimes C> = Tr(C^T (AB)) = \sum_{i,k} C_{i,k} (AB)_{i,k} = \sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
    # So M_{i,j,k} = 1 for all i,j,k? No, that would be the all-ones tensor.
    # Let's check the trace convention in the prompt: W[k*n+i, r].
    # The decomposition is: \sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = (AB)_{i,k} ? No.
    # The prompt says: "exactly decompose the matrix multiplication tensor".
    # Usually, this means: \sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'} is not right.
    # The standard tensor decomposition of matrix multiplication M_{n,m,p} is:
    # M_{i,j,k} = \sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # where M_{i,j,k} is the structure tensor.
    # For matrix multiplication, the structure tensor has entries:
    # M_{i,j,k} = 1 if the (i,k)-entry of the product depends on A[i,j] and B[j,k].
    # Actually, the structure tensor is defined by the bilinear form.
    # Let's use the standard definition:
    # The multiplication tensor is a 3-tensor T such that T[i,j,k] = 1 if i,j,k are valid indices?
    # No, T[i,j,k] is the coefficient of A[i,j] B[j,k] in the product C[i,k].
    # C[i,k] = \sum_j A[i,j] B[j,k].
    # So the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    # Thus, the tensor T has T[i,j,k] = 1 for all i,j,k?
    # If we contract T with A and B: \sum_{j,k} T[i,j,k] A[i,j] B[j,k] = \sum_j A[i,j] B[j,k] (if T[i,j,k]=1 for all k? No).
    # The tensor is T[i,j,k] where i is row of A, j is col of A/row of B, k is col of B.
    # The output is C[i,k].
    # The term is A[i,j] * B[j,k].
    # So for a fixed output (i,k), we sum over j.
    # The tensor decomposition represents the bilinear map.
    # The structure tensor M_{n,m,p} has entries M[i,j,k] = 1 if the pair (A[i,j], B[j,k]) contributes to C[i,k].
    # In the standard "multiplication tensor" notation used in algebraic complexity:
    # M_{n,m,p} = \sum_{i,j,k} e_i \otimes e_j \otimes e_k is NOT correct.
    # M_{n,m,p} = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p E_{i,j} \otimes E_{j,k} \otimes E_{i,k}^T ?
    # Let's stick to the interface:
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # The product \sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should equal 1 if the term A[i,j]B[j,k] is part of the computation for C[i,k]?
    # No, the sum over c gives the coefficient for the "monomial" in the bilinear expansion.
    # The bilinear map is (A, B) -> C.
    # C[i,k] = \sum_j A[i,j] B[j,k].
    # This can be written as \sum_c (\sum_j A[i,j] U_{i,j,c}) (\sum_k B[j,k] V_{j,k,c}) W_{k,i,c} ? No, indices don't match.
    
    # Let's look at the schoolbook in the prompt:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # For a specific (i,j,k), if c corresponds to (i,j,k), the product is 1.
    # If c corresponds to (i',j',k') different from (i,j,k), the product is 0.
    # So the sum is 1 if (i,j,k) is the specific term, 0 otherwise.
    # This means the tensor being decomposed is the "identity" tensor in the index space (i,j,k) -> (i,j,k).
    # i.e., T[i,j,k] = 1 for all i,j,k?
    # If T[i,j,k] = 1 for all i,j,k, then the schoolbook decomposition with 1s on the diagonal of the flattened tensor works.
    # Yes, the "multiplication tensor" in this context is the tensor that maps A and B to the *terms* of the product, or rather, it is the tensor such that its contraction with A and B yields the product.
    # Actually, the standard definition is:
    # T_{i,j,k} = 1 if the (i,k) entry of AB involves A[i,j] and B[j,k].
    # Since C[i,k] = \sum_j A[i,j] B[j,k], the term A[i,j] B[j,k] appears in C[i,k].
    # So T[i,j,k] is 1.
    # The decomposition \sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
    # Wait, the W index is k*n+i. So W depends on k and i.
    # The term is U[i,j,c] * V[j,k,c] * W[k,i,c].
    # In schoolbook, for c = (i,j,k), U=1, V=1, W=1.
    # So the sum is 1 for the specific (i,j,k) tuple.
    # This implies T[i,j,k] = 1 for all valid i,j,k.
    # So we are decomposing the all-ones tensor of shape (3,4,8)?
    # No, if T[i,j,k] = 1 for all i,j,k, then \sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for ALL i,j,k.
    # In schoolbook, for a fixed (i,j,k), only the term c=(i,j,k) contributes.
    # So yes, we are decomposing the tensor T where T[i,j,k] = 1 for all i,j,k.
    
    # Goal: Find rank r and matrices U, V, W such that:
    # \sum_{c=1}^r U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # This is equivalent to finding a low-rank decomposition of the tensor T_{n,m,p} where T[i,j,k]=1.
    
    # We will use a greedy algorithm to find a set of atoms (u_c, v_c, w_c) such that their sum equals T.
    # Atom c is defined by vectors u_c (size nm), v_c (size mp), w_c (size pn).
    # The contribution of atom c to T[i,j,k] is u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i].
    # We need \sum_c u_c[i*m+j] v_c[j*p+k] w_c[k*n+i] = 1.
    
    # We will search for atoms with small integer entries.
    # Candidate atoms: All combinations of vectors with entries in {-1, 0, 1} or {-2, -1, 0, 1, 2}?
    # Too many. 
    # Instead, we can use a "basis" approach.
    # We know that the all-ones tensor has rank at most nm*p.
    # We want to reduce this.
    
    # Let's try to find a decomposition of rank r by solving a linear system.
    # We fix a set of r "support patterns" for U, V, W.
    # A simpler heuristic: Use the fact that T is the sum of rank-1 tensors corresponding to the "diagonals" of the index space? No.
    
    # Let's use a randomized greedy algorithm (Orthogonal Matching Pursuit style) for exact recovery.
    # 1. Initialize residual R = T_flat.
    # 2. Generate a large dictionary of candidate atoms (small integer vectors).
    # 3. Iteratively pick the atom that best correlates with the residual (maximizes inner product with residual, considering the nonlinear structure? No, it's linear in the coefficients if we fix the atoms).
    # Wait, the tensor decomposition is linear in the coefficients if we view the atoms as fixed.
    # T = \sum c_a A_a.
    # We want to find c_a such that \sum c_a A_a = T.
    # This is a linear system: D c = T, where D is the matrix of atoms.
    # We want a sparse solution c (few non-zero atoms).
    # This is a Basis Pursuit problem.
    
    # Given the small dimension (96), we can try to find a basis of 96 atoms that includes T, then try to replace pairs of atoms with a single atom to reduce rank.
    
    # Algorithm:
    # 1. Start with the schoolbook basis (96 atoms).
    # 2. Try to merge atoms. If two atoms A and B can be replaced by a single atom C such that A+B = C (in the tensor space), we reduce rank by 1.
    #    A + B = C means:
    #    u_A \otimes v_A \otimes w_A + u_B \otimes v_B \otimes w_B = u_C \otimes v_C \otimes w_C
    #    This is a strong condition. It's unlikely to hold for arbitrary atoms.
    #    However, we can allow linear combinations.
    #    Actually, we are looking for a *different* set of atoms.
    
    # Let's try a direct search for rank 88-95 using random restarts of a "tensor train" or "CP" solver?
    # CP solver for exact integer solution is hard.
    
    # Alternative: Use the known result for 2x2x2 (rank 7).
    # We can embed 2x2x2 blocks.
    # 3x4x8 can be seen as:
    # A is 3x4. B is 4x8.
    # Split A into 1x4 and 2x4.
    # Split B into 4x2, 4x2, 4x2, 4x2 (four 4x2 blocks).
    # This doesn't align well with 2x2x2.
    
    # Let's try to construct a decomposition using Strassen's 7-multiplication for 2x2 blocks where possible.
    # We have 3 rows in A. We can group rows 1 and 2 into a 2x4 block. Row 3 is 1x4.
    # We have 8 cols in B. We can group cols into four 4x2 blocks? No, B is 4x8.
    # Let's group B cols into two 4x4 blocks.
    # Then we have:
    # Block 1: A[0:2, :] (2x4) * B[:, 0:4] (4x4) -> C[0:2, 0:4] (2x4)
    # Block 2: A[0:2, :] (2x4) * B[:, 4:8] (4x4) -> C[0:2, 4:8] (2x4)
    # Block 3: A[2:3, :] (1x4) * B[:, 0:4] (4x4) -> C[2:3, 0:4] (1x4)
    # Block 4: A[2:3, :] (1x4) * B[:, 4:8] (4x4) -> C[2:3, 4:8] (1x4)
    
    # For 2x4 * 4x4:
    # We can use Strassen for 2x2 * 2x2 sub-blocks.
    # 2x4 * 4x4 can be decomposed into four 2x2 * 2x2 multiplications?
    # A_2x4 = [A11 A12] where A11, A12 are 2x2.
    # B_4x4 = [[B11] [B21] [B12] [B22]]? No, B is 4x4.
    # B = [[B11, B12], [B21, B22]] where Bij are 2x2.
    # C = A*B = [A11 B11 + A12 B21, A11 B12 + A12 B22; ...]
    # This requires 4 multiplications of 2x2 matrices.
    # Each 2x2 * 2x2 can be done in 7 mults (Strassen).
    # So 4 * 7 = 28 mults for the 2x4 * 4x4 block.
    # Schoolbook for 2x4 * 4x4 is 2*4*4 = 32.
    # Savings: 4.
    
    # We have two such blocks (for rows 0-1). Total savings 8.
    # What about the 1x4 * 4x4 blocks?
    # 1x4 * 4x4 is just a linear combination of rows of B.
    # C = A_row * B.
    # A_row is 1x4. B is 4x4.
    # C = \sum_{j=0}^3 A[0,j] B[j,:].
    # This is 4 multiplications of a scalar by a 1x4 row vector.
    # In the tensor decomposition sense, this is 4 rank-1 terms?
    # U is 1x4, V is 4x4, W is 4x1.
    # The term is A[0,j] B[j,k] C[0,k].
    # This is essentially 4 independent products.
    # Can we reduce this?
    # The tensor for 1x4 * 4x4 is of rank 4?
    # T[i,j,k] where i=0, j in 0..3, k in 0..3.
    # T[0,j,k] = 1.
    # This is a matrix of all ones (4x4).
    # The rank of the all-ones matrix is 1.
    # So the tensor rank for 1x4 * 4x4 is 1?
    # Let's check.
    # We need \sum_c U[0,j,c] V[j,k,c] W[k,0,c] = 1 for all j,k.
    # If we set c=1:
    # U[0,j,1] = 1 for all j.
    # V[j,k,1] = 1 for all j,k.
    # W[k,0,1] = 1 for all k.
    # Then sum is 1*1*1 = 1.
    # So yes, the rank is 1!
    # Schoolbook for 1x4 * 4x4 is 1*4*4 = 16.
    # We can reduce it to 1.
    # Savings: 15.
    
    # We have two 1x4 * 4x4 blocks (for row 2).
    # Total savings: 15 * 2 = 30.
    
    # Total savings: 8 (from 2x4 blocks) + 30 (from 1x4 blocks) = 38.
    # New rank: 96 - 38 = 58.
    
    # Let's verify the 2x4 * 4x4 decomposition more carefully.
    # A (2x4), B (4x4), C (2x4).
    # Split A into A_L (2x2) and A_R (2x2).
    # Split B into B_TL (2x2), B_TR (2x2), B_BL (2x2), B_BR (2x2).
    # C_L = A_L B_TL + A_R B_BL.
    # C_R = A_L B_TR + A_R B_BR.
    # Each sum of two products of 2x2 matrices.
    # Product of two 2x2 matrices has rank 7.
    # Sum of two such products?
    # We need to compute X = P Q + R S.
    # This is equivalent to multiplying a 2x4 matrix [P R] by a 4x2 matrix [Q; S]?
    # Let M = [P R] (2x4) and N = [Q; S] (4x2).
    # Then M N = P Q + R S.
    # So we need the rank of 2x4 * 4x2.
    # Schoolbook: 2*4*2 = 16.
    # Can we do better?
    # 2x4 * 4x2 is equivalent to two 2x2 * 2x2 multiplications? No.
    # It is a specific tensor.
    # Let's use Strassen on 2x2 * 2x2 blocks again?
    # M (2x4) = [M1 M2], N (4x2) = [[N1] [N2]] where N1, N2 are 2x2.
    # M N = M1 N1 + M2 N2.
    # This is the same form: sum of two 2x2 products.
    # We can use the 2x2 * 2x2 Strassen decomposition for each product, but they share inputs?
    # No, M1, M2, N1, N2 are distinct.
    # However, we can view 2x4 * 4x2 as a single tensor.
    # Is there a known rank for 2x4 * 4x2?
    # It is related to 2x2 * 2x2.
    # Actually, 2x4 * 4x2 can be done in 2 * 7 = 14 mults?
    # If we compute M1 N1 using Strassen (7 mults) and M2 N2 using Strassen (7 mults), total 14.
    # Schoolbook is 16. Savings 2.
    # So for C_L = A_L B_TL + A_R B_BL, we use 14 mults.
    # For C_R = A_L B_TR + A_R B_BR, we use 14 mults.
    # Total for 2x4 * 4x4 block: 14 + 14 = 28.
    # Schoolbook: 32. Savings 4.
    
    # So the total rank calculation:
    # Block 1 (Rows 0-1, Cols 0-3): 28
    # Block 2 (Rows 0-1, Cols 4-7): 28
    # Block 3 (Row 2, Cols 0-3): 1
    # Block 4 (Row 2, Cols 4-7): 1
    # Total Rank: 28 + 28 + 1 + 1 = 58.
    
    # This is a significant improvement over 96.
    # Score: 96/58 ≈ 1.655.
    
    # Let's implement this block decomposition.
    
    def create_strassen_2x2():
        """
        Returns U, V, W for 2x2 * 2x2 multiplication with rank 7.
        U shape (4, 7), V shape (4, 7), W shape (4, 7).
        Indices:
        U: i*2 + j for i,j in 0,1
        V: j*2 + k for j,k in 0,1
        W: k*2 + i for k,i in 0,1
        """
        # Strassen's 7 mults:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices: 00=0, 01=1, 10=2, 11=3
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1 # A11, A22
        V[0, 0] = 1; V[3, 0] = 1 # B11, B22
        W[0, 0] = 1; W[3, 0] = 1 # C11, C22 (M1 contributes to C11 and C22)
        # Wait, W index is k*2+i.
        # C11 is i=0,k=0 -> idx 0.
        # C22 is i=1,k=1 -> idx 3.
        # C12 is i=0,k=1 -> idx 1.
        # C21 is i=1,k=0 -> idx 2.
        
        W[0, 0] = 1 # C11
        W[3, 0] = 1 # C22
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1 # A21, A22
        V[0, 1] = 1 # B11
        W[2, 1] = 1 # C21
        W[3, 1] = -1 # C22
        
        # M3
        U[0, 2] = 1 # A11
        V[1, 2] = 1; V[3, 2] = -1 # B12 - B22
        W[1, 2] = 1 # C12
        W[3, 2] = 1 # C22
        
        # M4
        U[3, 3] = 1 # A22
        V[2, 3] = 1; V[0, 3] = -1 # B21 - B11
        W[2, 3] = 1 # C21
        W[0, 3] = 1 # C11
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1 # A11 + A12
        V[3, 4] = 1 # B22
        W[1, 4] = 1 # C12
        W[0, 4] = -1 # C11
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1 # A21 - A11
        V[0, 5] = 1; V[1, 5] = 1 # B11 + B12
        W[3, 5] = 1 # C22
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1 # A12 - A22
        V[2, 6] = 1; V[3, 6] = 1 # B21 + B22
        W[0, 6] = 1 # C11
        W[1, 6] = -1 # C12
        
        return U, V, W

    def create_ones_1x4_4x4():
        """
        Returns U, V, W for 1x4 * 4x4 multiplication with rank 1.
        U shape (4, 1), V shape (16, 1), W shape (4, 1).
        A is 1x4 (indices 0..3).
        B is 4x4 (indices 0..15, row-major j*4+k).
        C is 1x4 (indices 0..3, k*1+0 = k).
        """
        U = np.ones((4, 1), dtype=float)
        V = np.ones((16, 1), dtype=float)
        W = np.ones((4, 1), dtype=float)
        return U, V, W

    # Construct the full decomposition
    n, m, p = 3, 4, 8
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Block 1: Rows 0-1, Cols 0-3 (2x4 * 4x4)
    # We need to map the local 2x4 * 4x4 indices to the global 3x4 * 4x8 indices.
    # This block computes C[0:2, 0:4].
    # Local A: 2x4. Global A: 3x4. Rows 0,1.
    # Local B: 4x4. Global B: 4x8. Cols 0-3.
    # Local C: 2x4. Global C: 3x8. Rows 0,1, Cols 0-3.
    
    # We decomposed 2x4 * 4x4 into two 2x4 * 4x2 parts?
    # No, we said 2x4 * 4x4 = C_L + C_R where C_L is 2x2, C_R is 2x2.
    # C_L = A_L B_TL + A_R B_BL.
    # This is a 2x2 * 2x2 + 2x2 * 2x2 sum.
    # We used the 2x4 * 4x2 trick for each.
    # Let's implement the 2x4 * 4x2 decomposition using two Strassen 2x2s.
    
    def create_2x4_4x2():
        """
        Decompose 2x4 * 4x2.
        A (2x4), B (4x2), C (2x2).
        C = A_L B_T + A_R B_B where A_L, A_R are 2x2, B_T, B_B are 2x2.
        We use Strassen for each product.
        Total rank 14.
        """
        U1, V1, W1 = create_strassen_2x2()
        U2, V2, W2 = create_strassen_2x2()
        
        U = np.zeros((8, 14), dtype=float) # 2x4 = 8
        V = np.zeros((8, 14), dtype=float) # 4x2 = 8
        W = np.zeros((4, 14), dtype=float) # 2x2 = 4
        
        # First product: A_L (cols 0,1) * B_T (rows 0,1)
        # A_L indices in global A (2x4): 0,1,4,5 (row 0 cols 0,1; row 1 cols 0,1)
        # Wait, A is 2x4. Index i*4+j.
        # A_L: i=0,1; j=0,1. Indices: 0,1,4,5.
        # A_R: i=0,1; j=2,3. Indices: 2,3,6,7.
        # B is 4x2. Index j*2+k.
        # B_T: j=0,1; k=0,1. Indices: 0,1,2,3.
        # B_B: j=2,3; k=0,1. Indices: 4,5,6,7.
        # C is 2x2. Index k*2+i.
        
        # Map Strassen U (4x7) to global U (8x14)
        # Strassen U indices: 0,1,2,3 corresponding to A11, A12, A21, A22.
        # In A_L (2x2), A11 is global A[0,0]=0, A12 is A[0,1]=1, A21 is A[1,0]=4, A22 is A[1,1]=5.
        # So map Strassen U col 0..6 to Global U col 0..6.
        # Strassen U[0] -> Global U[0]
        # Strassen U[1] -> Global U[1]
        # Strassen U[2] -> Global U[4]
        # Strassen U[3] -> Global U[5]
        
        for c in range(7):
            U[0, c] = U1[0, c]
            U[1, c] = U1[1, c]
            U[4, c] = U1[2, c]
            U[5, c] = U1[3, c]
            
            # V mapping: Strassen V indices 0,1,2,3 -> B11, B12, B21, B22.
            # In B_T (2x2), B11 is global B[0,0]=0, B12 is B[0,1]=1, B21 is B[1,0]=2, B22 is B[1,1]=3.
            V[0, c] = V1[0, c]
            V[1, c] = V1[1, c]
            V[2, c] = V1[2, c]
            V[3, c] = V1[3, c]
            
            # W mapping: Strassen W indices 0,1,2,3 -> C11, C12, C21, C22.
            # C is global C for this block (2x2).
            W[:, c] = W1[:, c]
            
        # Second product: A_R * B_B
        # A_R indices: 2,3,6,7.
        # B_B indices: 4,5,6,7.
        for c in range(7):
            U[2, c+7] = U2[0, c]
            U[3, c+7] = U2[1, c]
            U[6, c+7] = U2[2, c]
            U[7, c+7] = U2[3, c]
            
            V[4, c+7] = V2[0, c]
            V[5, c+7] = V2[1, c]
            V[6, c+7] = V2[2, c]
            V[7, c+7] = V2[3, c]
            
            W[:, c+7] = W2[:, c]
            
        return U, V, W

    # Now assemble the 2x4 * 4x4 block
    # C_L (cols 0,1) comes from 2x4 * 4x2 (B_T).
    # C_R (cols 2,3) comes from 2x4 * 4x2 (B_B).
    # Wait, my previous logic:
    # C = A*B.
    # A = [A_L A_R]. B = [[B_TL] [B_BL]]? No.
    # B is 4x4.
    # B = [[B11, B12], [B21, B22]].
    # C_L = A_L B11 + A_R B21. (Cols 0,1 of C)
    # C_R = A_L B12 + A_R B22. (Cols 2,3 of C)
    
    # So we need two 2x4 * 4x2 decompositions.
    # One for C_L: A (2x4) * B_L (4x2) where B_L is the first 2 cols of B.
    # One for C_R: A (2x4) * B_R (4x2) where B_R is the last 2 cols of B.
    
    def create_2x4_4x4():
        U1, V1, W1 = create_2x4_4x2() # For C_L
        U2, V2, W2 = create_2x4_4x2() # For C_R
        
        U = np.zeros((8, 28), dtype=float)
        V = np.zeros((16, 28), dtype=float) # B is 4x4 = 16
        W = np.zeros((8, 28), dtype=float) # C is 2x4 = 8
        
        # Map U1 (8x14) to U (8x28)
        U[:, 0:14] = U1
        U[:, 14:28] = U2
        
        # Map V1 (8x14) to V (16x28)
        # V1 corresponds to B_L (4x2, indices 0..7).
        # V2 corresponds to B_R (4x2, indices 8..15).
        V[0:8, 0:14] = V1
        V[8:16, 14:28] = V2
        
        # Map W1 (4x14) to W (8x28)
        # W1 corresponds to C_L (2x2, indices 0..3).
        # W2 corresponds to C_R (2x2, indices 4..7).
        # C index k*2+i.
        # C_L: k=0,1; i=0,1. Indices 0,1,2,3.
        # C_R: k=2,3; i=0,1. Indices 4,5,6,7.
        W[0:4, 0:14] = W1
        W[4:8, 14:28] = W2
        
        return U, V, W

    # Assemble full 3x4 * 4x8
    # U shape: 12 x R
    # V shape: 32 x R
    # W shape: 24 x R
    
    U_full = []
    V_full = []
    W_full = []
    
    # Block 1: Rows 0,1. Cols 0-3.
    U_b1, V_b1, W_b1 = create_2x4_4x4()
    # Map to global
    # U_b1 (8x28) -> U_full[0:8, :] (Rows 0,1 of A)
    # V_b1 (16x28) -> V_full[0:16, :] (Cols 0-3 of B)
    # W_b1 (8x28) -> W_full[0:8, :] (Cols 0-3, Rows 0-1 of C)
    
    U_full.append(U_b1)
    V_full.append(V_b1)
    W_full.append(W_b1)
    
    # Block 2: Rows 0,1. Cols 4-7.
    U_b2, V_b2, W_b2 = create_2x4_4x4()
    # U_b2 same as U_b1
    # V_b2 maps to V_full[16:32, :] (Cols 4-7 of B)
    # W_b2 maps to W_full[8:16, :] (Cols 4-7, Rows 0-1 of C)
    
    U_full.append(U_b2)
    V_full.append(V_b2)
    W_full.append(W_b2)
    
    # Block 3: Row 2. Cols 0-3.
    U_b3, V_b3, W_b3 = create_ones_1x4_4x4()
    # U_b3 (4x1) -> U_full[8:12, :] (Row 2 of A)
    # V_b3 (16x1) -> V_full[0:16, :] (Cols 0-3 of B)
    # W_b3 (4x1) -> W_full[16:20, :] (Cols 0-3, Row 2 of C)
    
    U_full.append(U_b3)
    V_full.append(V_b3)
    W_full.append(W_b3)
    
    # Block 4: Row 2. Cols 4-7.
    U_b4, V_b4, W_b4 = create_ones_1x4_4x4()
    # U_b4 same as U_b3
    # V_b4 maps to V_full[16:32, :]
    # W_b4 maps to W_full[20:24, :]
    
    U_full.append(U_b4)
    V_full.append(V_b4)
    W_full.append(W_b4)
    
    # Concatenate
    # U: [8x28, 8x28, 4x1, 4x1] -> 24x58
    # V: [16x28, 16x28, 16x1, 16x1] -> 32x58
    # W: [8x28, 8x28, 4x1, 4x1] -> 24x58
    
    U_final = np.hstack([
        np.vstack([U_b1, U_b2, U_b3, U_b4]),
        np.vstack([U_b2, U_b2, U_b3, U_b4]) # Wait, U is shared?
    ])
    # No, U is the same for Block 1 and Block 2?
    # Block 1 uses A[0:2, :]. Block 2 uses A[0:2, :].
    # So the U matrix for A[0:2, :] is the same.
    # But the columns of U are different for each block because the decomposition is independent?
    # Yes, the rank is the sum of the ranks.
    # So we just stack the columns.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Block 1
    U_parts.append(np.vstack([U_b1, np.zeros((4, 28))])) # Pad row 2 with 0
    V_parts.append(np.vstack([V_b1, np.zeros((16, 28))])) # Pad cols 4-7 with 0
    W_parts.append(np.vstack([W_b1, np.zeros((16, 28))])) # Pad rows 2, cols 4-7 with 0
    
    # Block 2
    U_parts.append(np.vstack([U_b2, np.zeros((4, 28))]))
    V_parts.append(np.vstack([np.zeros((16, 28)), V_b2]))
    W_parts.append(np.vstack([np.zeros((8, 28)), W_b2])) # Cols 4-7, Rows 0-1
    
    # Block 3
    U_parts.append(np.vstack([np.zeros((8, 1)), U_b3]))
    V_parts.append(np.vstack([V_b3, np.zeros((16, 1))]))
    W_parts.append(np.vstack([np.zeros((16, 1)), W_b3])) # Cols 0-3, Row 2
    
    # Block 4
    U_parts.append(np.vstack([np.zeros((8, 1)), U_b4]))
    V_parts.append(np.vstack([np.zeros((16, 1)), V_b4]))
    W_parts.append(np.vstack([np.zeros((20, 1)), W_b4])) # Cols 4-7, Row 2
    
    U_final = np.hstack(U_parts)
    V_final = np.hstack(V_parts)
    W_final = np.hstack(W_parts)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
