import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # We will build the rank-98 decomposition.
    # It consists of two rank-49 decompositions for the two 4x4 blocks of B.
    
    # Helper to generate rank-49 decomposition for 4x4 * 4x4 -> 4x4
    # Returns list of (u_indices, v_indices, w_indices, coeffs)
    # u_indices: dict mapping index in A (0-15) to coeff
    # v_indices: dict mapping index in B (0-15) to coeff
    # w_indices: dict mapping index in C (0-15) to coeff
    
    def strassen_2x2_blocks(A_blocks, B_blocks):
        # A_blocks: list of 4 dicts, each mapping sub-index to coeff
        # Actually, let's represent the linear forms directly.
        # A 2x2 matrix of linear forms.
        # A[0] is top-left, A[1] top-right, A[2] bottom-left, A[3] bottom-right
        pass

    # Let's represent the entries of A, B, C as indices.
    # A is 4x4, index i*4+j
    # B is 4x4, index j*4+k
    # C is 4x4, index i*4+k
    
    # Level 1: 4x4 decomposition into 2x2 blocks of 2x2 matrices.
    # A blocks:
    # A11: indices (0,1,4,5) -> local (0,1,2,3)
    # A12: indices (2,3,6,7) -> local (0,1,2,3)
    # A21: indices (8,9,12,13) -> local (0,1,2,3)
    # A22: indices (10,11,14,15) -> local (0,1,2,3)
    
    # We define the 7 Strassen combinations at the block level.
    # Each combination is a pair of linear forms of the 4 blocks of A and 4 blocks of B.
    # But the blocks themselves are 2x2 matrices.
    # So we need to expand these into scalar multiplications.
    
    # Let's define the 7 "super" multiplications M1..M7 for 4x4.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each M_k is a product of two 2x2 matrices (which are sums of 2x2 blocks of A and B).
    # Let X_k be the A-side matrix and Y_k be the B-side matrix for M_k.
    # X_k and Y_k are 2x2 matrices where each entry is a 2x2 matrix (block of A or B).
    # So X_k is effectively a 4x4 matrix of coefficients for the entries of A.
    # Similarly for Y_k.
    
    # We can represent X_k as a 4x4 matrix of indices into A (or 0 if not present).
    # Actually, it's a linear combination. So a 4x4 matrix of coefficients.
    
    def get_block_indices(block_idx):
        # block_idx 0: A11, 1: A12, 2: A21, 3: A22
        # Returns 4 indices into the 16-entry vector of A
        base = [0, 2, 8, 10]
        offsets = [0, 1, 4, 5]
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        if block_idx == 0: return [0, 1, 4, 5]
        if block_idx == 1: return [2, 3, 6, 7]
        if block_idx == 2: return [8, 9, 12, 13]
        if block_idx == 3: return [10, 11, 14, 15]

    # Define the 7 combinations as lists of (block_idx, coeff) for A and B
    # M1: A: [(0,1), (3,1)], B: [(0,1), (3,1)]
    # M2: A: [(2,1), (3,1)], B: [(0,1)]
    # M3: A: [(0,1)], B: [(1,1), (3,-1)]
    # M4: A: [(3,1)], B: [(2,1), (0,-1)]
    # M5: A: [(0,1), (1,1)], B: [(3,1)]
    # M6: A: [(2,1), (0,-1)], B: [(0,1), (1,1)]
    # M7: A: [(1,1), (3,-1)], B: [(2,1), (3,1)]
    
    super_mults = [
        ([(0,1), (3,1)], [(0,1), (3,1)]),
        ([(2,1), (3,1)], [(0,1)]),
        ([(0,1)], [(1,1), (3,-1)]),
        ([(3,1)], [(2,1), (0,-1)]),
        ([(0,1), (1,1)], [(3,1)]),
        ([(2,1), (0,-1)], [(0,1), (1,1)]),
        ([(1,1), (3,-1)], [(2,1), (3,1)])
    ]
    
    # For each super multiplication, we have two 2x2 matrices X and Y.
    # X is formed by summing the blocks of A.
    # Y is formed by summing the blocks of B.
    # X and Y are 2x2 matrices of 2x2 blocks.
    # So X is a 4x4 matrix where each element is a linear combination of A's entries.
    # Let's construct the 4x4 coefficient matrices for X and Y.
    
    def build_4x4_coeff_matrix(block_coeffs):
        # block_coeffs: list of (block_idx, coeff)
        # Returns a 4x4 numpy array of coefficients for A's 16 entries
        coeff_mat = np.zeros((4, 4, 16)) # 4x4 grid of coeffs for each of 16 entries? No.
        # We want a 4x4 matrix where entry (r,c) is a linear form.
        # Let's just create a 16x4 matrix? No.
        # Let's create a list of 16 linear forms? No.
        # We want to represent X as a 4x4 matrix of linear forms.
        # Each linear form is a vector of 16 coefficients.
        # So X is a 4x4x16 array? Or 4x4 lists of dicts.
        
        # Let's use a 4x4 list of dicts {a_index: coeff}
        X = [[{} for _ in range(4)] for _ in range(4)]
        
        for block_idx, coeff in block_coeffs:
            indices = get_block_indices(block_idx)
            # The block is 2x2.
            # Block 0 (A11) corresponds to local rows 0,1 and cols 0,1.
            # Block 1 (A12) corresponds to local rows 0,1 and cols 2,3.
            # Block 2 (A21) corresponds to local rows 2,3 and cols 0,1.
            # Block 3 (A22) corresponds to local rows 2,3 and cols 2,3.
            
            r_start = (block_idx // 2) * 2
            c_start = (block_idx % 2) * 2
            
            for dr in range(2):
                for dc in range(2):
                    r = r_start + dr
                    c = c_start + dc
                    a_idx = indices[dr*2 + dc]
                    if coeff != 0:
                        X[r][c][a_idx] = X[r][c].get(a_idx, 0) + coeff
                        
        return X

    # Now, for each super multiplication k, we have X_k and Y_k.
    # X_k is a 4x4 matrix of linear forms on A.
    # Y_k is a 4x4 matrix of linear forms on B.
    # We need to decompose the product Z_k = X_k Y_k into scalar multiplications.
    # Z_k is a 4x4 matrix of linear forms on C.
    # We use Strassen's algorithm on X_k and Y_k.
    # X_k has 2x2 blocks of linear forms.
    # Let X_k[0,0] be the top-left 2x2 block of linear forms.
    # This block is itself a 2x2 matrix of linear forms? No.
    # X_k is 4x4. We split it into four 2x2 blocks.
    # X_k11 = X_k[0:2, 0:2]
    # X_k12 = X_k[0:2, 2:4]
    # X_k21 = X_k[2:4, 0:2]
    # X_k22 = X_k[2:4, 2:4]
    
    # Each of these is a 2x2 matrix of linear forms.
    # To multiply two 2x2 matrices of linear forms using Strassen,
    # we treat the linear forms as "scalars" that can be added and multiplied?
    # No, the "scalars" here are linear forms.
    # Strassen works on the coefficients.
    # If we have matrices of coefficients, Strassen reduces the number of multiplications of coefficients.
    # Here, the "coefficients" are the actual scalar entries of A and B.
    # So we apply Strassen to the 4x4 coefficient matrices?
    # No, X_k is a 4x4 matrix where each entry is a linear form.
    # This is equivalent to a 4x4 matrix of vectors.
    # We can't just apply Strassen to the 4x4 matrix of vectors directly in the standard way
    # because Strassen assumes the entries are from a field (scalars).
    # However, linearity allows us to distribute.
    # The product of two linear forms is a bilinear form.
    # We want to decompose the bilinear form into rank-1 terms.
    
    # Alternative view:
    # The total rank-49 decomposition is the tensor product of the rank-7 Strassen decomposition
    # applied at the block level and the rank-7 Strassen decomposition applied at the scalar level?
    # Not exactly.
    
    # Let's go back to the explicit expansion.
    # We have 7 super multiplications.
    # Each super multiplication M_k = X_k Y_k.
    # X_k is a 4x4 matrix of linear forms.
    # Y_k is a 4x4 matrix of linear forms.
    # Z_k = X_k Y_k.
    # We need to decompose Z_k into scalar multiplications.
    # Z_k[r,c] = sum_s X_k[r,s] Y_k[s,c].
    # X_k[r,s] is a linear form in A. Y_k[s,c] is a linear form in B.
    # The product is a bilinear form.
    # We can decompose this bilinear form using Strassen on the 4x4 structure?
    # Yes, because the "entries" of X_k and Y_k are linear forms, which behave like scalars
    # with respect to addition, but their multiplication is the scalar multiplication we want to count.
    # So we can apply the rank-7 Strassen decomposition to the 4x4 matrices X_k and Y_k,
    # where the "scalar" multiplications are the products of the linear forms.
    # But wait, if we apply Strassen to X_k and Y_k, we get 7 products of linear forms.
    # Each product of linear forms (sum a_i A_i) * (sum b_j B_j) is a sum of a_i b_j A_i B_j.
    # This is a sum of scalar multiplications.
    # So each "Strassen step" at this level generates 7 scalar multiplications?
    # No.
    # Let's trace carefully.
    # Level 1 (4x4 blocks): 7 multiplications of 2x2 block matrices.
    # Level 2 (2x2 scalars): Each 2x2 block matrix multiplication is decomposed into 7 scalar multiplications.
    # So total 7 * 7 = 49 scalar multiplications.
    
    # So for each of the 7 super multiplications, we apply Strassen to the 4x4 coefficient matrices?
    # No, X_k is not a 4x4 matrix of scalars. It is a 4x4 matrix of linear forms.
    # But the structure is the same.
    # We can represent X_k as a 4x4 matrix of vectors (coefficients).
    # We apply the Strassen algorithm to the 4x4 matrix of vectors.
    # The "addition" of vectors is component-wise.
    # The "multiplication" of vectors is... wait.
    # Strassen multiplies entries. Here entries are vectors.
    # We don't multiply vectors. We multiply the *original scalars* that form the vectors.
    # This means we must expand the vector entries into their scalar components before applying Strassen?
    # No, that would be schoolbook.
    
    # Correct logic:
    # The tensor is $T = \sum_{i,j,k} A_{ij} B_{jk} C_{ki}$.
    # We are finding a decomposition $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
    # The rank-49 decomposition for $4 \times 4 \times 4$ is known.
    # It can be constructed by taking the tensor product of the rank-7 Strassen tensor for $2 \times 2$
    # with itself?
    # $T_{4 \times 4} = T_{2 \times 2} \otimes T_{2 \times 2}$?
    # The rank of $T \otimes T'$ is rank(T) * rank(T').
    # So rank 7 * 7 = 49.
    # Yes! The rank-49 decomposition is the Kronecker product of the rank-7 Strassen decomposition.
    
    # So we just need the rank-7 decomposition for $2 \times 2 \times 2 \times 2$ and take its Kronecker product with itself.
    
    # Rank-7 Strassen decomposition for 2x2:
    # U, V, W are 4x7 matrices.
    # We define them explicitly.
    
    # Indices for 2x2: 0,1,2,3
    # A: 0 1 / 2 3
    # B: 0 1 / 2 3
    # C: 0 1 / 2 3
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's build the 4x7 U, V, W for 2x2.
    # U[i, r] is coeff of A_i in r-th mult.
    # V[j, r] is coeff of B_j in r-th mult.
    # W[k, r] is coeff of C_k in r-th mult.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0 (M1)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1 (M2)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # r=2 (M3)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # r=3 (M4)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # r=4 (M5)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # r=5 (M6)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6 (M7)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify C00: M1+M4-M5+M7 -> W2[0,:] = [1, 0, 0, 1, -1, 0, 1]
    # My W2[0,0]=1, W2[0,3]=1, W2[0,4]=-1, W2[0,6]=1. Correct.
    # C01: M3+M5 -> W2[1,:] = [0, 0, 1, 0, 1, 0, 0]
    # My W2[1,2]=1, W2[1,4]=1. Correct.
    # C10: M2+M4 -> W2[2,:] = [0, 1, 0, 1, 0, 0, 0]
    # My W2[2,1]=1, W2[2,3]=1. Correct.
    # C11: M1-M2+M3+M6 -> W2[3,:] = [1, -1, 1, 0, 0, 1, 0]
    # My W2[3,0]=1, W2[3,1]=-1, W2[3,2]=1, W2[3,5]=1. Correct.
    
    # Now, for 4x4, we take Kronecker product.
    # U4 = U2 ⊗ U2
    # V4 = V2 ⊗ V2
    # W4 = W2 ⊗ W2
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # This gives a rank-49 decomposition for 4x4 * 4x4 -> 4x4.
    # U4 is 16x49. V4 is 16x49. W4 is 16x49.
    
    # Now we need to handle 4x4 * 4x8 -> 4x8.
    # We split B into B1 (cols 0-3) and B2 (cols 4-7).
    # C into C1 (cols 0-3) and C2 (cols 4-7).
    # C1 = A B1. C2 = A B2.
    
    # We use the same U4 for both.
    # For B1, we use V4. For B2, we use V4.
    # For C1, we use W4. For C2, we use W4.
    
    # Total rank = 49 + 49 = 98.
    
    U_total = np.zeros((16, 98))
    V_total = np.zeros((32, 98))
    W_total = np.zeros((32, 98))
    
    # First 49 columns: for B1
    U_total[:, 0:49] = U4
    V_total[0:16, 0:49] = V4  # B1 is rows 0-15 of B (j*4+k, k=0..3)
    W_total[0:16, 0:49] = W4   # C1 is rows 0-15 of C (k*4+i, k=0..3)
    
    # Next 49 columns: for B2
    U_total[:, 49:98] = U4
    V_total[16:32, 49:98] = V4 # B2 is rows 16-31 of B (j*4+k, k=4..7)
    W_total[16:32, 49:98] = W4 # C2 is rows 16-31 of C (k*4+i, k=4..7)
    
    # Convert to lists
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
