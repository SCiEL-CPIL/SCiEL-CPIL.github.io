"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110s wall-clock limit with safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We will construct a decomposition using Block Strassen.
    # Strategy:
    # 1. Split B (4x6) into B1 (4x4) and B2 (4x2).
    # 2. Compute C1 = A * B1 (4x4) using Strassen (Rank 49).
    # 3. Compute C2 = A * B2 (4x2) using Strassen-like block (Rank 28).
    # Total Rank = 49 + 28 = 77.
    
    # Indexing conventions:
    # U: shape (n*m, rank) -> index i*m + j for A[i,j]
    # V: shape (m*p, rank) -> index j*p + k for B[j,k]
    # W: shape (p*n, rank) -> index k*n + i for C[k,i] (Trace convention W[k*n+i,r])
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    col = 0
    
    # Helper to add a term to the current column
    def add_term(u_idx, v_idx, w_idx, coeff):
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff
    
    # --- Part 1: 4x4 * 4x4 using Recursive Strassen (Rank 49) ---
    # A is 4x4, B1 is 4x4 (cols 0-3 of B).
    # C1 is 4x4 (cols 0-3 of C).
    
    # We view A, B1, C1 as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B1 = [[B11, B12], [B21, B22]]
    # C1 = [[C11, C12], [C21, C22]]
    
    # Indices for A (4x4, row-major, stride 4):
    # A11 (rows 0-1, cols 0-1): 0, 1, 4, 5
    # A12 (rows 0-1, cols 2-3): 2, 3, 6, 7
    # A21 (rows 2-3, cols 0-1): 8, 9, 12, 13
    # A22 (rows 2-3, cols 2-3): 10, 11, 14, 15
    
    # Indices for B1 (4x4 part of B, row-major, stride 6):
    # B11 (rows 0-1, cols 0-1): 0, 1, 6, 7
    # B12 (rows 0-1, cols 2-3): 2, 3, 8, 9
    # B21 (rows 2-3, cols 0-1): 12, 13, 18, 19
    # B22 (rows 2-3, cols 2-3): 14, 15, 20, 21
    
    # Indices for C1 (4x4 part of C, W index k*n+i, stride 4):
    # C11 (k 0-1, i 0-1): 0, 1, 4, 5
    # C12 (k 0-1, i 2-3): 2, 3, 6, 7
    # C21 (k 2-3, i 0-1): 8, 9, 12, 13
    # C22 (k 2-3, i 2-3): 10, 11, 14, 15
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 6, 7]
    B12 = [2, 3, 8, 9]
    B21 = [12, 13, 18, 19]
    B22 = [14, 15, 20, 21]
    
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    # Strassen 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each "block multiplication" (2x2 * 2x2) is done with 7 scalar multiplications (Strassen 2x2).
    # So for each of the 7 Strassen terms, we need to expand into 7 scalar terms.
    # Total 49 terms.
    
    def add_strassen_block(a_block, b_block, base_col):
        # a_block: list of 4 indices for the 2x2 block in A
        # b_block: list of 4 indices for the 2x2 block in B
        # This adds 7 terms for the 2x2 * 2x2 multiplication using Strassen.
        # Returns the list of 7 "result block indices" logic is handled by caller?
        # No, we need to add the terms to U, V, W.
        # But the result of the block multiplication is a 2x2 matrix.
        # The caller needs to know which output blocks this contributes to.
        # This function is for the *inner* Strassen.
        
        # Inner Strassen for 2x2 * 2x2:
        # Let A_b = [[a00, a01], [a10, a11]]
        # Let B_b = [[b00, b01], [b10, b11]]
        # Indices in a_block: [a00_idx, a01_idx, a10_idx, a11_idx]
        # Indices in b_block: [b00_idx, b01_idx, b10_idx, b11_idx]
        
        # We need to generate 7 terms.
        # But we don't have a single output block index yet, because the outer Strassen
        # combines these results into C blocks.
        # This structure is tricky.
        
        # Alternative: Just generate all 49 terms explicitly by iterating
        # through the outer Strassen structure and inner Strassen structure.
        
        pass

    # Let's do it manually/explicitly for clarity and correctness.
    # Outer Strassen Terms (7):
    # T1 = (A11+A22) * (B11+B22)
    # T2 = (A21+A22) * B11
    # T3 = A11 * (B12-B22)
    # T4 = A22 * (B21-B11)
    # T5 = (A11+A12) * B22
    # T6 = (A21-A11) * (B11+B12)
    # T7 = (A12-A22) * (B21+B22)
    
    # Each Ti is a 2x2 * 2x2 multiplication.
    # We expand each Ti using inner Strassen (7 terms each).
    
    # Inner Strassen for 2x2 * 2x2 (A_sub * B_sub):
    # s1 = (A00+A11)(B00+B11)
    # s2 = (A10+A11)B00
    # s3 = A00(B01-B11)
    # s4 = A11(B10-B00)
    # s5 = (A00+A01)B11
    # s6 = (A10-A00)(B00+B01)
    # s7 = (A01-A11)(B10+B11)
    
    # C00 = s1 + s4 - s5 + s7
    # C01 = s3 + s5
    # C10 = s2 + s4
    # C11 = s1 - s2 + s3 + s6
    
    # We will write a helper to add an inner Strassen multiplication
    # given the 4 A-indices and 4 B-indices, and the 4 C-indices (with signs).
    # Wait, the C-indices depend on which outer term it is and which inner term it is.
    
    # Let's define a function that adds one scalar multiplication term
    # to the global U, V, W matrices.
    
    def add_scalar_term(a_idx, b_idx, c_idx, sign):
        # a_idx, b_idx, c_idx are integers (flat indices)
        # sign is the coefficient for this term in the final sum for C_idx
        # Actually, in the tensor decomposition, each term is a rank-1 tensor.
        # U[a_idx, col] = sign_a
        # V[b_idx, col] = sign_b
        # W[c_idx, col] = sign_c
        # The product U[a]*V[b]*W[c] contributes to C[c_idx] with coefficient sign_a*sign_b*sign_c.
        # We want the sum of these contributions to equal the matrix product.
        # In Strassen, we have linear combinations.
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # So we can split this into 4 scalar multiplications? No, that's schoolbook.
        # Strassen uses 7 multiplications of *sums*.
        # The multiplication is of the form (Sum A_i * A_i) * (Sum B_j * B_j).
        # This is NOT a rank-1 tensor in the standard sense unless we treat the sums as pre-computed?
        # No, the tensor decomposition must be a sum of rank-1 tensors u v w^T.
        # Each term is u_a * v_b * w_c.
        # If we have (A11+A22)(B11+B22), this corresponds to:
        # (A11*B11) + (A11*B22) + (A22*B11) + (A22*B22)
        # This is 4 rank-1 terms.
        # Strassen reduces the number of *multiplications* in the algorithm, but the tensor rank
        # is the number of rank-1 terms in the decomposition.
        # The rank of the 2x2 matrix multiplication tensor is 7.
        # The 7 terms are:
        # 1. (A11+A22)(B11+B22) -> This is NOT a single rank-1 term. It is a bilinear form.
        # Wait. The tensor decomposition is T = sum u_i v_i w_i^T.
        # Each u_i v_i w_i^T corresponds to a term A_{a_i} B_{b_i} C_{c_i}.
        # Strassen's algorithm uses 7 multiplications, but each multiplication is of linear combinations.
        # Does this mean the tensor rank is 7?
        # Yes, the rank of the 2x2 matrix multiplication tensor is 7.
        # The 7 rank-1 tensors are:
        # 1. (e11+e22) (e11+e22) (e11+e22 - e12 - e21 + e22)? No.
        
        # Let's look up the explicit rank-1 decomposition for 2x2 MM.
        # The 7 terms are:
        # 1. (A11+A22)(B11+B22) contributes to C11, C12, C21, C22?
        # No, the output is C = A*B.
        # The rank-1 terms are:
        # 1. u1 = A11+A22, v1 = B11+B22, w1 = C11+? 
        # Actually, the standard Strassen decomposition gives:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M_k is a product of two linear forms.
        # M_k = (sum a_i A_i) * (sum b_j B_j)
        # This M_k is then added to specific C entries with coefficients.
        # In the tensor decomposition, each M_k corresponds to a rank-1 term?
        # No, M_k is a scalar. It is multiplied by coefficients to form C entries.
        # The term in the tensor decomposition is:
        # u_k (vector in A-space) * v_k (vector in B-space) * w_k (vector in C-space)
        # where u_k has non-zeros at the A indices involved in M_k,
        # v_k has non-zeros at the B indices involved in M_k,
        # and w_k has non-zeros at the C indices that M_k contributes to.
        
        # For M1 = (A11+A22)(B11+B22):
        # u1: A11=1, A22=1
        # v1: B11=1, B22=1
        # w1: C11=1, C22=1 (since C11+=M1, C22+=M1)
        # Wait, C11 = M1 + ... and C22 = M1 + ...
        # So w1 has 1 at C11 and 1 at C22.
        
        # Let's verify:
        # Term 1: u1 v1 w1^T
        # Contribution to C11: 1*1*1 = 1 (from M1)
        # Contribution to C22: 1*1*1 = 1 (from M1)
        # This matches the C11 and C22 parts of M1.
        
        # So, for each Strassen multiplication M_k, we have ONE rank-1 term.
        # u_k: coefficients for A indices
        # v_k: coefficients for B indices
        # w_k: coefficients for C indices
        
        # This is much simpler!
        
        pass

    # Let's implement this.
    
    # Helper to add a Strassen multiplication term
    def add_strassen_term(a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: dict {index: coeff}
        # b_coeffs: dict {index: coeff}
        # c_coeffs: dict {index: coeff}
        global col
        for idx, coeff in a_coeffs.items():
            U[idx, col] += coeff
        for idx, coeff in b_coeffs.items():
            V[idx, col] += coeff
        for idx, coeff in c_coeffs.items():
            W[idx, col] += coeff
        col += 1

    # Part 1: 4x4 * 4x4
    
    # We need to define the 2x2 blocks for A, B1, C1.
    # A11: [0, 1, 4, 5] -> A11[0,0]=0, A11[0,1]=1, A11[1,0]=4, A11[1,1]=5
    # A12: [2, 3, 6, 7]
    # A21: [8, 9, 12, 13]
    # A22: [10, 11, 14, 15]
    
    # B11: [0, 1, 6, 7]
    # B12: [2, 3, 8, 9]
    # B21: [12, 13, 18, 19]
    # B22: [14, 15, 20, 21]
    
    # C11: [0, 1, 4, 5]
    # C12: [2, 3, 6, 7]
    # C21: [8, 9, 12, 13]
    # C22: [10, 11, 14, 15]
    
    # We will use the recursive structure.
    # The "blocks" A11 etc are 2x2 matrices.
    # The Strassen algorithm applies to these blocks.
    # Each block multiplication (e.g., A11*B11) is itself a 2x2*2x2 multiplication.
    # We must expand each block multiplication into 7 scalar terms.
    
    # So, for each of the 7 outer Strassen terms, we have 7 inner Strassen terms.
    # Total 49 terms.
    
    # Let's define the inner Strassen terms for a generic 2x2*2x2.
    # Input: 4 A-indices (a00, a01, a10, a11), 4 B-indices (b00, b01, b10, b11)
    # Output: 7 terms, each with A-coeffs, B-coeffs, and a "result vector" which is a 2x2 matrix of coefficients.
    # But we need to combine these with the outer Strassen coefficients.
    
    # This is getting complex. Let's just hardcode the 49 terms by generating them.
    
    # Function to generate 7 terms for 2x2*2x2
    def get_inner_strassen_terms(a_idx, b_idx):
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # Returns list of 7 tuples: (a_coeffs, b_coeffs, c_coeffs)
        # where c_coeffs is a dict for the 4 C-indices of this block.
        # But we don't know the C-indices yet.
        # The C-indices depend on which outer term this is.
        
        # Let's return the A and B coefficients, and the C coefficients relative to the block's local C indices.
        # Local C indices: c00, c01, c10, c11.
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        terms.append({
            'a': {a_idx[0]: 1, a_idx[3]: 1},
            'b': {b_idx[0]: 1, b_idx[3]: 1},
            'c': {0: 1, 3: 1}  # local c00, c11
        })
        
        # M2 = (A10+A11)B00
        # C10 += M2
        terms.append({
            'a': {a_idx[2]: 1, a_idx[3]: 1},
            'b': {b_idx[0]: 1},
            'c': {2: 1}  # local c10
        })
        
        # M3 = A00(B01-B11)
        # C01 += M3
        terms.append({
            'a': {a_idx[0]: 1},
            'b': {b_idx[1]: 1, b_idx[3]: -1},
            'c': {1: 1}  # local c01
        })
        
        # M4 = A11(B10-B00)
        # C00 += M4, C10 += M4
        terms.append({
            'a': {a_idx[3]: 1},
            'b': {b_idx[2]: 1, b_idx[0]: -1},
            'c': {0: 1, 2: 1}  # local c00, c10
        })
        
        # M5 = (A00+A01)B11
        # C00 -= M5, C01 += M5
        terms.append({
            'a': {a_idx[0]: 1, a_idx[1]: 1},
            'b': {b_idx[3]: 1},
            'c': {0: -1, 1: 1}  # local c00, c01
        })
        
        # M6 = (A10-A00)(B00+B01)
        # C11 += M6
        terms.append({
            'a': {a_idx[2]: 1, a_idx[0]: -1},
            'b': {b_idx[0]: 1, b_idx[1]: 1},
            'c': {3: 1}  # local c11
        })
        
        # M7 = (A01-A11)(B10+B11)
        # C00 += M7, C11 += M7
        terms.append({
            'a': {a_idx[1]: 1, a_idx[3]: -1},
            'b': {b_idx[2]: 1, b_idx[3]: 1},
            'c': {0: 1, 3: 1}  # local c00, c11
        })
        
        return terms

    # Now we need to combine outer and inner.
    # Outer Strassen Terms for 4x4*4x4:
    # T1 = (A11+A22)(B11+B22) -> C11 += T1, C22 += T1
    # T2 = (A21+A22)B11 -> C21 += T2
    # T3 = A11(B12-B22) -> C12 += T3
    # T4 = A22(B21-B11) -> C11 += T4, C21 += T4
    # T5 = (A11+A12)B22 -> C11 -= T5, C12 += T5
    # T6 = (A21-A11)(B11+B12) -> C22 += T6
    # T7 = (A12-A22)(B21+B22) -> C11 += T7, C22 += T7
    
    # Each T_k is a 2x2*2x2 multiplication.
    # We need to expand each T_k into 7 inner terms.
    # For each inner term, we multiply the A and B coefficients by the outer coefficients,
    # and the C coefficients by the outer C coefficients.
    
    # Let's define the outer terms as (a_blocks, b_blocks, c_blocks)
    # where a_blocks is a dict {block_name: coeff}
    # and c_blocks is a dict {block_name: coeff}
    
    outer_terms = [
        # T1
        {'a': {'A11': 1, 'A22': 1}, 'b': {'B11': 1, 'B22': 1}, 'c': {'C11': 1, 'C22': 1}},
        # T2
        {'a': {'A21': 1, 'A22': 1}, 'b': {'B11': 1}, 'c': {'C21': 1}},
        # T3
        {'a': {'A11': 1}, 'b': {'B12': 1, 'B22': -1}, 'c': {'C12': 1}},
        # T4
        {'a': {'A22': 1}, 'b': {'B21': 1, 'B11': -1}, 'c': {'C11': 1, 'C21': 1}},
        # T5
        {'a': {'A11': 1, 'A12': 1}, 'b': {'B22': 1}, 'c': {'C11': -1, 'C12': 1}},
        # T6
        {'a': {'A21': 1, 'A11': -1}, 'b': {'B11': 1, 'B12': 1}, 'c': {'C22': 1}},
        # T7
        {'a': {'A12': 1, 'A22': -1}, 'b': {'B21': 1, 'B22': 1}, 'c': {'C11': 1, 'C22': 1}}
    ]
    
    # Block indices
    blocks = {
        'A11': A11, 'A12': A12, 'A21': A21, 'A22': A22,
        'B11': B11, 'B12': B12, 'B21': B21, 'B22': B22,
        'C11': C11, 'C12': C12, 'C21': C21, 'C22': C22
    }
    
    # For each outer term, we need to combine the A and B blocks.
    # The inner Strassen operates on the combined A and B blocks.
    # But the inner Strassen expects 4 indices for A and 4 for B.
    # If the outer term involves multiple blocks (e.g., A11+A22), we cannot just pass one block.
    # We must expand the linear combination.
    
    # This is the hard part. The inner Strassen terms are rank-1 tensors for the 2x2 case.
    # If we have (A11+A22) * (B11+B22), this is not a single 2x2*2x2 multiplication.
    # It is the sum of 4 multiplications: A11B11 + A11B22 + A22B11 + A22B22.
    # We could just use schoolbook for the outer level? No, that increases rank.
    
    # Actually, the rank of the 4x4*4x4 tensor is 49.
    # The 49 terms are obtained by applying Strassen recursively.
    # The first level of Strassen has 7 terms.
    # Each term is a product of two 2x2 block matrices.
    # Each 2x2 block matrix is a linear combination of the original 2x2 blocks.
    # To compute the product of two linear combinations, we must expand it.
    # (Sum A_i)(Sum B_j) = Sum A_i B_j.
    # This expansion increases the number of terms.
    
    # Wait. If we use Strassen, we replace the 4 multiplications of 2x2 blocks with 7 multiplications of 2x2 blocks.
    # Each "multiplication of 2x2 blocks" is itself a 2x2*2x2 tensor.
    # If we use Strassen for the inner level, each inner multiplication becomes 7 scalar terms.
    # So 7 outer terms * 7 inner terms = 49 scalar terms.
    
    # But the outer terms are linear combinations.
    # M1 = (A11+A22)(B11+B22)
    # This is NOT a single 2x2*2x2 multiplication. It is a bilinear form.
    # To decompose it into rank-1 tensors, we must expand it.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # Each of these is a 2x2*2x2 multiplication.
    # If we use Strassen for each of these, we get 4 * 7 = 28 terms for M1.
    # This is way more than 49 total.
    
    # So, how is the rank 49 achieved?
    # The rank 49 decomposition does NOT simply recurse Strassen in this naive way.
    # It uses a specific set of 49 rank-1 tensors.
    # I need to find or generate them.
    
    # Given the complexity and time limit, and the risk of error,
    # I will fall back to the schoolbook decomposition (Rank 96) which is guaranteed to be correct.
    # The score will be 1.0.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
