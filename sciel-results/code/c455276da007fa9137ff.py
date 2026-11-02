"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x7 matrix.
    
    We split the problem into two parts:
    1. Multiplying A (4x4) by B1 (4x4): Use Strassen's algorithm recursively
       to get rank 49 (which is 7*7, as 4x4x4 can be computed via 7 multiplications
       of 2x2x2 blocks, each of rank 7).
    2. Multiplying A (4x4) by B2 (4x3): Use schoolbook (rank 48) or potentially
       a more efficient decomposition if available.
    
    Total rank = 49 + 48 = 97.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes
    (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # We'll construct the decomposition in two parts:
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4)
    # Part 2: A (4x4) * B2 (4x3) -> C2 (4x3)
    # C = C1 + C2
    
    # For Part 1, we use the recursive Strassen decomposition for 4x4x4.
    # This gives rank 49.
    # For Part 2, we use schoolbook for 4x4x3, giving rank 48.
    # Total rank = 49 + 48 = 97.
    
    # Helper to get Strassen decomposition for 2x2x2 (rank 7)
    def strassen_2x2x2():
        # Returns U (4x7), V (4x7), W (4x7) for M_{2,2,2}
        # The 7 products in Strassen's algorithm:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # Row indices for U (A matrix): 0=A11, 1=A12, 2=A21, 3=A22
        # Row indices for V (B matrix): 0=B11, 1=B12, 2=B21, 3=B22
        # Row indices for W (C matrix): 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 -= M1, C22 += M1
        U[0, 0] += 1  # A11
        U[3, 0] += 1  # A22
        V[0, 0] += 1  # B11
        V[3, 0] += 1  # B22
        W[0, 0] += 1  # C11
        W[1, 0] += 1  # C12
        W[2, 0] -= 1  # C21
        W[3, 0] += 1  # C22
        
        # M2: (A21+A22)B11 -> C21 += M2, C22 += M2
        U[2, 1] += 1  # A21
        U[3, 1] += 1  # A22
        V[0, 1] += 1  # B11
        W[2, 1] += 1  # C21
        W[3, 1] += 1  # C22
        
        # M3: A11(B12-B22) -> C12 += M3, C11 -= M3
        U[0, 2] += 1  # A11
        V[1, 2] += 1  # B12
        V[3, 2] -= 1  # -B22
        W[1, 2] += 1  # C12
        W[0, 2] -= 1  # C11
        
        # M4: A22(B21-B11) -> C21 += M4, C22 -= M4
        U[3, 3] += 1  # A22
        V[2, 3] += 1  # B21
        V[0, 3] -= 1  # -B11
        W[2, 3] += 1  # C21
        W[3, 3] -= 1  # C22
        
        # M5: (A11+A12)B22 -> C12 += M5, C22 += M5
        U[0, 4] += 1  # A11
        U[1, 4] += 1  # A12
        V[3, 4] += 1  # B22
        W[1, 4] += 1  # C12
        W[3, 4] += 1  # C22
        
        # M6: (A21-A11)(B11+B12) -> C21 += M6, C11 += M6
        U[2, 5] += 1  # A21
        U[0, 5] -= 1  # -A11
        V[0, 5] += 1  # B11
        V[1, 5] += 1  # B12
        W[2, 5] += 1  # C21
        W[0, 5] += 1  # C11
        
        # M7: (A12-A22)(B21+B22) -> C12 += M7, C22 += M7
        U[1, 6] += 1  # A12
        U[3, 6] -= 1  # -A22
        V[2, 6] += 1  # B21
        V[3, 6] += 1  # B22
        W[1, 6] += 1  # C12
        W[3, 6] += 1  # C22
        
        return U, V, W
    
    # Helper to get schoolbook decomposition for m x p with fixed n
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Build Part 1: 4x4x4 using recursive Strassen
    # We need to combine 4 blocks of 2x2x2.
    # A is 4x4, split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B is 4x4, split into 2x2 blocks:
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    
    # The U matrix for 4x4x4 has rows indexed by (i, j) where i in {0,1}, j in {0,1,2,3}
    # Actually, let's think of it as:
    # The 4x4 matrix A has 16 elements. In the 2x2 block decomposition:
    # A11 corresponds to A[0:2, 0:2]
    # A12 corresponds to A[0:2, 2:4]
    # A21 corresponds to A[2:4, 0:2]
    # A22 corresponds to A[2:4, 2:4]
    
    # So the rows of U for the 4x4x4 problem correspond to the 16 elements of A.
    # Row index in U: i*m + j where i in 0..3, j in 0..3
    # We need to map these to the 4 blocks.
    
    # Let's define a function to build the 4x4x4 decomposition from 2x2x2.
    def build_4x4x4_strassen():
        U_small, V_small, W_small = strassen_2x2x2()
        rank_small = 7
        rank_4 = rank_small * rank_small  # 49
        
        U = np.zeros((16, rank_4), dtype=int)
        V = np.zeros((16, rank_4), dtype=int)
        W = np.zeros((16, rank_4), dtype=int)
        
        # The 4x4 matrix is composed of 4 2x2 blocks.
        # For U: each element of the 4x4 matrix appears in one of the 7 Strassen products
        # for its corresponding block.
        
        # Let's define the mapping from 4x4 indices to 2x2 block indices.
        # Block (bi, bj) where bi, bj in {0, 1}
        # A[2*bi + di, 2*bj + dj] where di, dj in {0, 1}
        
        # For the 4x4x4 problem, we have a tensor product structure.
        # The 7 products for the outer 2x2 blocks are combined with the 7 products
        # for the inner 2x2 blocks.
        
        # Actually, the standard way to do this is:
        # For each of the 7 outer Strassen products, we compute a 2x2x2 multiplication
        # for the inner blocks.
        
        # Let's label the outer Strassen products 0..6 and inner 0..6.
        # Total rank = 7 * 7 = 49.
        
        # For each (outer_prod, inner_prod) pair:
        # The U coefficient is U_outer[outer_elem, outer_prod] * U_inner[inner_elem, inner_prod]
        # where outer_elem is the block index (0..3) and inner_elem is the position within
        # the block (0..3).
        
        # This is getting complex. Let's use a different approach:
        # Directly construct the 4x4x4 decomposition using the known Strassen formulas.
        
        # Actually, the simplest correct approach is to use the Kronecker product
        # structure. The 4x4x4 tensor is the tensor product of two 2x2x2 tensors
        # in a specific way, but Strassen's algorithm for 4x4 is not simply the
        # tensor product of two 2x2x2 Strassen decompositions.
        
        # Let me use a verified rank-49 decomposition for 4x4x4.
        # I'll construct it by combining the 2x2x2 Strassen in a hierarchical way.
        
        # For now, let's use schoolbook for 4x4x4 as well (rank 64) to ensure correctness,
        # then optimize if time permits.
        # But 64 + 48 = 112, which is the same as the original.
        
        # Let me try to implement the hierarchical Strassen properly.
        
        # The idea:
        # A (4x4) = [[A11, A12], [A21, A22]] where each Aij is 2x2
        # B (4x4) = [[B11, B12], [B21, B22]] where each Bij is 2x2
        # C (4x4) = [[C11, C12], [C21, C22]] where each Cij is 2x2
        
        # Strassen for the 2x2 "super" blocks:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mj is a 2x2 matrix, computed as a 2x2x2 multiplication.
        # So each Mj requires 7 scalar multiplications.
        # Total: 7 * 7 = 49 scalar multiplications.
        
        # Now, let's build the U, V, W matrices.
        # U has 16 rows (for A's 16 elements) and 49 columns.
        # V has 16 rows (for B's 16 elements) and 49 columns.
        # W has 16 rows (for C's 16 elements) and 49 columns.
        
        # Let's define the block structure:
        # A element at (r, c) where r, c in 0..3
        # Block index: (r // 2, c // 2)
        # Position within block: (r % 2, c % 2)
        
        # For each of the 7 outer products, we have a 2x2x2 sub-problem.
        # The sub-problem involves specific blocks of A and B, and produces
        # contributions to specific blocks of C.
        
        # Let's define the 7 outer products and their contributions.
        # For each outer product j (0..6):
        #   It involves certain A-blocks and B-blocks with certain signs.
        #   The result contributes to certain C-blocks with certain signs.
        
        # We'll iterate over the 7 outer products and 7 inner products.
        
        U_4 = np.zeros((16, 49), dtype=int)
        V_4 = np.zeros((16, 49), dtype=int)
        W_4 = np.zeros((16, 49), dtype=int)
        
        col = 0
        for j_outer in range(7):
            # For each outer product, determine which A-blocks and B-blocks are involved
            # and their signs.
            
            # Let's define the 7 outer Strassen products explicitly.
            # j_outer=0: M1 = (A11+A22)(B11+B22)
            # j_outer=1: M2 = (A21+A22)B11
            # j_outer=2: M3 = A11(B12-B22)
            # j_outer=3: M4 = A22(B21-B11)
            # j_outer=4: M5 = (A11+A12)B22
            # j_outer=5: M6 = (A21-A11)(B11+B12)
            # j_outer=6: M7 = (A12-A22)(B21+B22)
            
            # For each outer product, we have a 2x2x2 sub-problem.
            # The A-inputs for the sub-problem are linear combinations of A-blocks.
            # The B-inputs for the sub-problem are linear combinations of B-blocks.
            # The C-output for the sub-problem is a linear combination of C-blocks.
            
            # Let's define the A-coefficients, B-coefficients, and C-coefficients
            # for each of the 4 positions in the 2x2 sub-problem.
            
            # This is getting very detailed. Let me use a known correct implementation.
            
            # Alternative: use the fact that the 4x4x4 Strassen decomposition
            # can be built from the 2x2x2 one by "expanding" each element.
            
            # For now, let's fall back to schoolbook for 4x4x4 to ensure correctness.
            # We can optimize later.
            pass
        
        # Let's just use schoolbook for 4x4x4 (rank 64) and 4x4x3 (rank 48).
        # Total rank = 64 + 48 = 112. This is the same as the original.
        
        # To get an improvement, let me try a different approach:
        # Use the known rank 49 for 4x4x4. I'll hardcode a verified decomposition.
        
        # Actually, let me implement the hierarchical Strassen correctly.
        # I'll define the 7 outer products and for each, the 7 inner products.
        
        # For the outer product j, the A-inputs are:
        # a0, a1, a2, a3 (coefficients for A11, A12, A21, A22 blocks)
        # For the B-inputs:
        # b0, b1, b2, b3 (coefficients for B11, B12, B21, B22 blocks)
        # For the C-output:
        # c0, c1, c2, c3 (coefficients for C11, C12, C21, C22 blocks)
        
        # The 7 outer Strassen products:
        outer_products = [
            # (a_coeffs, b_coeffs, c_coeffs)
            # Each is a tuple of 4 integers for the 4 blocks.
            ((1, 0, 0, 1), (1, 0, 0, 1), (1, 1, -1, 1)),  # M1
            ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 1)),    # M2
            ((1, 0, 0, 0), (0, 1, 0, -1), (-1, 1, 0, 0)),  # M3
            ((0, 0, 0, 1), (0, 0, 1, -1), (0, 0, 1, -1)),  # M4
            ((1, 1, 0, 0), (0, 0, 0, 1), (0, 1, 0, 1)),    # M5
            ((-1, 0, 1, 0), (1, 1, 0, 0), (1, 0, 1, 0)),   # M6
            ((0, 1, 0, -1), (0, 0, 1, 1), (0, 1, 0, 1)),   # M7
        ]
        
        for j_outer in range(7):
            a_coeffs, b_coeffs, c_coeffs = outer_products[j_outer]
            
            # For each of the 7 inner Strassen products:
            U_inner, V_inner, W_inner = strassen_2x2x2()
            
            for j_inner in range(7):
                # The column index is j_outer * 7 + j_inner
                col = j_outer * 7 + j_inner
                
                # For the A-input:
                # The j_inner-th Strassen product uses certain A elements
                # from the 2x2 sub-problem.
                # The sub-problem's A elements are:
                # A_sub[0] = a_coeffs[0] * A11 (block 0)
                # A_sub[1] = a_coeffs[1] * A12 (block 1)
                # A_sub[2] = a_coeffs[2] * A21 (block 2)
                # A_sub[3] = a_coeffs[3] * A22 (block 3)
                
                # For the inner Strassen product j_inner, the A-coefficient
                # for A_sub element k is U_inner[k, j_inner].
                
                # So for block b (0..3), the coefficient is:
                # a_coeffs[b] * U_inner[2*(b//2) + (b%2), j_inner]
                # Wait, the mapping from block index to A_sub index is:
                # Block 0 (A11) -> A_sub[0]
                # Block 1 (A12) -> A_sub[1]
                # Block 2 (A21) -> A_sub[2]
                # Block 3 (A22) -> A_sub[3]
                
                for b in range(4):
                    # Block b corresponds to A_sub element b
                    coeff_a = a_coeffs[b] * U_inner[b, j_inner]
                    if coeff_a != 0:
                        # This block has 4 elements (2x2)
                        # Block b is at position:
                        # b=0: rows 0-1, cols 0-1
                        # b=1: rows 0-1, cols 2-3
                        # b=2: rows 2-3, cols 0-1
                        # b=3: rows 2-3, cols 2-3
                        row_start = (b // 2) * 2
                        col_start = (b % 2) * 2
                        for dr in range(2):
                            for dc in range(2):
                                r = row_start + dr
                                c = col_start + dc
                                U_4[r * 4 + c, col] += coeff_a
                
                # Similarly for B:
                for b in range(4):
                    coeff_b = b_coeffs[b] * V_inner[b, j_inner]
                    if coeff_b != 0:
                        row_start = (b // 2) * 2
                        col_start = (b % 2) * 2
                        for dr in range(2):
                            for dc in range(2):
                                r = row_start + dr
                                c = col_start + dc
                                V_4[r * 4 + c, col] += coeff_b
                
                # For C:
                for b in range(4):
                    coeff_c = c_coeffs[b] * W_inner[b, j_inner]
                    if coeff_c != 0:
                        row_start = (b // 2) * 2
                        col_start = (b % 2) * 2
                        for dr in range(2):
                            for dc in range(2):
                                r = row_start + dr
                                c = col_start + dc
                                W_4[r * 4 + c, col] += coeff_c
        
        return U_4, V_4, W_4
    
    # Build Part 2: 4x4x3 using schoolbook
    U_2, V_2, W_2 = schoolbook(4, 4, 3)
    
    # Combine Part 1 and Part 2
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4)
    # Part 2: A (4x4) * B2 (4x3) -> C2 (4x3)
    
    # The full B is 4x7, so B = [B1, B2] where B1 is 4x4 and B2 is 4x3.
    # The full C is 4x7, so C = [C1, C2] where C1 is 4x4 and C2 is 4x3.
    
    # U for Part 1: shape (16, 49)
    # V for Part 1: shape (16, 49) -> but B1 is 4x4, so V should be (4*4, 49) = (16, 49)
    # W for Part 1: shape (16, 49) -> but C1 is 4x4, so W should be (4*4, 49) = (16, 49)
    # Wait, the W convention is W[k*n+i, r]. For C1 (4x4), n=4, p=4.
    # So W has p*n = 4*4 = 16 rows.
    
    # U for Part 2: shape (16, 48)
    # V for Part 2: shape (4*3, 48) = (12, 48)
    # W for Part 2: shape (3*4, 48) = (12, 48)
    
    # The full U has 16 rows (for A, which is 4x4) and 49+48=97 columns.
    # The full V has 4*7=28 rows (for B, which is 4x7) and 97 columns.
    #   - First 16 rows for B1 (4x4)
    #   - Next 12 rows for B2 (4x3)
    # The full W has 7*4=28 rows (for C, which is 4x7) and 97 columns.
    #   - First 16 rows for C1 (4x4)
    #   - Next 12 rows for C2 (4x3)
    
    U_full = np.zeros((16, 97), dtype=int)
    V_full = np.zeros((28, 97), dtype=int)
    W_full = np.zeros((28, 97), dtype=int)
    
    # Copy Part 1
    U_full[:, 0:49] = U_4
    V_full[0:16, 0:49] = V_4
    W_full[0:16, 0:49] = W_4
    
    # Copy Part 2
    U_full[:, 49:97] = U_2
    V_full[16:28, 49:97] = V_2
    W_full[16:28, 49:97] = W_2
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
