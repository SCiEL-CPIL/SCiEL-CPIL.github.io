"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Part 1: 4x4 * 4x4 using block Strassen (rank 49)
    # Part 2: 4x4 * 4x2 using block Strassen (rank 28)
    # Total rank: 77
    
    total_rank = 77
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # A is 4x4: A[i,j] at index i*4+j in U
    # B is 4x6: B[j,k] at index j*6+k in V
    # C is 6x4: C[k,i] at index k*4+i in W
    
    def a_idx(i, j):
        return i * 4 + j
    
    def b_idx(j, k):
        return j * 6 + k
    
    def w_idx(k, i):
        return k * 4 + i
    
    col = 0
    
    # --- Part 1: C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4] ---
    # Block Strassen: 2x2 blocks of 2x2 matrices
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    
    # Each 2x2*2x2 multiplication uses 7 scalar multiplications
    # We'll add 7 terms for each Strassen multiplication
    
    def add_scalar_term(a_i, a_j, b_j, b_k, w_k, w_i, coefficient):
        """Add a single scalar multiplication term."""
        nonlocal col
        U[a_idx(a_i, a_j)][col] += coefficient
        V[b_idx(b_j, b_k)][col] += coefficient
        W[w_idx(w_k, w_i)][col] += coefficient
        col += 1
    
    # Strassen 2x2*2x2 -> 2x2*2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen multiplication, we need to expand into scalar terms
    # A11 block: i in {0,1}, j in {0,1}
    # A12 block: i in {0,1}, j in {2,3}
    # A21 block: i in {2,3}, j in {0,1}
    # A22 block: i in {2,3}, j in {2,3}
    
    # B11 block: j in {0,1}, k in {0,1}
    # B12 block: j in {0,1}, k in {2,3}
    # B21 block: j in {2,3}, k in {0,1}
    # B22 block: j in {2,3}, k in {2,3}
    
    # C11 block: k in {0,1}, i in {0,1}
    # C12 block: k in {2,3}, i in {0,1}
    # C21 block: k in {0,1}, i in {2,3}
    # C22 block: k in {2,3}, i in {2,3}
    
    # M1 = (A11+A22)(B11+B22)
    # A11+A22: elements at (i,j) where i,j in {0,1} get +1, i,j in {2,3} get +1
    # B11+B22: elements at (j,k) where j,k in {0,1} get +1, j,k in {2,3} get +1
    # This is a 4x4 * 4x4 multiplication but only certain elements are nonzero
    # Actually, (A11+A22) is a 4x4 matrix where top-left 2x2 = A11, bottom-right 2x2 = A22, others 0
    # Similarly for B11+B22
    
    # The product (A11+A22)(B11+B22) gives a 4x4 result
    # But we only care about how it contributes to C11 and C22
    
    # Let me think differently. Each Strassen multiplication M is a 2x2 matrix product.
    # M = X * Y where X and Y are 2x2 matrices (blocks).
    # To compute this 2x2*2x2 = 2x2, we use schoolbook (4 scalar mults) or Strassen (7 scalar mults).
    # Since we want rank 49 for 4x4*4x4, we use Strassen at the block level,
    # and schoolbook at the scalar level within each block multiplication.
    
    # So: 7 block multiplications * 4 scalar multiplications per block mult = 28? No.
    # Wait, 2x2 * 2x2 with Strassen = 7 scalar multiplications.
    # So 7 block-level Strassen multiplications, each being a 2x2*2x2 = 7 scalar mults.
    # Total: 7 * 7 = 49. Yes!
    
    # Let me define a helper to add a 2x2*2x2 Strassen decomposition
    
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, c_stride_r, c_stride_c):
        """
        Add Strassen decomposition for a 2x2*2x2 multiplication.
        a_r0, a_c0: top-left corner of A block in A (4x4)
        b_r0, b_c0: top-left corner of B block in B (4x6)
        c_r0, c_c0: top-left corner of C block in C (6x4)
        c_stride_r, c_stride_c: strides for C
        Returns number of columns used.
        """
        nonlocal col
        num_cols = 0
        
        # A block indices: (a_r0, a_c0), (a_r0, a_c0+1), (a_r0+1, a_c0), (a_r0+1, a_c0+1)
        # B block indices: (b_r0, b_c0), (b_r0, b_c0+1), (b_r0+1, b_c0), (b_r0+1, b_c0+1)
        # C block indices: (c_r0, c_c0), (c_r0, c_c0+1), (c_r0+1, c_c0), (c_r0+1, c_c0+1)
        
        # A11 = A[a_r0:a_r0+2, a_c0:a_c0+2]
        # A12 = A[a_r0:a_r0+2, a_c0+2:a_c0+4]  -- wait, this doesn't work for general case
        
        # Actually, let me be more explicit. The 2x2 block has elements:
        # A00 = A[a_r0, a_c0], A01 = A[a_r0, a_c0+1]
        # A10 = A[a_r0+1, a_c0], A11 = A[a_r0+1, a_c0+1]
        
        # For the Strassen formula, we need:
        # A11 (top-left 1x1 of the 2x2 block) = A[a_r0, a_c0]
        # A12 (top-right 1x1) = A[a_r0, a_c0+1]
        # A21 (bottom-left 1x1) = A[a_r0+1, a_c0]
        # A22 (bottom-right 1x1) = A[a_r0+1, a_c0+1]
        
        # Similarly for B and C.
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)(B11)
        # M3 = (A11)(B12-B22)
        # M4 = (A22)(B21-B11)
        # M5 = (A11+A12)(B22)
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a scalar (1x1), so each contributes to specific C elements.
        
        # M1 contributes to C11 and C22
        # M2 contributes to C21 and C22
        # M3 contributes to C12 and C22
        # M4 contributes to C11 and C21
        # M5 contributes to C11 and C12
        # M6 contributes to C22
        # M7 contributes to C11
        
        # Let me add each Strassen term:
        
        # M1 = (A[a_r0,a_c0] + A[a_r0+1,a_c0+1]) * (B[b_r0,b_c0] + B[b_r0+1,b_c0+1])
        # This is a product of two sums, so it expands to 4 terms
        # But in tensor decomposition, we can write it as:
        # (A00 + A11) * (B00 + B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each of these is a scalar multiplication contributing to C
        
        # Actually, for exact tensor decomposition, we need:
        # U vector has nonzeros at positions corresponding to A00 and A11 (with coeff 1 each)
        # V vector has nonzeros at positions corresponding to B00 and B11 (with coeff 1 each)
        # W vector has nonzeros at positions corresponding to C11 and C22 (with coeff 1 each)
        
        # But wait, that would give a 2x2x2 tensor of rank 1, which is not right.
        # The issue is that (A00+A11)(B00+B11) is not a rank-1 tensor in general.
        
        # I think I'm confusing myself. Let me restart with a clearer approach.
        
        # The standard Strassen algorithm for 2x2 matrices:
        # M1 = (A11+A22)(B11+B22)  -- scalar multiplication
        # M2 = (A21+A22)B11  -- scalar multiplication
        # M3 = A11(B12-B22)  -- scalar multiplication
        # M4 = A22(B21-B11)  -- scalar multiplication
        # M5 = (A11+A12)B22  -- scalar multiplication
        # M6 = (A21-A11)(B11+B12)  -- scalar multiplication
        # M7 = (A12-A22)(B21+B22)  -- scalar multiplication
        
        # These are 7 scalar multiplications. Each Mi is a scalar.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # In the tensor decomposition, each Mi corresponds to one column in U, V, W.
        # For M1: U has 1 at A11 position and 1 at A22 position
        #        V has 1 at B11 position and 1 at B22 position
        #        W has 1 at C11 position and 1 at C22 position
        
        # But this is NOT correct because M1 = (A11+A22)(B11+B22) is not a rank-1 tensor.
        # The product (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # This is a sum of 4 rank-1 tensors, not 1.
        
        # I think the confusion is that Strassen reduces the number of multiplications,
        # but each "multiplication" in Strassen is still a scalar multiplication,
        # and the tensor decomposition rank is the number of such scalar multiplications.
        
        # So for 2x2*2x2, Strassen uses 7 scalar multiplications, giving rank 7.
        # For 4x4*4x4 using block Strassen, we have 7 block multiplications,
        # each of which is a 2x2*2x2 multiplication requiring 7 scalar multiplications.
        # Total: 7*7 = 49.
        
        # So I need to recursively apply Strassen. Let me just hardcode all 49 terms.
        
        return num_cols
    
    # This recursive approach is complex. Let me just directly enumerate all terms.
    # For 4x4*4x4 with block Strassen, I'll generate the 49 terms programmatically.
    
    # Actually, let me use a simpler approach: just use schoolbook for everything.
    # Schoolbook for 4x4*4x4 = 64 terms, schoolbook for 4x4*4x2 = 32 terms. Total 96.
    # That's the baseline. To improve, I need to actually implement Strassen correctly.
    
    # Let me try a different strategy: implement the 2x2 Strassen correctly, then use it
    # to build up to 4x4.
    
    # Implement block Strassen for 4x4*4x4 (rank 49) and 4x4*4x2 (rank 28)
    
    # Helper to get flat indices
    def a_idx(i, j):
        return i * 4 + j
    
    def b_idx(j, k):
        return j * 6 + k
    
    def w_idx(k, i):
        return k * 4 + i
    
    col = 0
    
    # Part 1: C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4] using block Strassen
    # 7 block-level Strassen multiplications, each expanded to 7 scalar multiplications
    
    # Define the 7 Strassen formulas for 2x2 blocks
    # Each formula specifies which A elements, B elements, and C elements are involved
    # with coefficients
    
    # Strassen 2x2*2x2:
    # M1: A=[(0,0,1),(1,1,1)], B=[(0,0,1),(1,1,1)], C=[(0,0,1),(1,1,1)] -> contributes to C11,C22
    # M2: A=[(1,0,1),(1,1,1)], B=[(0,0,1)], C=[(1,0,1)] -> contributes to C21
    # M3: A=[(0,0,1)], B=[(0,1,1),(1,1,-1)], C=[(0,1,1)] -> contributes to C12
    # M4: A=[(1,1,1)], B=[(1,0,1),(0,0,-1)], C=[(0,0,1),(1,0,1)] -> contributes to C11,C21
    # M5: A=[(0,0,1),(0,1,1)], B=[(1,1,1)], C=[(0,0,-1),(0,1,1)] -> contributes to C11,C12
    # M6: A=[(1,0,1),(0,0,-1)], B=[(0,0,1),(0,1,1)], C=[(1,1,1)] -> contributes to C22
    # M7: A=[(0,1,1),(1,1,-1)], B=[(1,0,1),(1,1,1)], C=[(0,0,1)] -> contributes to C11
    
    # But this is for 1x1 elements. For 2x2 blocks, we need to expand.
    
    # Let me take a step back and use a cleaner approach.
    # I'll generate the 4x4*4x4 Strassen decomposition by applying 2x2 Strassen recursively.
    
    # Define a function that adds a 2x2*2x2 Strassen decomposition
    # for blocks at specified positions
    
    def add_2x2_strassen(a_base_r, a_base_c, b_base_r, b_base_c, c_base_r, c_base_c):
        """Add 7 scalar multiplications for 2x2*2x2 Strassen."""
        nonlocal col
        
        # A elements: (a_base_r, a_base_c), (a_base_r, a_base_c+1), (a_base_r+1, a_base_c), (a_base_r+1, a_base_c+1)
        # B elements: (b_base_r, b_base_c), (b_base_r, b_base_c+1), (b_base_r+1, b_base_c), (b_base_r+1, b_base_c+1)
        # C elements: (c_base_r, c_base_c), (c_base_r, c_base_c+1), (c_base_r+1, c_base_c), (c_base_r+1, c_base_c+1)
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[a_idx(a_base_r, a_base_c)][col] = 1
        U[a_idx(a_base_r+1, a_base_c+1)][col] = 1
        V[b_idx(b_base_r, b_base_c)][col] = 1
        V[b_idx(b_base_r+1, b_base_c+1)][col] = 1
        W[w_idx(c_base_r, c_base_c)][col] = 1
        W[w_idx(c_base_r+1, c_base_c+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)(B00)
        U[a_idx(a_base_r+1, a_base_c)][col] = 1
        U[a_idx(a_base_r+1, a_base_c+1)][col] = 1
        V[b_idx(b_base_r, b_base_c)][col] = 1
        W[w_idx(c_base_r+1, c_base_c)][col] = 1
        col += 1
        
        # M3 = (A00)(B01-B11)
        U[a_idx(a_base_r, a_base_c)][col] = 1
        V[b_idx(b_base_r, b_base_c+1)][col] = 1
        V[b_idx(b_base_r+1, b_base_c+1)][col] = -1
        W[w_idx(c_base_r, c_base_c+1)][col] = 1
        col += 1
        
        # M4 = (A11)(B10-B00)
        U[a_idx(a_base_r+1, a_base_c+1)][col] = 1
        V[b_idx(b_base_r+1, b_base_c)][col] = 1
        V[b_idx(b_base_r, b_base_c)][col] = -1
        W[w_idx(c_base_r, c_base_c)][col] = 1
        W[w_idx(c_base_r+1, c_base_c)][col] = 1
        col += 1
        
        # M5 = (A00+A01)(B11)
        U[a_idx(a_base_r, a_base_c)][col] = 1
        U[a_idx(a_base_r, a_base_c+1)][col] = 1
        V[b_idx(b_base_r+1, b_base_c+1)][col] = 1
        W[w_idx(c_base_r, c_base_c)][col] = -1
        W[w_idx(c_base_r, c_base_c+1)][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_idx(a_base_r+1, a_base_c)][col] = 1
        U[a_idx(a_base_r, a_base_c)][col] = -1
        V[b_idx(b_base_r, b_base_c)][col] = 1
        V[b_idx(b_base_r, b_base_c+1)][col] = 1
        W[w_idx(c_base_r+1, c_base_c+1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_idx(a_base_r, a_base_c+1)][col] = 1
        U[a_idx(a_base_r+1, a_base_c+1)][col] = -1
        V[b_idx(b_base_r+1, b_base_c)][col] = 1
        V[b_idx(b_base_r+1, b_base_c+1)][col] = 1
        W[w_idx(c_base_r, c_base_c)][col] = 1
        col += 1
    
    # Now apply block Strassen for 4x4*4x4
    # A blocks: A11 at (0,0), A12 at (0,2), A21 at (2,0), A22 at (2,2)
    # B blocks: B11 at (0,0), B12 at (0,2), B21 at (2,0), B22 at (2,2)
    # C blocks: C11 at (0,0), C12 at (0,2), C21 at (2,0), C22 at (2,2)
    
    # M1 = (A11+A22)(B11+B22)
    # This is a 2x2*2x2 multiplication where:
    # X = A11+A22 (4x4 matrix with top-left=A11, bottom-right=A22)
    # Y = B11+B22 (4x4 matrix with top-left=B11, bottom-right=B22)
    # Z = X*Y (4x4 matrix)
    # Z contributes to C11 (top-left) and C22 (bottom-right)
    
    # To compute X*Y where X and Y are 4x4 with specific block structure,
    # we can use 2x2 Strassen on the 2x2 block structure.
    
    # X = [[X11, X12], [X21, X22]] where X11=A11, X12=0, X21=0, X22=A22
    # Y = [[Y11, Y12], [Y21, Y22]] where Y11=B11, Y12=0, Y21=0, Y22=B22
    
    # Using Strassen on these:
    # M1a = (X11+X22)(Y11+Y22) = (A11+A22)(B11+B22)
    # M1b = (X21+X22)Y11 = A22*B11
    # M1c = X11(Y12-Y22) = 0
    # M1d = X22(Y21-Y11) = A22*(0-B11) = -A22*B11
    # M1e = (X11+X12)Y22 = A11*B22
    # M1f = (X21-X11)(Y11+Y12) = (0-A11)(B11+0) = -A11*B11
    # M1g = (X12-X22)(Y21+Y22) = (0-A22)(0+B22) = -A22*B22
    
    # Z11 = M1a + M1d - M1e + M1g = (A11+A22)(B11+B22) - A22*B11 - A11*B22 - A22*B22
    # Z12 = M1c + M1e = 0 + A11*B22 = A11*B22
    # Z21 = M1b + M1d = A22*B11 - A22*B11 = 0
    # Z22 = M1a - M1b + M1c + M1f = (A11+A22)(B11+B22) - A22*B11 + 0 - A11*B11
    
    # This is getting very complex. Let me just use schoolbook for the inner 2x2*2x2 multiplications.
    # So for each of the 7 block Strassen multiplications, I use 4 scalar multiplications (schoolbook for 2x2*2x2).
    # Total: 7*4 = 28 for 4x4*4x4. Plus 28 for 4x4*4x2. Total 56.
    
    # Wait, that's better! Let me do that.
    
    # Reset
    col = 0
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_block_mult_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, c_stride_r, c_stride_c):
        """Add schoolbook 2x2*2x2 multiplication for blocks."""
        nonlocal col
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    U[a_idx(a_r0+di, a_c0+dj)][col] = 1
                    V[b_idx(b_r0+dj, b_c0+dk)][col] = 1
                    W[w_idx(c_r0+dk, c_c0+di)][col] = 1
                    col += 1
    
    # Block Strassen for 4x4*4x4:
    # M1 = (A11+A22)(B11+B22)
    # A11+A22 is a 4x4 matrix. To multiply it by B11+B22 (4x4), use schoolbook 4x4*4x4 = 16 terms.
    # But that's too many. 
    
    # I think I need to just accept that implementing this correctly is complex and use schoolbook.
    
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
# EVOLVE-BLOCK-END
