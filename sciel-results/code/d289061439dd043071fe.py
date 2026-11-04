"""
Program to find a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.

Strategy: Use a partition of the 5x5 matrix into a 4x4 block and a 1x1 block.
- The 4x4x4 core uses Strassen's algorithm (rank 49).
- Cross terms between the 4x4 part and the 1x1 part use schoolbook.
- Total rank: 49 (core) + 16 (A4x4 * B1x1) + 16 (A1x1 * B4x4) + 4 (A4x4 rows * B1x1 cols) 
  + 4 (A1x1 rows * B4x4 cols) + 1 (1x1*1x1) + 16 (A4x4*1x1 for C cols) + 16 (1x1*4x4 for C rows) ...
  
  Let's count carefully:
  A = [[A00 (4x4), A01 (4x1)],
       [A10 (1x4), A11 (1x1)]]
  B = [[B00 (4x4), B01 (4x1)],
       [B10 (1x4), B11 (1x1)]]
  C = [[C00 (4x4), C01 (4x1)],
       [C10 (1x4), C11 (1x1)]]

  C00 = A00*B00 (rank 49 via Strassen)
  C01 = A00*B01 + A01*B11
       A00*B01: 4x4 * 4x1 -> 16 terms (schoolbook)
       A01*B11: 4x1 * 1x1 -> 4 terms (schoolbook)
       Total C01: 20 terms
  C10 = A10*B00 + A11*B10
       A10*B00: 1x4 * 4x4 -> 16 terms (schoolbook)
       A11*B10: 1x1 * 1x4 -> 4 terms (schoolbook)
       Total C10: 20 terms
  C11 = A10*B01 + A11*B11
       A10*B01: 1x4 * 4x1 -> 4 terms (schoolbook)
       A11*B11: 1x1 * 1x1 -> 1 term (schoolbook)
       Total C11: 5 terms

  Total Rank = 49 + 20 + 20 + 5 = 94.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Strategy: Partition indices as {0,1,2,3} (4x4 part) and {4} (1x1 part).
    Use Strassen's algorithm for the 4x4x4 multiplication (rank 49).
    Use schoolbook for all cross terms involving the 1x1 part.
    
    Total rank: 49 + 20 + 20 + 5 = 94
    """
    n, m, p = 5, 5, 5
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    rank = 94
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    
    # Helper: add a schoolbook term for A[i,j] * B[j,k] -> C[i,k]
    def add_schoolbook_single(i, j, k):
        nonlocal col
        if col < rank:
            U[i * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Helper: add Strassen's 7 terms for 2x2 matrix multiplication
    # A at rows i1,i2 and cols j1,j2
    # B at rows j1,j2 and cols k1,k2
    # C at rows i1,i2 and cols k1,k2
    def add_strassen_2x2(i1, j1, k1):
        nonlocal col
        i2 = i1 + 1
        j2 = j1 + 1
        k2 = k1 + 1
        
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # Contributes to C[i1,k1] (+1) and C[i2,k2] (+1)
        if col < rank:
            U[i1 * m + j1, col] = 1.0
            U[i2 * m + j2, col] = 1.0
            V[j1 * p + k1, col] = 1.0
            V[j2 * p + k2, col] = 1.0
            W[k1 * n + i1, col] = 1.0
            W[k2 * n + i2, col] = 1.0
            col += 1
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        # Contributes to C[i2,k1] (+1)
        if col < rank:
            U[i2 * m + j1, col] = 1.0
            U[i2 * m + j2, col] = 1.0
            V[j1 * p + k1, col] = 1.0
            W[k1 * n + i2, col] = 1.0
            col += 1
        
        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        # Contributes to C[i1,k2] (+1)
        if col < rank:
            U[i1 * m + j1, col] = 1.0
            V[j1 * p + k2, col] = 1.0
            V[j2 * p + k2, col] = -1.0
            W[k2 * n + i1, col] = 1.0
            col += 1
        
        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        # Contributes to C[i2,k1] (+1) and C[i1,k1] (-1)
        if col < rank:
            U[i2 * m + j2, col] = 1.0
            V[j2 * p + k1, col] = 1.0
            V[j1 * p + k1, col] = -1.0
            W[k1 * n + i2, col] = 1.0
            W[k1 * n + i1, col] = -1.0
            col += 1
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        # Contributes to C[i1,k2] (+1) and C[i1,k1] (-1)
        if col < rank:
            U[i1 * m + j1, col] = 1.0
            U[i1 * m + j2, col] = 1.0
            V[j2 * p + k2, col] = 1.0
            W[k2 * n + i1, col] = 1.0
            W[k1 * n + i1, col] = -1.0
            col += 1
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        # Contributes to C[i2,k2] (+1)
        if col < rank:
            U[i2 * m + j1, col] = 1.0
            U[i1 * m + j1, col] = -1.0
            V[j1 * p + k1, col] = 1.0
            V[j1 * p + k2, col] = 1.0
            W[k2 * n + i2, col] = 1.0
            col += 1
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        # Contributes to C[i1,k1] (+1) and C[i2,k2] (-1)
        if col < rank:
            U[i1 * m + j2, col] = 1.0
            U[i2 * m + j2, col] = -1.0
            V[j2 * p + k1, col] = 1.0
            V[j2 * p + k2, col] = 1.0
            W[k1 * n + i1, col] = 1.0
            W[k2 * n + i2, col] = -1.0
            col += 1

    # --- 4x4 Core (Strassen) ---
    # The 4x4 core involves indices 0,1,2,3.
    # We apply Strassen to the 2x2 blocks of the 4x4 matrix.
    # The 4x4 matrix is partitioned into four 2x2 blocks:
    # Block (0,0): rows 0-1, cols 0-1
    # Block (0,1): rows 0-1, cols 2-3
    # Block (1,0): rows 2-3, cols 0-1
    # Block (1,1): rows 2-3, cols 2-3
    
    # We need to add Strassen terms for the 4x4 product.
    # This is done by applying the 2x2 Strassen algorithm to each of the 7 top-level products,
    # where each top-level product is itself a 2x2*2x2 multiplication.
    # However, the indices in the 2x2 Strassen helper refer to global indices.
    # So for the top-level Strassen of the 4x4 block:
    # M1 = (A00+A11)(B00+B11)
    # This is a 2x2*2x2 product where the "A" is formed from blocks A00 and A11.
    # But my helper `add_strassen_2x2` assumes a contiguous 2x2 block.
    # A00 is at (0,0)-(1,1) and A11 is at (2,2)-(3,3). They are not contiguous in a 2x2 sense for a single helper call.
    
    # I need a more flexible approach for the 4x4 Strassen.
    # I will manually define the 49 terms for the 4x4 core.
    # Or, I can use a recursive helper that takes lists of indices.
    
    # Let's write a helper that adds Strassen terms for a 2x2*2x2 product
    # where the indices are given explicitly.
    
    def add_strassen_2x2_explicit(A_idx, B_idx, C_idx):
        """
        A_idx: list of 4 tuples (i, j) for the 2x2 A block
        B_idx: list of 4 tuples (j, k) for the 2x2 B block
        C_idx: list of 4 tuples (i, k) for the 2x2 C block
        """
        nonlocal col
        
        # A00, A01, A10, A11
        A00, A01, A10, A11 = A_idx
        # B00, B01, B10, B11
        B00, B01, B10, B11 = B_idx
        # C00, C01, C10, C11
        C00, C01, C10, C11 = C_idx
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        if col < rank:
            U[A00[0] * m + A00[1], col] = 1.0
            U[A11[0] * m + A11[1], col] = 1.0
            V[B00[0] * p + B00[1], col] = 1.0
            V[B11[0] * p + B11[1], col] = 1.0
            W[C00[1] * n + C00[0], col] = 1.0
            W[C11[1] * n + C11[0], col] = 1.0
            col += 1
            
        # M2 = (A10+A11)B00 -> C10
        if col < rank:
            U[A10[0] * m + A10[1], col] = 1.0
            U[A11[0] * m + A11[1], col] = 1.0
            V[B00[0] * p + B00[1], col] = 1.0
            W[C10[1] * n + C10[0], col] = 1.0
            col += 1
            
        # M3 = A00(B01-B11) -> C01
        if col < rank:
            U[A00[0] * m + A00[1], col] = 1.0
            V[B01[0] * p + B01[1], col] = 1.0
            V[B11[0] * p + B11[1], col] = -1.0
            W[C01[1] * n + C01[0], col] = 1.0
            col += 1
            
        # M4 = A11(B10-B00) -> C10, C00
        if col < rank:
            U[A11[0] * m + A11[1], col] = 1.0
            V[B10[0] * p + B10[1], col] = 1.0
            V[B00[0] * p + B00[1], col] = -1.0
            W[C10[1] * n + C10[0], col] = 1.0
            W[C00[1] * n + C00[0], col] = -1.0
            col += 1
            
        # M5 = (A00+A01)B11 -> C01, C00
        if col < rank:
            U[A00[0] * m + A00[1], col] = 1.0
            U[A01[0] * m + A01[1], col] = 1.0
            V[B11[0] * p + B11[1], col] = 1.0
            W[C01[1] * n + C01[0], col] = 1.0
            W[C00[1] * n + C00[0], col] = -1.0
            col += 1
            
        # M6 = (A10-A00)(B00+B01) -> C11
        if col < rank:
            U[A10[0] * m + A10[1], col] = 1.0
            U[A00[0] * m + A00[1], col] = -1.0
            V[B00[0] * p + B00[1], col] = 1.0
            V[B01[0] * p + B01[1], col] = 1.0
            W[C11[1] * n + C11[0], col] = 1.0
            col += 1
            
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        if col < rank:
            U[A01[0] * m + A01[1], col] = 1.0
            U[A11[0] * m + A11[1], col] = -1.0
            V[B10[0] * p + B10[1], col] = 1.0
            V[B11[0] * p + B11[1], col] = 1.0
            W[C00[1] * n + C00[0], col] = 1.0
            W[C11[1] * n + C11[0], col] = -1.0
            col += 1

    # Define indices for the 4x4 core (indices 0,1,2,3)
    # A blocks:
    A00_idx = [(0,0), (0,1), (1,0), (1,1)]
    A01_idx = [(0,2), (0,3), (1,2), (1,3)]
    A10_idx = [(2,0), (2,1), (3,0), (3,1)]
    A11_idx = [(2,2), (2,3), (3,2), (3,3)]
    
    # B blocks:
    B00_idx = [(0,0), (0,1), (1,0), (1,1)]
    B01_idx = [(0,2), (0,3), (1,2), (1,3)]
    B10_idx = [(2,0), (2,1), (3,0), (3,1)]
    B11_idx = [(2,2), (2,3), (3,2), (3,3)]
    
    # C blocks:
    C00_idx = [(0,0), (0,1), (1,0), (1,1)]
    C01_idx = [(0,2), (0,3), (1,2), (1,3)]
    C10_idx = [(2,0), (2,1), (3,0), (3,1)]
    C11_idx = [(2,2), (2,3), (3,2), (3,3)]
    
    # Now, we need to apply the 4x4 Strassen.
    # The 4x4 Strassen itself consists of 7 products of 2x2 blocks.
    # Each of these 7 products is a 2x2*2x2 multiplication.
    # So we need to call `add_strassen_2x2_explicit` 7 times, each with the appropriate indices.
    # But wait, `add_strassen_2x2_explicit` adds 7 terms.
    # So 7 * 7 = 49 terms total for the 4x4 core.
    
    # Top-level Strassen for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # This is a 2x2*2x2 product where the "A" is the sum of A00 and A11.
    # In the tensor decomposition, this means the u-vector has non-zero entries for both A00 and A11.
    # My `add_strassen_2x2_explicit` helper takes explicit indices for the 2x2 blocks.
    # For M1, the "A" block is not a single 2x2 block in the matrix, but a combination.
    # However, the structure of the 2x2 Strassen terms is local to the 2x2 indices.
    # The indices in `add_strassen_2x2_explicit` are the positions of the 4 elements of the 2x2 block.
    # For M1, the "A" 2x2 block consists of elements from A00 and A11.
    # Specifically:
    #   A_M1[0,0] = A[0,0] + A[2,2]
    #   A_M1[0,1] = A[0,1] + A[2,3]
    #   A_M1[1,0] = A[1,0] + A[3,2]
    #   A_M1[1,1] = A[1,1] + A[3,3]
    # This is not a standard 2x2 block. My helper assumes the 4 indices form a 2x2 block.
    # I need to generalize the helper to accept arbitrary indices for the 4 elements of the 2x2 A, B, and C matrices.
    
    # Let's redefine the helper to accept lists of indices for the 4 elements of A, B, and C.
    
    def add_strassen_2x2_general(A_elems, B_elems, C_elems):
        """
        A_elems: list of 4 (i, j) tuples corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
        B_elems: list of 4 (j, k) tuples corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
        C_elems: list of 4 (i, k) tuples corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
        """
        nonlocal col
        
        A00, A01, A10, A11 = A_elems
        B00, B01, B10, B11 = B_elems
        C00, C01, C10, C11 = C_elems
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        if col < rank:
            U[A00[0] * m + A00[1], col] = 1.0
            U[A11[0] * m + A11[1], col] = 1.0
            V[B00[0] * p + B00[1], col] = 1.0
            V[B11[0] * p + B11[1], col] = 1.0
            W[C00[1] * n + C00[0], col] = 1.0
            W[C11[1] * n + C11[0], col] = 1.0
            col += 1
            
        # M2 = (A10+A11)B00 -> C10
        if col < rank:
            U[A10[0] * m + A10[1], col] = 1.0
            U[A11[0] * m + A11[1], col] = 1.0
            V[B00[0] * p + B00[1], col] = 1.0
            W[C10[1] * n + C10[0], col] = 1.0
            col += 1
            
        # M3 = A00(B01-B11) -> C01
        if col < rank:
            U[A00[0] * m + A00[1], col] = 1.0
            V[B01[0] * p + B01[1], col] = 1.0
            V[B11[0] * p + B11[1], col] = -1.0
            W[C01[1] * n + C01[0], col] = 1.0
            col += 1
            
        # M4 = A11(B10-B00) -> C10, C00
        if col < rank:
            U[A11[0] * m + A11[1], col] = 1.0
            V[B10[0] * p + B10[1], col] = 1.0
            V[B00[0] * p + B00[1], col] = -1.0
            W[C10[1] * n + C10[0], col] = 1.0
            W[C00[1] * n + C00[0], col] = -1.0
            col += 1
            
        # M5 = (A00+A01)B11 -> C01, C00
        if col < rank:
            U[A00[0] * m + A00[1], col] = 1.0
            U[A01[0] * m + A01[1], col] = 1.0
            V[B11[0] * p + B11[1], col] = 1.0
            W[C01[1] * n + C01[0], col] = 1.0
            W[C00[1] * n + C00[0], col] = -1.0
            col += 1
            
        # M6 = (A10-A00)(B00+B01) -> C11
        if col < rank:
            U[A10[0] * m + A10[1], col] = 1.0
            U[A00[0] * m + A00[1], col] = -1.0
            V[B00[0] * p + B00[1], col] = 1.0
            V[B01[0] * p + B01[1], col] = 1.0
            W[C11[1] * n + C11[0], col] = 1.0
            col += 1
            
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        if col < rank:
            U[A01[0] * m + A01[1], col] = 1.0
            U[A11[0] * m + A11[1], col] = -1.0
            V[B10[0] * p + B10[1], col] = 1.0
            V[B11[0] * p + B11[1], col] = 1.0
            W[C00[1] * n + C00[0], col] = 1.0
            W[C11[1] * n + C11[0], col] = -1.0
            col += 1

    # Now apply the 4x4 Strassen.
    # The 4x4 matrix is partitioned into 2x2 blocks.
    # The top-level Strassen has 7 products.
    # For each product, we need to specify the A, B, C elements.
    
    # M1 = (A00+A11)(B00+B11)
    # A elements: A00[0,0], A00[0,1], A00[1,0], A00[1,1] and A11[0,0], A11[0,1], A11[1,0], A11[1,1]
    # Wait, the "A" in M1 is a 2x2 matrix where each element is a sum of two elements from A.
    # So A_M1[0,0] = A[0,0] + A[2,2]
    # This means the u-vector for the Strassen terms will have entries for both A[0,0] and A[2,2].
    # My `add_strassen_2x2_general` helper takes 4 elements for A.
    # But here, each "element" of the 2x2 A is actually a sum of two elements.
    # This is not directly supported by the helper which assumes each index is a single element.
    
    # I need to modify the approach.
    # Instead of using the helper, I will manually construct the 49 terms.
    # Or, I can use a different helper that takes linear combinations.
    
    # Let's try a different strategy: use schoolbook for the 4x4 part as well, but optimize the cross terms.
    # Rank 64 (4x4 schoolbook) + 20 + 20 + 5 = 109.
    # This is better than 125, but worse than 94.
    
    # Let's stick with the 94 rank strategy but implement the 4x4 Strassen correctly.
    # I will create a helper that adds Strassen terms for a 2x2*2x2 product where the A, B, C elements
    # can be linear combinations of the matrix elements.
    
    # Actually, the standard way to implement Strassen for 4x4 is to view the 4x4 matrices as 2x2 matrices of 2x2 blocks.
    # Then apply the 7 Strassen formulas to these blocks.
    # Each formula involves 2x2 block multiplications.
    # Each 2x2 block multiplication can be decomposed into 7 rank-1 terms using Strassen.
    # So we have 7 * 7 = 49 terms.
    
    # The key is that the "elements" in the 2x2 block multiplication are themselves 2x2 matrices.
    # So when we apply Strassen to the block multiplication, the "indices" in the u, v, w vectors
    # will refer to the elements of the 2x2 blocks.
    
    # Let's define the 2x2 blocks explicitly.
    # Block (bi, bj) for A:
    # bi=0, bj=0: indices [(0,0), (0,1), (1,0), (1,1)]
    # bi=0, bj=1: indices [(0,2), (0,3), (1,2), (1,3)]
    # bi=1, bj=0: indices [(2,0), (2,1), (3,0), (3,1)]
    # bi=1, bj=1: indices [(2,2), (2,3), (3,2), (3,3)]
    
    # For the 4x4 Strassen, the 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2*2x2 product.
    # For example, M1 = (A00+A11)(B00+B11).
    # Let X = A00+A11 and Y = B00+B11.
    # X is a 2x2 matrix where X[0,0] = A00[0,0] + A11[0,0], etc.
    # Y is a 2x2 matrix where Y[0,0] = B00[0,0] + B11[0,0], etc.
    # The product Z = X*Y is a 2x2 matrix.
    # Z contributes to C00 and C11.
    # Specifically, C00 = Z + ... and C11 = Z + ...
    
    # To decompose Z = X*Y into rank-1 terms, we apply Strassen to the 2x2*2x2 product.
    # The 7 Strassen terms for Z will have u-vectors with entries from X (which are sums of A00 and A11),
    # v-vectors with entries from Y (which are sums of B00 and B11),
    # and w-vectors with entries from Z (which contribute to C00 and C11).
    
    # So for each of the 7 Strassen terms of Z, we need to add 1 rank-1 term to the global decomposition.
    # The u-vector will have non-zero entries for the elements of A00 and A11.
    # The v-vector will have non-zero entries for the elements of B00 and B11.
    # The w-vector will have non-zero entries for the elements of C00 and C11.
    
    # Let's implement this.
    
    # Define the 2x2 blocks as lists of (i, j) tuples.
    A_blocks = [
        [(0,0), (0,1), (1,0), (1,1)],  # A00
        [(0,2), (0,3), (1,2), (1,3)],  # A01
        [(2,0), (2,1), (3,0), (3,1)],  # A10
        [(2,2), (2,3), (3,2), (3,3)]   # A11
    ]
    B_blocks = [
        [(0,0), (0,1), (1,0), (1,1)],  # B00
        [(0,2), (0,3), (1,2), (1,3)],  # B01
        [(2,0), (2,1), (3,0), (3,1)],  # B10
        [(2,2), (2,3), (3,2), (3,3)]   # B11
    ]
    C_blocks = [
        [(0,0), (0,1), (1,0), (1,1)],  # C00
        [(0,2), (0,3), (1,2), (1,3)],  # C01
        [(2,0), (2,1), (3,0), (3,1)],  # C10
        [(2,2), (2,3), (3,2), (3,3)]   # C11
    ]
    
    # Helper to add Strassen terms for a 2x2*2x2 product where the A, B, C blocks are given.
    # A_blocks: list of 4 (i, j) tuples for the 2x2 A block
    # B_blocks: list of 4 (j, k) tuples for the 2x2 B block
    # C_blocks: list of 4 (i, k) tuples for the 2x2 C block
    # But here, the A and B blocks can be linear combinations of the original blocks.
    
    # I will create a helper that takes the A, B, C blocks as lists of (block_index, coefficient) pairs.
    
    def add_strassen_2x2_blocks(A_terms, B_terms, C_terms):
        """
        A_terms: list of 4 tuples (block_idx, coeff) for the 2x2 A block
        B_terms: list of 4 tuples (block_idx, coeff) for the 2x2 B block
        C_terms: list of 4 tuples (block_idx, coeff) for the 2x2 C block
        """
        nonlocal col
        
        # A00, A01, A10, A11 in terms of blocks
        A00 = A_terms[0]
        A01 = A_terms[1]
        A10 = A_terms[2]
        A11 = A_terms[3]
        
        B00 = B_terms[0]
        B01 = B_terms[1]
        B10 = B_terms[2]
        B11 = B_terms[3]
        
        C00 = C_terms[0]
        C01 = C_terms[1]
        C10 = C_terms[2]
        C11 = C_terms[3]
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        if col < rank:
            # u vector
            for bi, ci in [A00, A11]:
                for idx in A_blocks[bi]:
                    U[idx[0] * m + idx[1], col] += ci
            # v vector
            for bi, ci in [B00, B11]:
                for idx in B_blocks[bi]:
                    V[idx[0] * p + idx[1], col] += ci
            # w vector
            for bi, ci in [C00, C11]:
                for idx in C_blocks[bi]:
                    W[idx[1] * n + idx[0], col] += ci
            col += 1
            
        # M2 = (A10+A11)B00 -> C10
        if col < rank:
            for bi, ci in [A10, A11]:
                for idx in A_blocks[bi]:
                    U[idx[0] * m + idx[1], col] += ci
            for bi, ci in [B00]:
                for idx in B_blocks[bi]:
                    V[idx[0] * p + idx[1], col] += ci
            for bi, ci in [C10]:
                for idx in C_blocks[bi]:
                    W[idx[1] * n + idx[0], col] += ci
            col += 1
            
        # M3 = A00(B01-B11) -> C01
        if col < rank:
            for bi, ci in [A00]:
                for idx in A_blocks[bi]:
                    U[idx[0] * m + idx[1], col] += ci
            for bi, ci in [(B01[0], 1.0), (B11[0], -1.0)]:
                for idx in B_blocks[bi]:
                    V[idx[0] * p + idx[1], col] += ci
            for bi, ci in [C01]:
                for idx in C_blocks[bi]:
                    W[idx[1] * n + idx[0], col] += ci
            col += 1
            
        # M4 = A11(B10-B00) -> C10, C00
        if col < rank:
            for bi, ci in [A11]:
                for idx in A_blocks[bi]:
                    U[idx[0] * m + idx[1], col] += ci
            for bi, ci in [(B10[0], 1.0), (B00[0], -1.0)]:
                for idx in B_blocks[bi]:
                    V[idx[0] * p + idx[1], col] += ci
            for bi, ci in [(C10[0], 1.0), (C00[0], -1.0)]:
                for idx in C_blocks[bi]:
                    W[idx[1] * n + idx[0], col] += ci
            col += 1
            
        # M5 = (A00+A01)B11 -> C01, C00
        if col < rank:
            for bi, ci in [A00, A01]:
                for idx in A_blocks[bi]:
                    U[idx[0] * m + idx[1], col] += ci
            for bi, ci in [B11]:
                for idx in B_blocks[bi]:
                    V[idx[0] * p + idx[1], col] += ci
            for bi, ci in [(C01[0], 1.0), (C00[0], -1.0)]:
                for idx in C_blocks[bi]:
                    W[idx[1] * n + idx[0], col] += ci
            col += 1
            
        # M6 = (A10-A00)(B00+B01) -> C11
        if col < rank:
            for bi, ci in [(A10[0], 1.0), (A00[0], -1.0)]:
                for idx in A_blocks[bi]:
                    U[idx[0] * m + idx[1], col] += ci
            for bi, ci in [B00, B01]:
                for idx in B_blocks[bi]:
                    V[idx[0] * p + idx[1], col] += ci
            for bi, ci in [C11]:
                for idx in C_blocks[bi]:
                    W[idx[1] * n + idx[0], col] += ci
            col += 1
            
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        if col < rank:
            for bi, ci in [(A01[0], 1.0), (A11[0], -1.0)]:
                for idx in A_blocks[bi]:
                    U[idx[0] * m + idx[1], col] += ci
            for bi, ci in [B10, B11]:
                for idx in B_blocks[bi]:
                    V[idx[0] * p + idx[1], col] += ci
            for bi, ci in [(C00[0], 1.0), (C11[0], -1.0)]:
                for idx in C_blocks[bi]:
                    W[idx[1] * n + idx[0], col] += ci
            col += 1

    # Now apply the 4x4 Strassen.
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # A terms: A00 (coeff 1), A11 (coeff 1)
    # B terms: B00 (coeff 1), B11 (coeff 1)
    # C terms: C00 (coeff 1), C11 (coeff 1)
    add_strassen_2x2_blocks(
        [(0, 1.0), (0, 0.0), (0, 0.0), (3, 1.0)],  # A00=1, A01=0, A10=0, A11=1
        [(0, 1.0), (0, 0.0), (0, 0.0), (3, 1.0)],  # B00=1, B01=0, B10=0, B11=1
        [(0, 1.0), (0, 0.0), (0, 0.0), (3, 1.0)]   # C00=1, C01=0, C10=0, C11=1
    )
    
    # M2 = (A10+A11)B00 -> C10
    add_strassen_2x2_blocks(
        [(0, 0.0), (0, 0.0), (2, 1.0), (3, 1.0)],  # A00=0, A01=0, A10=1, A11=1
        [(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)],  # B00=1, B01=0, B10=0, B11=0
        [(0, 0.0), (0, 0.0), (2, 1.0), (0, 0.0)]   # C00=0, C01=0, C10=1, C11=0
    )
    
    # M3 = A00(B01-B11) -> C01
    add_strassen_2x2_blocks(
        [(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)],  # A00=1, A01=0, A10=0, A11=0
        [(0, 0.0), (1, 1.0), (0, 0.0), (3, -1.0)], # B00=0, B01=1, B10=0, B11=-1
        [(0, 0.0), (1, 1.0), (0, 0.0), (0, 0.0)]   # C00=0, C01=1, C10=0, C11=0
    )
    
    # M4 = A11(B10-B00) -> C10, C00
    add_strassen_2x2_blocks(
        [(0, 0.0), (0, 0.0), (0, 0.0), (3, 1.0)],  # A00=0, A01=0, A10=0, A11=1
        [(0, -1.0), (0, 0.0), (2, 1.0), (0, 0.0)], # B00=-1, B01=0, B10=1, B11=0
        [(0, -1.0), (0, 0.0), (2, 1.0), (0, 0.0)]  # C00=-1, C01=0, C10=1, C11=0
    )
    
    # M5 = (A00+A01)B11 -> C01, C00
    add_strassen_2x2_blocks(
        [(0, 1.0), (1, 1.0), (0, 0.0), (0, 0.0)],  # A00=1, A01=1, A10=0, A11=0
        [(0, 0.0), (0, 0.0), (0, 0.0), (3, 1.0)],  # B00=0, B01=0, B10=0, B11=1
        [(0, -1.0), (1, 1.0), (0, 0.0), (0, 0.0)]  # C00=-1, C01=1, C10=0, C11=0
    )
    
    # M6 = (A10-A00)(B00+B01) -> C11
    add_strassen_2x2_blocks(
        [(0, -1.0), (0, 0.0), (2, 1.0), (0, 0.0)], # A00=-1, A01=0, A10=1, A11=0
        [(0, 1.0), (1, 1.0), (0, 0.0), (0, 0.0)],  # B00=1, B01=1, B10=0, B11=0
        [(0, 0.0), (0, 0.0), (0, 0.0), (3, 1.0)]   # C00=0, C01=0, C10=0, C11=1
    )
    
    # M7 = (A01-A11)(B10+B11) -> C00, C11
    add_strassen_2x2_blocks(
        [(0, 0.0), (1, 1.0), (0, 0.0), (3, -1.0)], # A00=0, A01=1, A10=0, A11=-1
        [(0, 0.0), (0, 0.0), (2, 1.0), (3, 1.0)],  # B00=0, B01=0, B10=1, B11=1
        [(0, 1.0), (0, 0.0), (0, 0.0), (3, -1.0)]  # C00=1, C01=0, C10=0, C11=-1
    )
    
    # Now add the cross terms using schoolbook.
    # C01 = A00*B01 + A01*B11
    # A00*B01: 4x4 * 4x1 -> 16 terms
    for i in range(4):
        for j in range(4):
            add_schoolbook_single(i, j, 4)
    
    # A01*B11: 4x1 * 1x1 -> 4 terms
    for i in range(4):
        add_schoolbook_single(i, 4, 4)
    
    # C10 = A10*B00 + A11*B10
    # A10*B00: 1x4 * 4x4 -> 16 terms
    for j in range(4):
        for k in range(4):
            add_schoolbook_single(4, j, k)
    
    # A11*B10: 1x1 * 1x4 -> 4 terms
    for k in range(4):
        add_schoolbook_single(4, 4, k)
    
    # C11 = A10*B01 + A11*B11
    # A10*B01: 1x4 * 4x1 -> 4 terms
    for j in range(4):
        add_schoolbook_single(4, j, 4)
    
    # A11*B11: 1x1 * 1x1 -> 1 term
    add_schoolbook_single(4, 4, 4)
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
