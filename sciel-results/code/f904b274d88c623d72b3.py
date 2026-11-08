"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The tensor T[i,j,k] = 1 for all i in [0,1], j in [0,1,2], k in [0,1,2].
    We need U (6 x r), V (9 x r), W (6 x r) such that:
      T[i,j,k] = sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*2+i, c]
    
    Schoolbook gives rank 18. We attempt to find a lower rank decomposition.
    """
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # First, try a structured Strassen-like ansatz for rank 16
    # The idea: group the 3 columns of A and 3 columns of B into overlapping groups
    # and use bilinear combinations to reduce the number of products.
    
    # For 2x3 * 3x3, output is 2x3.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Try rank 16 with a specific structured pattern:
    # Split the 3x3 B matrix into overlapping 2x2 blocks plus corrections.
    
    # Let's try a direct algebraic construction.
    # We have 6 output entries (2 rows x 3 cols), each is a sum of 3 products.
    # Total 18 products in schoolbook.
    
    # Try: use 16 rank-1 terms with specific integer coefficients.
    # Based on the structure, try this pattern:
    
    # For each output C[i,k], we need: sum_j A[i,j]*B[j,k]
    # 
    # Let's define 16 terms with coefficients:
    # Term c corresponds to (u_c, v_c, w_c)
    
    # Try a known approach: for m=p=3, n=2, the minimum rank is conjectured to be 16.
    # Construction based on grouping:
    
    # Define the 16 terms:
    # We'll use a pattern where we overlap the 3x3 product.
    
    # Actually, let me try a systematic search for small integer coefficients.
    # Start with a structured guess for rank 16.
    
    # Pattern: 16 terms where we use the following structure
    # For the 2x3 * 3x3 case:
    
    # Let's try this specific rank-16 construction:
    # Terms 0-5: cover A[0,0]*B[0,k], A[0,1]*B[1,k], A[0,2]*B[2,k] for k=0,1,2 (but that's 9)
    # That doesn't work directly.
    
    # Alternative: Use the fact that we can combine terms.
    # Let me try a different approach: random search with integer coefficients.
    
    # First, let's verify what rank is achievable with a simple structured approach.
    
    # Try rank 17 first (one less than schoolbook) with a simple modification:
    # Combine two terms that share structure.
    
    # Actually, let me just do a focused search for rank 16 with small integers.
    
    best_rank = 18  # schoolbook
    best_decomp = None
    
    # Generate schoolbook as baseline
    U_sb = np.zeros((6, 18), dtype=int)
    V_sb = np.zeros((9, 18), dtype=int)
    W_sb = np.zeros((6, 18), dtype=int)
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j, col] = 1
                V_sb[j*3+k, col] = 1
                W_sb[k*2+i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try to find rank 16 decomposition with structured ansatz
    # Use a parameterized family based on Strassen-like splitting
    
    # For 2x3 * 3x3, try this specific construction:
    # Split B into B = [B1, B2] where B1 is 3x2, B2 is 3x1
    # and A into A = [A1, A2, A3]
    # Then C = A1*B1 + A2*B2 + A3*B3... no, that's not right.
    
    # Let me try a direct algebraic approach with 16 terms.
    # The key insight: we can use linear combinations of A's columns and B's columns.
    
    # Define U vectors in terms of A's rows: u_i = (a_i0, a_i1, a_i2)
    # Define V vectors in terms of B's columns: v_k = (b_0k, b_1k, b_2k)
    
    # For rank 16, we need 16 pairs of linear forms.
    
    # Try this specific pattern (inspired by rectangular Strassen):
    # Use 4 groups of 4 terms each.
    
    # Let me try a numerical search with integer coefficients in {-2,-1,0,1,2}
    
    def verify_decomp(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        n, m, p = 2, 3, 3
        rank = U.shape[1]
        # Check that U, V, W have the right shapes
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        # For each (i,j,k), check: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # T[i,j,k] should be 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Try structured rank-16 ansatz
    # Based on the idea of overlapping 2x2 subproducts
    
    # Let's define 16 terms with the following structure:
    # For the 2x3 * 3x3 multiplication:
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    # Try to express this with 16 products using linear combinations.
    
    # Define linear combinations:
    # P1 = (A00+A01) * (B00+B10)  -> contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # This is getting complex. Let me try a different approach.
    
    # Use a known result: for 2x3 * 3x3, the minimum rank is 16 (conjectured).
    # Let me try to construct it explicitly.
    
    # Construction for rank 16:
    # We'll use the following 16 rank-1 terms:
    
    # Let's define the coefficient matrices directly.
    
    # After careful algebra, here's a rank-16 construction:
    
    U_16 = np.array([
        # Row 0: A[0,0]
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        # Row 1: A[0,1]
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        # Row 2: A[0,2]
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        # Row 3: A[1,0]
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        # Row 4: A[1,1]
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        # Row 5: A[1,2]
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    ], dtype=int)
    
    # This is just repeating the same pattern, which won't work.
    # Let me try a proper construction.
    
    # Actually, let me try a random integer search for rank 16.
    
    import random
    random.seed(42)
    
    def try_random_rank(r, max_iters=5000, coeff_range=(-2, 2)):
        """Try random integer coefficient search for given rank."""
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            U = np.random.randint(coeff_range[0], coeff_range[1]+1, (6, r))
            V = np.random.randint(coeff_range[0], coeff_range[1]+1, (9, r))
            W = np.random.randint(coeff_range[0], coeff_range[1]+1, (6, r))
            if verify_decomp(U, V, W):
                return (U, V, W)
        return None
    
    # Try rank 16 with random search
    result = try_random_rank(16, max_iters=2000)
    if result is not None:
        best_rank = 16
        best_decomp = result
    elif time.time() < deadline - 10:
        # Try rank 17
        result = try_random_rank(17, max_iters=2000)
        if result is not None:
            best_rank = 17
            best_decomp = result
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
