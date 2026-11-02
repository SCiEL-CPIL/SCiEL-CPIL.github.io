"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2_terms():
    """Return the 7 Strassen rank-1 terms for 2x2 * 2x2 -> 2x2.
    Each term is (u_vec, v_vec, w_vec) where:
      u_vec has length 4 (indices: a00,a01,a10,a11)
      v_vec has length 4 (indices: b00,b01,b10,b11)
      w_vec has length 4 (indices: c00,c01,c10,c11)
    """
    terms = []
    # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1, c01 += M2, c10 += M3
    # M2 = (a10+a11)b00 -> c00 -= M2, c10 += M2
    # M3 = a00(b01-b11) -> c01 += M3, c11 += M3
    # M4 = a11(b10-b00) -> c10 += M4, c11 -= M4
    # M5 = (a00+a01)b11 -> c00 -= M5, c11 += M5
    # M6 = (a10-a00)(b00+b01) -> c10 += M6
    # M7 = (a01-a11)(b10+b11) -> c01 += M7
    
    # Using the standard Strassen formulation:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    # c00 = p1 - p2 + p3 + p4
    # c01 = p5 + p7
    # c10 = p2 + p6
    # c11 = p1 - p4 + p5 - p7
    
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))    # p1
    terms.append(([0,0,1,1], [1,0,0,0], [-1,0,1,0]))    # p2
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))    # p3
    terms.append(([0,0,0,1], [0,0,1,-1], [0,0,1,-1]))   # p4
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,0,1]))    # p5
    terms.append(([0,0,1,-1], [1,1,0,0], [0,0,1,0]))    # p6
    terms.append(([0,1,0,-1], [0,0,1,1], [0,1,0,-1]))   # p7
    return terms

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # Strategy: Decompose using Kronecker product structure.
    # Write A (2x3) as [A_left(2x2) | A_right(2x1)]
    # Write B (3x8) as [B_top(2x8); B_bot(1x8)]
    # C = A_left * B_top + A_right * B_bot
    # 
    # For A_left(2x2) * B_top(2x8): use Strassen for the 2x2 part on A_left,
    #   and for each Strassen term, the B part is a 2x8 row combination.
    #   We can further decompose: treat B_top as 4 blocks of 2x2.
    #   For each 2x2 block of B_top, use Strassen again? No, that doesn't work directly.
    #
    # Alternative structural approach: 
    # Use the fact that 2x3 * 3x8 can be seen as:
    #   C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # Let's use a different decomposition:
    # Split A into columns: A = [a0 | a1 | a2] where ai is 2x1
    # Split B into rows: B = [b0; b1; b2] where bk is 1x8
    # C = a0*b0 + a1*b1 + a2*b2 (sum of 3 rank terms, each outer product 2x1 * 1x8 = 2x8)
    # That's rank 6 (3 terms x 2 for the 2x1 * 1x8 outer products... wait)
    # Actually each ai*bk is a 2x8 matrix, which has rank at most 1 as a matrix,
    # but as a tensor term it's one rank-1 term in the multiplication tensor.
    # So schoolbook gives rank = 2*3*8 = 48.
    #
    # Better approach: Use Strassen on the 2x2 sub-block.
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00..b07], [b10..b17], [b20..b27]]
    #
    # C[0, j] = a00*b0j + a01*b1j + a02*b2j
    # C[1, j] = a10*b0j + a11*b1j + a12*b2j
    #
    # Group: For j in {0,1}, use Strassen on the 2x2 subproblem (a00,a10) x (b00,b01) etc.
    # But the third column/row (a02, a12) and (b2j) are separate.
    #
    # Let me try: decompose as 4 independent 2x3 * 3x2 subproblems (one per pair of j's).
    # For each subproblem (j=2b, 2b+1):
    #   A_sub = A (2x3), B_sub = B[:, 2b:2b+2] (3x2), C_sub = C[:, 2b:2b+2] (2x2)
    #   Use rank-11 decomposition per subproblem -> total rank 44.
    #
    # To do better, we need to share structure across the 4 subproblems.
    #
    # New idea: Use a Kronecker product decomposition.
    # Consider the tensor T_{i,k,j} = delta_{i,k} for the multiplication tensor.
    # We can write: T = sum_r u_r ⊗ v_r ⊗ w_r
    #
    # Let's try rank 28 = 7 * 4:
    # Use Strassen (rank 7) for the 2x2 * 2x2 part, and replicate for 4 blocks of size 2 in the output.
    # But the input has 3 rows/columns, not 2.
    #
    # Let me try a different factorization:
    # Write the multiplication as:
    # C[0,j] = a00*b0j + a01*b1j + a02*b2j
    # C[1,j] = a10*b0j + a11*b1j + a12*b2j
    #
    # Apply Strassen-like combinations on the (a00,a10) and (b0j) part,
    # and handle (a01,a11) and (b1j) and (a02,a12) and (b2j) separately.
    #
    # Actually, let me just use the rank-44 solution but verify it's correct,
    # and then try to find a lower rank via a different structural decomposition.
    #
    # For now, let me implement a verified rank-44 solution and then attempt optimization.
    
    # First, let's build and verify the rank-44 solution
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        V[p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][col] = 1
        V[k1][col] = 1
        V[p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[4][col] = 1
        V[p + k0][col] = 1
        V[k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][col] = 1
        U[1][col] = 1
        V[p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3][col] = 1
        U[0][col] = -1
        V[k0][col] = 1
        V[k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] = 1
        U[4][col] = -1
        V[p + k0][col] = 1
        V[p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # M8 = A02 * B2k0
        U[2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # M9 = A02 * B2k1
        U[2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M10 = A12 * B2k0
        U[5][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M11 = A12 * B2k1
        U[5][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_r U[i,r]*V[k,r]*W[j*n+i, r] should equal delta_{i,k}... 
    # Actually the tensor is T[i,k,j] = 1 if the multiplication gives C[i,j] = sum_k A[i,k]*B[k,j]
    # The decomposition is: T[i,k,j] = sum_r U[i,r] * V[k,r] * W[j*n+i, r]
    # Wait, W has shape (p*n, rank), indexed by (j, i) -> j*n + i
    # So: sum_r U[i,r] * V[k,r] * W[j*n+i, r] = delta_{i,k}... no.
    # 
    # The matrix multiplication tensor: M[i,k,j] = 1 if we're computing C[i,j] += A[i,k]*B[k,j]
    # So M[i,k,j] = 1 for all valid i,k,j (it's the identity in the "multiplication" sense)
    # Actually M[i,k,j] = delta_{i,i} * delta_{k,k} * delta_{j,j}... no.
    # 
    # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
    # T[i, k, j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    # because C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # The decomposition: T[i,k,j] = sum_r U[i,r] * V[k,r] * W[j*n+i, r]
    # 
    # Let's verify:
    T_reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_reconstructed[i, k, j] += U_np[i, r] * V_np[k, r] * W_np[j*n+i, r]
    
    T_true = np.ones((n, m, p))
    
    if not np.allclose(T_reconstructed, T_true, atol=1e-10):
        # Fallback to schoolbook rank-48
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m+k][col] = 1
                    V[k*p+j][col] = 1
                    W[j*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
