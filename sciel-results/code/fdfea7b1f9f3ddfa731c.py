"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses recursive Strassen algorithm.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s wall-clock limit with safety margin

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Recursive Strassen decomposition builder
    def build_decomp(n, m, p):
        """
        Returns U, V, W for n x m * m x p multiplication.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        # Base case: 1x1 * 1x1
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        
        # If any dimension is odd, fallback to schoolbook
        if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            return U, V, W

        # Recursive case: n, m, p are even
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        # Get sub-decomposition for n2 x m2 * m2 x p2
        U_sub, V_sub, W_sub = build_decomp(n2, m2, p2)
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub

        # Initialize full matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        # Define the 7 Strassen terms
        # Each term is defined by linear combinations of blocks
        # A blocks: A11, A12, A21, A22
        # B blocks: B11, B12, B21, B22
        # C blocks: C11, C12, C21, C22
        
        # Term 1: (A11+A22)(B11+B22) -> C11 + C22 - A11*B22 - A22*B11
        # Actually, standard Strassen:
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
        # C22 = M1 - M3 + M2 - M6

        # We need to map the local indices of the sub-problem to the global indices.
        # Local indices in sub-problem:
        # U: i in 0..n2-1, j in 0..m2-1 -> idx = i*m2 + j
        # V: j in 0..m2-1, k in 0..p2-1 -> idx = j*p2 + k
        # W: k in 0..p2-1, i in 0..n2-1 -> idx = k*n2 + i

        # For each term t (0..6), we define:
        # u_maps: list of (local_u_idx, coeff) for A part
        # v_maps: list of (local_v_idx, coeff) for B part
        # w_maps: list of (local_w_idx, coeff) for C part
        
        # Note: The sub-decomposition computes X * Y = Z where X is n2 x m2, Y is m2 x p2, Z is n2 x p2.
        # We are using it to compute linear combinations.
        
        # Let's iterate over the local indices of the sub-problem to build the global matrices.
        # For each local column c in 0..rank_sub-1:
        #   For each local row in U_sub:
        #     val = U_sub[u_idx][c]
        #     Add val * coeff to global U at mapped indices
        
        # This approach is O(rank_sub * n2*m2 * 7) which is fine.
        
        # Precompute mappings for each term
        # Term 0: M1 = (A11+A22)(B11+B22)
        # Term 1: M2 = (A21+A22)B11
        # Term 2: M3 = A11(B12-B22)
        # Term 3: M4 = A22(B21-B11)
        # Term 4: M5 = (A11+A12)B22
        # Term 5: M6 = (A21-A11)(B11+B12)
        # Term 6: M7 = (A12-A22)(B21+B22)
        
        # We will construct U, V, W by iterating over the sub-decomposition terms.
        # Let's use a more direct construction.
        
        # For each term t, we have a "virtual" multiplication X_t * Y_t = Z_t.
        # X_t is a linear combination of A blocks.
        # Y_t is a linear combination of B blocks.
        # Z_t is a linear combination of C blocks.
        
        # The sub-decomposition gives us U_sub, V_sub, W_sub such that sum_c U_sub[:,c] V_sub[:,c]^T = W_sub[:,c] ... wait, tensor definition.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor T[i,j,k] = 1 if ... no, T is the multiplication tensor.
        # T[i,j,k] = 1.
        # Decomposition: T = sum_c U[:,c] V[:,c] W[:,c]
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k]) W[i,k,c]? No.
        
        # Standard tensor decomposition:
        # C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k]) ... this is not right.
        
        # Correct definition:
        # The multiplication tensor T has entries T[i,j,k] = 1.
        # We want to write T as sum_{c=1}^R u_c v_c w_c
        # where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.
        # Specifically, u_c[i*m+j], v_c[j*p+k], w_c[k*n+i].
        # The condition is:
        # sum_c u_c[i*m+j] v_c[j*p+k] w_c[k*n+i] = 1 for all i,j,k? No.
        
        # The matrix multiplication is a bilinear map (A, B) -> C.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # This can be written as:
        # C = sum_c (A * U_c) * (V_c * B) * W_c? No.
        
        # Let's use the standard form:
        # C[i,k] = sum_c (sum_j A[i,j] U_c[i,j]) (sum_j B[j,k] V_c[j,k]) W_c[i,k]
        # This is not standard.
        
        # Standard tensor rank definition for matrix multiplication:
        # We are looking for a decomposition of the tensor T where T[i,j,k] = 1.
        # The decomposition is:
        # T[i,j,k] = sum_c u_c[i,j] v_c[j,k] w_c[i,k]
        # where u_c is indexed by (i,j), v_c by (j,k), w_c by (i,k).
        # Note: The indices in the prompt are:
        # U: (n*m, rank) -> U[i*m+j, c]
        # V: (m*p, rank) -> V[j*p+k, c]
        # W: (p*n, rank) -> W[k*n+i, c]
        
        # So the condition is:
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
        
        # Wait, is it 1?
        # C[i,k] = sum_j A[i,j] B[j,k].
        # If we expand A[i,j] = sum_i A[i,j] * 1 and B[j,k] = sum_k B[j,k] * 1...
        # The tensor T represents the structure of the multiplication.
        # T[i,j,k] = 1.
        # So yes, sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
        
        # Let's verify for schoolbook:
        # Rank = n*m*p.
        # For each (i,j,k), we have a term c.
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
        # Sum is 1. Correct.
        
        # Now for Strassen:
        # We have 7 sub-multiplications.
        # Each sub-multiplication corresponds to a tensor T_sub of size n2*m2*p2.
        # T_sub[i,j,k] = 1.
        # Decomposition of T_sub: sum_c U_sub[i*m2+j, c] V_sub[j*p2+k, c] W_sub[k*n2+i, c] = 1.
        
        # The global tensor T is a linear combination of these sub-tensors mapped to the global indices.
        
        # Let's define the mapping for each term.
        
        # Term 0: M1 = (A11+A22)(B11+B22)
        # A part: A11 (+1), A22 (+1)
        # B part: B11 (+1), B22 (+1)
        # C part: This term contributes to C11, C22, etc.
        # Specifically, M1 is used in C11 and C22.
        # C11 = M1 + ...
        # C22 = M1 + ...
        
        # The "C part" of the sub-tensor M1 is actually the result of the multiplication of the combined A and B.
        # But in the tensor decomposition, the W part corresponds to the output C.
        # So for Term 0, the "virtual" C is M1.
        # How does M1 map to global C?
        # M1 is an n2 x p2 matrix.
        # It contributes to C11 and C22?
        # No, M1 is just an intermediate value.
        # The final C is a linear combination of M1...M7.
        
        # So, we need to express the global tensor T as a sum of 7 sub-tensors, each scaled and mapped appropriately.
        
        # Let T_t be the tensor for term t.
        # T_t[i,j,k] (local) = 1.
        # Global contribution:
        # For a local index (i,j,k) in term t:
        # The A part is a linear combination of global A blocks.
        # The B part is a linear combination of global B blocks.
        # The C part is a linear combination of global C blocks.
        
        # Actually, it's easier to think of it as:
        # We are constructing U, V, W such that sum_c U[i,j,c] V[j,k,c] W[i,k,c] = 1.
        
        # Let's build U, V, W by iterating over the 7 terms and the sub-decomposition.
        
        # For each term t (0..6):
        #   For each sub-column c_sub (0..rank_sub-1):
        #     Global column c = t * rank_sub + c_sub
        #     For each local i, j, k:
        #       val_u = U_sub[i*m2+j, c_sub]
        #       val_v = V_sub[j*p2+k, c_sub]
        #       val_w = W_sub[k*n2+i, c_sub]
        
        #       We need to add to global U, V, W.
        #       But U, V, W are not just single values, they are sums of contributions from different blocks.
        
        #       For Term t, the A part is sum_{block} coeff_a[block] * A_block.
        #       So U_global[i,j,c] = sum_{block} coeff_a[block] * val_u (if (i,j) is in that block).
        #       Wait, val_u is already the coefficient for the local A[i,j].
        #       So if local (i,j) maps to global (I,J) in block B_a, then:
        #       U_global[I,J,c] += coeff_a * val_u.
        
        #       Similarly for V and W.
        
        # Let's define the mappings for each term.
        
        # Term 0: M1 = (A11+A22)(B11+B22)
        # A: A11 (+1), A22 (+1)
        # B: B11 (+1), B22 (+1)
        # C: M1 contributes to C11 (+1) and C22 (+1)??
        # No, M1 is a matrix.
        # C11 = M1 + M4 - M5 + M7
        # C22 = M1 - M3 + M2 - M6
        # So M1 contributes +1 to C11 and +1 to C22.
        # But C11 is n2 x p2, C22 is n2 x p2.
        # So for a local (i,k) in M1, it maps to global (i,k) in C11 and (i+n2, k+p2) in C22.
        
        # So for Term 0:
        # A maps:
        #   Local (i,j) in A11 -> Global (i, j) with coeff +1
        #   Local (i,j) in A22 -> Global (i+n2, j+m2) with coeff +1
        # B maps:
        #   Local (j,k) in B11 -> Global (j, k) with coeff +1
        #   Local (j,k) in B22 -> Global (j+m2, k+p2) with coeff +1
        # C maps:
        #   Local (i,k) in M1 -> Global (i, k) in C11 with coeff +1
        #   Local (i,k) in M1 -> Global (i+n2, k+p2) in C22 with coeff +1
        
        # Let's implement this.
        
        terms = [
            {
                'A': [(0, 0, 1), (1, 1, 1)],  # (block_row, block_col, coeff)
                'B': [(0, 0, 1), (1, 1, 1)],
                'C': [(0, 0, 1), (1, 1, 1)]
            },
            {
                'A': [(1, 0, 1), (1, 1, 1)],
                'B': [(0, 0, 1)],
                'C': [(0, 1, 1), (1, 1, 1)] # M2 contributes to C21 (+1) and C22 (-1)?
                # C21 = M2 + M4
                # C22 = M1 - M3 + M2 - M6
                # So M2 contributes +1 to C21 and +1 to C22.
                # Wait, C22 formula: M1 - M3 + M2 - M6. So M2 is +1.
            },
            {
                'A': [(0, 0, 1)],
                'B': [(0, 1, 1), (1, 1, -1)],
                'C': [(0, 1, 1), (1, 1, -1)] # M3 contributes to C12 (+1) and C22 (-1)
                # C12 = M3 + M5
                # C22 = M1 - M3 + M2 - M6
            },
            {
                'A': [(1, 1, 1)],
                'B': [(1, 0, 1), (0, 0, -1)],
                'C': [(0, 0, 1), (1, 0, 1)] # M4 contributes to C11 (+1) and C21 (+1)
                # C11 = M1 + M4 - M5 + M7
                # C21 = M2 + M4
            },
            {
                'A': [(0, 0, 1), (0, 1, 1)],
                'B': [(1, 1, 1)],
                'C': [(0, 0, -1), (0, 1, 1)] # M5 contributes to C11 (-1) and C12 (+1)
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
            },
            {
                'A': [(1, 0, 1), (0, 0, -1)],
                'B': [(0, 0, 1), (0, 1, 1)],
                'C': [(1, 1, -1)] # M6 contributes to C22 (-1)
                # C22 = M1 - M3 + M2 - M6
            },
            {
                'A': [(0, 1, 1), (1, 1, -1)],
                'B': [(1, 0, 1), (1, 1, 1)],
                'C': [(0, 0, 1)] # M7 contributes to C11 (+1)
                # C11 = M1 + M4 - M5 + M7
            }
        ]
        
        # Note: The 'C' mappings above are for the contribution of M_t to the final C.
        # But in the tensor decomposition, the W part corresponds to the output of the sub-multiplication.
        # So for Term t, the "virtual" output is M_t.
        # The global W should map the local (i,k) of M_t to the global C positions where M_t contributes.
        
        # Let's refine the loop.
        
        for t in range(7):
            base_col = t * rank_sub
            term = terms[t]
            
            # Iterate over local indices
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        # Get sub-values
                        u_val = U_sub[i*m2 + j][c_sub] # Wait, c_sub is in the outer loop?
                        # We need to iterate over c_sub as well.
                        
        # Let's restructure the loop.
        
        for t in range(7):
            base_col = t * rank_sub
            term = terms[t]
            
            for c_sub in range(rank_sub):
                # We need to update U, V, W for this column.
                # U, V, W are lists of lists.
                
                # It is more efficient to iterate over the non-zero entries of U_sub, V_sub, W_sub.
                # But they are dense in the sub-problem?
                # For schoolbook, they are sparse (one 1 per column).
                # For Strassen, they are sums of schoolbook, so they are dense.
                
                # Let's just iterate over all i, j, k.
                for i in range(n2):
                    for j in range(m2):
                        u_local = i*m2 + j
                        u_val = U_sub[u_local][c_sub]
                        if u_val != 0:
                            for br, bc, coeff in term['A']:
                                # Map local (i,j) to global
                                gi = i + br * n2
                                gj = j + bc * m2
                                U[gi*m + gj][base_col + c_sub] += u_val * coeff
                        
                        for k in range(p2):
                            v_local = j*p2 + k
                            v_val = V_sub[v_local][c_sub]
                            if v_val != 0:
                                for br, bc, coeff in term['B']:
                                    # Map local (j,k) to global
                                    gj = j + br * m2
                                    gk = k + bc * p2
                                    V[gj*p + gk][base_col + c_sub] += v_val * coeff
                            
                            w_local = k*n2 + i
                            w_val = W_sub[w_local][c_sub]
                            if w_val != 0:
                                for br, bc, coeff in term['C']:
                                    # Map local (i,k) to global
                                    gi = i + br * n2
                                    gk = k + bc * p2
                                    W[gk*n + gi][base_col + c_sub] += w_val * coeff

        return U, V, W

    U, V, W = build_decomp(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
