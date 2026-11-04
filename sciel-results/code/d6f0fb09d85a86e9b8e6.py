"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import null_space

_start_time = time.time()
_DEADLINE = 130  # seconds, safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_mat = U[:, r].reshape(n, m)
        v_mat = V[:, r].reshape(m, p)
        w_mat = W[:, r].reshape(p, n)
        M_r = u_mat @ v_mat
        C_decomp += w_mat.T * M_r
    
    if not np.allclose(C_decomp, C_expected, atol=tol):
        return False
    
    for seed in [123, 456, 789]:
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            M_r = u_mat @ v_mat
            C_decomp += w_mat.T * M_r
        if not np.allclose(C_decomp, C_expected, atol=tol):
            return False
    
    return True


def _build_strassen_4x4x4():
    """
    Build the standard rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns U, V, W for n=2, m=2, p=2.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U: for each product, which A-entries are used (with coefficients)
    # A = [[a, b], [c, d]] -> indices: 0=a, 1=b, 2=c, 3=d
    strassen_u = [
        {0: 1, 2: 1},        # M1: a+d
        {2: 1, 3: 1},        # M2: c+d
        {0: 1},              # M3: a
        {3: 1},              # M4: d
        {0: 1, 1: 1},        # M5: a+b
        {2: 1, 0: -1},       # M6: c-a
        {1: 1, 3: -1},       # M7: b-d
    ]
    
    # V: for each product, which B-entries are used
    # B = [[e, f], [g, h]] -> indices: 0=e, 1=f, 2=g, 3=h
    strassen_v = [
        {0: 1, 3: 1},        # M1: e+h
        {0: 1},              # M2: e
        {1: 1, 3: -1},       # M3: f-h
        {2: 1, 0: -1},       # M4: g-e
        {3: 1},              # M5: h
        {0: 1, 1: 1},        # M6: e+f
        {2: 1, 3: 1},        # M7: g+h
    ]
    
    # W: for each product, which C-entries are affected
    # C = [[C00, C01], [C10, C11]] -> indices: 0=C00, 1=C01, 2=C10, 3=C11
    strassen_w = [
        {0: 1, 3: 1},        # M1
        {2: 1, 3: -1},       # M2
        {1: 1, 3: 1},        # M3
        {0: 1, 2: 1},        # M4
        {0: -1, 1: 1},       # M5
        {3: 1},              # M6
        {0: 1},              # M7
    ]
    
    for r in range(rank):
        for pos, coeff in strassen_u[r].items():
            U[pos, r] = coeff
        for pos, coeff in strassen_v[r].items():
            V[pos, r] = coeff
        for pos, coeff in strassen_w[r].items():
            W[pos, r] = coeff
    
    return U, V, W


def _build_4x4x8_via_border_rank():
    """
    Build a decomposition for 4x4 * 4x8 using the border rank lifting technique.
    
    The key insight: The 4x4x8 matrix multiplication tensor can be viewed as a
    degenerate (border) version of two 4x4x4 tensors. We use the known rank-7
    Strassen decomposition for 2x2*2x2 and extend it to 4x4*4x4 (rank 49 via
    Kronecker product or block Strassen), then adapt for the 4x4*4x8 case.
    
    Strategy: Split the 4x4*4x8 problem into blocks:
    - A is 4x4, B is 4x8
    - Split B into B1 (4x4) and B2 (4x4)
    - C1 = A * B1 (4x4), C2 = A * B2 (4x4)
    - C = [C1 | C2] (4x8)
    
    Use rank-49 Strassen for each 4x4*4x4: total rank 98.
    But we can potentially do better by sharing structure.
    
    For now, use the straightforward block approach with rank-49 per block = 98.
    """
    n, m, p = 4, 4, 8
    
    # Build 4x4*4x4 Strassen decomposition (rank 49)
    # Use 2x2 blocks: 4x4 = 2x2 blocks of 2x2
    # 4 sub-problems (i1,k1 in {0,1}^2), each 2x2*2x2 with rank 7
    # Total: 4 * 7 = 28? No, that's for the output.
    # Actually for 4x4*4x4: split into 2x2 blocks
    # i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # C[i0,k0] for each (i1,k1) block: sum over j1,j0 of A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    # This is a (2, 4, 2) matmul: M(2x4) * N(4x2) = C(2x2)
    # Split M = [M1 | M2], N = [N1; N2]
    # C = M1*N1 + M2*N2
    # Each 2x2*2x2 uses rank 7 (Strassen)
    # Total per (i1,k1): 14
    # 4 (i1,k1) blocks * 14 = 56
    
    # Wait, let me reconsider. For 4x4*4x4:
    # Using block Strassen with 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    # 
    # Standard Strassen on 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2*2x2 product, which itself can be done with rank 7 (Strassen on 2x2)
    # So each Mi contributes 7 terms
    # Total: 7 * 7 = 49 terms for 4x4*4x4
    
    # Now for 4x4*4x8:
    # Split B = [B_left | B_right] where each is 4x4
    # C_left = A * B_left (4x4), C_right = A * B_right (4x4)
    # Use rank-49 Strassen for each: total rank 98
    
    # Let's build the rank-49 4x4*4x4 decomposition first
    def _build_4x4x4_strassen():
        n, m, p = 4, 4, 4
        rank = 49
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        # 2x2 Strassen for the outer block structure
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # Each sub-block is 2x2
        
        # Outer Strassen products (7 of them)
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each outer product, we need to compute a 2x2*2x2 product
        # using inner Strassen (7 terms)
        
        # Define the 2x2 Strassen patterns
        u_2x2 = [
            {0: 1, 2: 1},        # a+d
            {2: 1, 3: 1},        # c+d
            {0: 1},              # a
            {3: 1},              # d
            {0: 1, 1: 1},        # a+b
            {2: 1, 0: -1},       # c-a
            {1: 1, 3: -1},       # b-d
        ]
        v_2x2 = [
            {0: 1, 3: 1},        # e+h
            {0: 1},              # e
            {1: 1, 3: -1},       # f-h
            {2: 1, 0: -1},       # g-e
            {3: 1},              # h
            {0: 1, 1: 1},        # e+f
            {2: 1, 3: 1},        # g+h
        ]
        w_2x2 = [
            {0: 1, 3: 1},        # C00 + C11
            {2: 1, 3: -1},       # C10 - C11
            {1: 1, 3: 1},        # C01 + C11
            {0: 1, 2: 1},        # C00 + C10
            {0: -1, 1: 1},       # -C00 + C01
            {3: 1},              # C11
            {0: 1},              # C00
        ]
        
        # Outer Strassen: for each of the 7 outer products,
        # we have a linear combination of A-blocks times a linear combination of B-blocks.
        # Each such product is a 2x2*2x2 matmul, decomposed with inner Strassen.
        
        # Outer product definitions:
        # For each outer product r_outer, we have:
        #   U_outer: dict mapping block_index -> coefficient (for A blocks)
        #   V_outer: dict mapping block_index -> coefficient (for B blocks)
        #   W_outer: dict mapping output_block -> coefficient
        
        # A blocks: 0=A11, 1=A12, 2=A21, 3=A22 (each 2x2)
        # B blocks: 0=B11, 1=B12, 2=B21, 3=B22
        # C blocks: 0=C11, 1=C12, 2=C21, 3=C22
        
        outer_products = [
            # (U_blocks, V_blocks, W_blocks)
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1: (A11+A22)(B11+B22) -> C11 + C22
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2: (A21+A22)B11 -> C21 - C22
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3: A11(B12-B22) -> C12 + C22
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4: A22(B21-B11) -> C11 + C21
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5: (A11+A12)B22 -> -C11 + C12
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6: (A21-A11)(B11+B12) -> C22
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),            # M7: (A12-A22)(B21+B22) -> C11
        ]
        
        col = 0
        for r_outer in range(7):
            u_blocks, v_blocks, w_blocks = outer_products[r_outer]
            
            # For this outer product, we compute:
            #   (sum_{b1} u_blocks[b1] * A_block[b1]) * (sum_{b2} v_blocks[b2] * B_block[b2])
            #   = sum_{b1,b2} u_blocks[b1] * v_blocks[b2] * A_block[b1] * B_block[b2]
            #
            # Each A_block[b1] * B_block[b2] is a 2x2*2x2 product.
            # We decompose it with inner Strassen (7 terms).
            #
            # But we can combine: the total contribution to the outer product is:
            #   sum_{b1,b2} u_blocks[b1] * v_blocks[b2] * (A_block[b1] * B_block[b2])
            #
            # For the inner Strassen, we need to figure out the effective U and V vectors.
            #
            # Actually, a cleaner approach: for each pair (b1, b2) where u_blocks[b1] != 0 
            # and v_blocks[b2] != 0, we add the inner Strassen decomposition of 
            # A_block[b1] * B_block[b2] scaled by u_blocks[b1] * v_blocks[b2].
            #
            # But this would give up to 4*4*7 = 112 terms per outer product, which is too many.
            #
            # Better approach: The product (sum u_b1 A_b1) * (sum v_b2 B_b2) is itself
            # a 2x2 * 2x2 product where:
            #   A_eff = sum_{b1} u_blocks[b1] * A_block[b1]  (2x2 matrix)
            #   B_eff = sum_{b2} v_blocks[b2] * B_block[b2]  (2x2 matrix)
            #
            # So we just need the inner Strassen decomposition of A_eff * B_eff,
            # where A_eff and B_eff are linear combinations of the original blocks.
            
            # For the inner Strassen, we need to express A_eff and B_eff in terms of the original A and B entries.
            # A_eff[i,j] = sum_{b1} u_blocks[b1] * A_block[b1][i,j]
            # B_eff[i,j] = sum_{b2} v_blocks[b2] * B_block[b2][i,j]
            
            # The inner Strassen U vectors for A_eff:
            # For each inner product r_inner, U_inner[r_inner] is a 2x2 matrix with entries from A_eff.
            # A_eff entry at position (ii, jj) where ii in {0,1}, jj in {0,1}:
            #   A_eff[ii, jj] = sum_{b1} u_blocks[b1] * A[2*row(b1)+ii, 2*col(b1)+jj]
            #   where row(b1) = b1 // 2, col(b1) = b1 % 2
            #   A_block[b1] is the 2x2 block at rows 2*(b1//2) to 2*(b1//2)+1, cols 2*(b1%2) to 2*(b1%2)+1
            
            # Let's compute the effective U and V for the inner Strassen
            for r_inner in range(7):
                # U for inner product r_inner:
                # u_2x2[r_inner] maps position (0,1,2,3) to coefficients
                # Position 0 = A_eff[0,0], 1 = A_eff[0,1], 2 = A_eff[1,0], 3 = A_eff[1,1]
                
                # For each position in A_eff, expand in terms of original A entries
                for pos, coeff_inner in u_2x2[r_inner].items():
                    ii = pos // 2
                    jj = pos % 2
                    # A_eff[ii, jj] = sum_{b1} u_blocks[b1] * A[2*(b1//2)+ii, 2*(b1%2)+jj]
                    for b1, coeff_outer in u_blocks.items():
                        row_a = 2 * (b1 // 2) + ii
                        col_a = 2 * (b1 % 2) + jj
                        U[row_a * m + col_a, col] += coeff_inner * coeff_outer
                
                # V for inner product r_inner:
                for pos, coeff_inner in v_2x2[r_inner].items():
                    ii = pos // 2
                    jj = pos % 2
                    # B_eff[ii, jj] = sum_{b2} v_blocks[b2] * B[2*(b2//2)+ii, 2*(b2%2)+jj]
                    for b2, coeff_outer in v_blocks.items():
                        row_b = 2 * (b2 // 2) + ii
                        col_b = 2 * (b2 % 2) + jj
                        V[row_b * p + col_b, col] += coeff_inner * coeff_outer
                
                # W for inner product r_inner:
                # w_2x2[r_inner] maps position (0,1,2,3) to coefficients
                # Position 0 = C_eff[0,0], 1 = C_eff[0,1], 2 = C_eff[1,0], 3 = C_eff[1,1]
                # C_eff is the 2x2 result of A_eff * B_eff
                # The output block for this outer product is determined by w_blocks
                
                for pos, coeff_inner in w_2x2[r_inner].items():
                    ii = pos // 2
                    jj = pos % 2
                    # C_eff[ii, jj] contributes to the output
                    # The output C has blocks: C_block[c0] where c0 in {0,1,2,3}
                    # C_eff corresponds to which output block?
                    # 
                    # The outer product M_r contributes to output blocks specified by w_blocks.
                    # C_eff[ii, jj] is the (ii,jj) entry of the 2x2 result.
                    # This maps to C[2*i_out+ii, 2*j_out+jj] where (i_out, j_out) is the output block.
                    #
                    # For outer product M1: w_blocks = {0: 1, 3: 1}
                    #   C_eff contributes to C_block[0] (C11) and C_block[3] (C22)
                    #   C_eff[0,0] -> C[0,0], C_eff[0,1] -> C[0,1], C_eff[1,0] -> C[1,0], C_eff[1,1] -> C[1,1]
                    #   C_block[3]: C_eff[0,0] -> C[2,2], C_eff[0,1] -> C[2,3], C_eff[1,0] -> C[3,2], C_eff[1,1] -> C[3,3]
                    
                    # The output block index for C_eff[ii, jj]:
                    # We need to map (ii, jj) to the global C index.
                    # The outer product's w_blocks tells us which output blocks are affected.
                    # For each output block c0 in w_blocks, the contribution is:
                    #   C[2*(c0//2)+ii, 2*(c0%2)+jj] += coeff_outer * C_eff[ii, jj]
                    
                    for c0, coeff_outer_w in w_blocks.items():
                        i_out = 2 * (c0 // 2) + ii
                        k_out = 2 * (c0 % 2) + jj
                        w_idx = k_out * n + i_out
                        W[w_idx, col] += coeff_inner * coeff_outer_w
                
                col += 1
        
        return U, V, W
    
    # Build 4x4*4x4 decomposition
    U_444, V_444, W_444 = _build_4x4x4_strassen()
    
    # Now extend to 4x4*4x8:
    # B = [B_left | B_right], each 4x4
    # C_left = A * B_left, C_right = A * B_right
    # Use the 4x4*4x4 decomposition for each
    
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Left half: B columns 0-3
    # V_left: B[j, k] for k in 0..3 -> V[j*4 + k] in the 4x4*4x4 decomp
    # In the full 4x4*4x8, V index is j*p + k = j*8 + k
    U[:, 0:49] = U_444
    for j in range(4):
        for k in range(4):
            V[j * p + k, 0:49] = V_444[j * 4 + k, 0:49]
    # W for left half: C[i, k] for k in 0..3
    # W_444 index: k*n + i for k in 0..3
    # In full: W index is k*n + i (same, since p doesn't affect W indexing)
    W[:, 0:49] = W_444
    
    # Right half: B columns 4-7
    # V_right: B[j, k] for k in 4..7 -> in 4x4*4x4 decomp, this is B[j, k-4]
    # In the full 4x4*4x8, V index is j*8 + k
    U[:, 49:98] = U_444
    for j in range(4):
        for k in range(4):
            V[j * p + (k + 4), 49:98] = V_444[j * 4 + k, 0:49]
    # W for right half: C[i, k] for k in 4..7
    # W_444 index: k*n + i for k in 0..3
    # In full: W index is (k+4)*n + i
    for k in range(4):
        for i in range(4):
            W[(k + 4) * n + i, 49:98] = W_444[k * n + i, 0:49]
    
    return U, V, W


def _find_minimal_rank(U, V, W, n, m, p, max_iter=50):
    """
    Try to find a lower-rank decomposition by identifying linear dependencies
    among the rank terms and removing redundant ones.
    
    This is a heuristic: we check if any column of [U;V;W] is in the span of
    others, and if so, we can potentially remove it (with appropriate adjustments).
    """
    rank = U.shape[1]
    
    # Check for linearly dependent terms
    # A term r is redundant if there exist coefficients c_s (s != r) such that
    # the tensor contribution of term r equals the sum of contributions of terms s.
    # This is complex, so we use a simpler heuristic: check if the combined vector
    # (U[:,r], V[:,r], W[:,r]) is in the span of other such vectors.
    
    # Build the combined matrix
    combined = np.vstack([U, V, W])  # shape (n*m + m*p + p*n, rank)
    
    # Find the rank of the combined matrix
    # If rank(combined) < rank, some terms are redundant
    # But removing a term requires adjusting the others, which is non-trivial.
    
    # For now, just return the current decomposition
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build the rank-98 border rank decomposition
    try:
        U, V, W = _build_4x4x8_via_border_rank()
        
        # Verify
        if _verify(U, V, W, n, m, p):
            # Try to reduce rank (heuristic)
            U, V, W = _find_minimal_rank(U, V, W, n, m, p)
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                # Rebuild if reduction failed
                U, V, W = _build_4x4x8_via_border_rank()
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Verification failed, fall through to schoolbook
            pass
    except Exception:
        pass
    
    # Fallback: schoolbook rank 128
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
