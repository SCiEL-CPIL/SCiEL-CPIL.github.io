"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            # W index: W[k*n+i, c] corresponds to C[i,k]
            C_c = np.zeros((n, p))
            for idx in range(p * n):
                k, i = idx // n, idx % n
                C_c[i, k] = wc[idx]
            C_computed += C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _is_nice_half_integer(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    return abs(doubled - rounded) < tol

def _build_strassen_444():
    """
    Build the Strassen decomposition for 4x4x4 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (16, 49), (16, 49), (16, 49).
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2 terms
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # 4x4 is block diagonal of 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # Indices:
    # A block (bi, bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # A element index: (2*bi + x) * 4 + (2*bj + y)
    
    terms = [
        # (coeff_A, coeff_B, coeff_C)
        # M1
        [(0, 0, 1), (1, 1, 1), (0, 0, 1)],
        [(1, 0, 1), (1, 1, 1), (0, 0, 1)],
        [(0, 1, 1), (0, 0, 1), (0, 1, 1)],
        [(1, 1, 1), (0, 0, 1), (0, 1, 1)],
        [(0, 0, 1), (1, 0, 1), (1, 0, 1)],
        [(1, 0, 1), (1, 0, 1), (1, 0, 1)],
        [(0, 1, 1), (0, 1, 1), (1, 1, 1)],
        [(1, 1, 1), (0, 1, 1), (1, 1, 1)],
        # M2
        [(1, 0, 1), (0, 0, 1), (1, 0, 1)],
        [(1, 1, 1), (0, 0, 1), (1, 0, 1)],
        [(0, 0, 1), (0, 0, 1), (1, 1, 1)],
        [(1, 1, 1), (0, 0, 1), (1, 1, 1)], # Wait, M2 is (A21+A22)B11
        # Let's stick to the standard 7 terms for 2x2 and expand each to 4x4 by block multiplication.
    ]
    
    # It is safer to generate the 49 terms by iterating over the 7 Strassen terms
    # and the 4 block elements for each.
    
    strassen_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # M1
        ((0, 0, 1), (1, 1, 1), (0, 0, 1)), # (A00+A11)(B00+B11) -> C00
        ((0, 0, 1), (1, 1, 1), (0, 1, 1)), # -> C01? No.
        # Let's use the explicit formulas for C.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
    ]
    
    # This manual expansion is error-prone. Let's use a recursive construction.
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For 4x4, each "block" is 2x2.
    # So M1 involves summing over all elements in A00, A11 and B00, B11.
    # This results in 4 scalar multiplications for M1 (one for each element pair? No).
    # Actually, (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
    # Their product is computed with 4 multiplications (schoolbook) if we don't recurse.
    # But Strassen for 4x4 usually means applying Strassen to the 2x2 blocks,
    # and then using schoolbook for the 2x2 blocks?
    # No, "Strassen 4x4" usually means the rank is 49, which is 7 * 7.
    # This implies we apply Strassen to the 2x2 blocks, AND use Strassen (or rank 7) for the 2x2 block multiplications?
    # No, rank 7 for 2x2 is Strassen.
    # So we have 7 terms. Each term is a product of two 2x2 matrices.
    # To compute the product of two 2x2 matrices with rank 7, we need 7 scalar multiplications.
    # So 7 * 7 = 49 scalar multiplications total.
    
    # We need to generate the 49 factors.
    # Let's define the 2x2 Strassen factors first.
    # U2, V2, W2 for 2x2x2.
    
    def get_strassen_222():
        n, m, p = 2, 2, 2
        rank = 7
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # M1 = (A00+A11)(B00+B11)
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices: 00=0, 01=1, 10=2, 11=3
        
        # Term 1: M1
        # A: 00, 11 -> 1, 1
        # B: 00, 11 -> 1, 1
        # C: 00, 11 -> 1, 1 (Wait, M1 contributes to C00, C11 with coeff 1, and C01, C10 with 0? No.)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's list the 7 terms and their contributions to A, B, C.
        # Each term is a rank-1 tensor.
        # Term k has vectors u_k, v_k, w_k.
        
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
        # M4: u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,0]
        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]
        # M6: u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,0,1]
        # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
        
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 1]
        
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[:, 1] = [0, 0, 1, 0]
        
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 1, 0, -1]
        W[:, 2] = [0, 1, 0, 0]
        
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [0, 0, 1, -1]
        W[:, 3] = [0, 0, 1, 0]
        
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[:, 4] = [0, 1, 0, 0]
        
        U[:, 5] = [0, 0, 1, -1]
        V[:, 5] = [1, 1, 0, 0]
        W[:, 5] = [0, 0, 0, 1]
        
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 0, 1, 1]
        W[:, 6] = [1, 0, 0, 0]
        
        return U, V, W

    U2, V2, W2 = get_strassen_222()
    
    # Now expand to 4x4.
    # Each 2x2 block element corresponds to a 2x2 sub-matrix in 4x4.
    # So for each of the 7 Strassen terms in the block space,
    # we have 7 scalar terms (from the 2x2 Strassen inside).
    # Total 49 terms.
    
    for term_idx in range(7):
        u_block = U2[:, term_idx]
        v_block = V2[:, term_idx]
        w_block = W2[:, term_idx]
        
        for inner_term in range(7):
            col = term_idx * 7 + inner_term
            u_inner = U2[:, inner_term]
            v_inner = V2[:, inner_term]
            w_inner = W2[:, inner_term]
            
            # u_block[i] corresponds to block i.
            # u_inner[x] corresponds to element x within block i.
            # Global index in U: (2*block_i + x) * 4 + (2*block_j + y) ??
            # No, U is for A. A is 4x4.
            # Block (bi, bj) contains elements (2*bi+x, 2*bj+y).
            # U index is (2*bi+x) * 4 + (2*bj+y).
            
            # For a fixed term_idx and inner_term:
            # The coefficient for A[2*bi+x, 2*bj+y] is u_block[bi*2+bj] * u_inner[inner_idx]?
            # No.
            # The Strassen term is M_k = (sum_bi,bj u_k[bi,bj] A_bi,bj) * (sum_bi,bj v_k[bi,bj] B_bi,bj).
            # This is a product of two 2x2 matrices.
            # Let A' = sum u_k A_bi,bj and B' = sum v_k B_bi,bj.
            # We compute A' B' using Strassen 2x2 again?
            # Yes, if we want rank 49.
            # So we apply the 2x2 Strassen decomposition to A' and B'.
            # The factors for A' are linear combinations of A_bi,bj.
            # The factors for B' are linear combinations of B_bi,bj.
            
            # So, U_col = sum_{bi,bj} u_k[bi,bj] * (U2_inner for A')
            # Wait, U2_inner is for the 2x2 matrix A'.
            # A' is a 2x2 matrix. Its elements are linear combinations of A_bi,bj.
            # Let A'_xy = sum_{bi,bj} u_k[bi,bj] A_{bi,bj, x,y}? No.
            # A_bi,bj is a 2x2 matrix.
            # A' is a 2x2 matrix.
            # A' = sum_{bi,bj} u_k[bi,bj] A_bi,bj.
            # So A'_xy = sum_{bi,bj} u_k[bi,bj] A_{bi,bj, xy}.
            
            # We decompose A' B' = C' using Strassen 2x2.
            # C' = sum_{inner} (u_inner * A') (v_inner * B') w_inner.
            # This means we multiply the scalar (u_inner[xy] * A'_xy) ... no.
            # The rank-1 tensor is u_vec A' v_vec B' w_vec C'.
            # u_vec is 2x2. A' is 2x2.
            # So the term is (u_vec * A') * (v_vec * B') * w_vec.
            # This is equivalent to:
            # sum_{x,y} u_vec[x,y] A'_xy * sum_{x,y} v_vec[x,y] B'_xy * w_vec...
            
            # Let's just construct the global U, V, W vectors.
            # For a fixed outer term k and inner term l:
            # U_global[i,j] = sum_{bi,bj} u_k[bi,bj] * U2[l, (2*bi+x) + 2*(2*bj+y)??]
            # No, U2[l, idx] is the coefficient for A'_idx.
            # A'_idx = sum_{bi,bj} u_k[bi,bj] A_{bi,bj, idx}.
            # So U_global[global_idx] = sum_{bi,bj} u_k[bi,bj] * U2[l, idx].
            # Where global_idx corresponds to A_{bi,bj, idx}.
            
            for bi in range(2):
                for bj in range(2):
                    coeff_u = u_block[bi * 2 + bj]
                    if coeff_u == 0: continue
                    for x in range(2):
                        for y in range(2):
                            global_idx = (2 * bi + x) * 4 + (2 * bj + y)
                            U[global_idx, col] += coeff_u * U2[l, x * 2 + y]
                            
            for bi in range(2):
                for bj in range(2):
                    coeff_v = v_block[bi * 2 + bj]
                    if coeff_v == 0: continue
                    for x in range(2):
                        for y in range(2):
                            global_idx = (2 * bi + x) * 4 + (2 * bj + y)
                            V[global_idx, col] += coeff_v * V2[l, x * 2 + y]
                            
            for bi in range(2):
                for bj in range(2):
                    coeff_w = w_block[bi * 2 + bj]
                    if coeff_w == 0: continue
                    for x in range(2):
                        for y in range(2):
                            global_idx = (2 * bi + x) * 4 + (2 * bj + y)
                            W[global_idx, col] += coeff_w * W2[l, x * 2 + y]
                            
    return U, V, W

def _build_hybrid_447():
    """
    Build a hybrid decomposition for 4x4x7.
    Uses Strassen for the first 4 columns of B, and Schoolbook for the last 3.
    Rank = 49 + 48 = 97.
    """
    n, m, p = 4, 4, 7
    rank_s = 49
    rank_sb = 48
    rank_total = rank_s + rank_sb
    
    U_total = np.zeros((n * m, rank_total))
    V_total = np.zeros((m * p, rank_total))
    W_total = np.zeros((p * n, rank_total))
    
    # Part 1: Strassen for 4x4x4
    U_s, V_s, W_s = _build_strassen_444()
    
    # U_s is 16x49. U_total first 49 cols are U_s.
    U_total[:, :rank_s] = U_s
    
    # V_s is 16x49. V_total first 49 cols are V_s padded with 12 zeros.
    V_total[:16, :rank_s] = V_s
    
    # W_s is 16x49. W_total first 49 cols are W_s padded with 12 zeros.
    W_total[:16, :rank_s] = W_s
    
    # Part 2: Schoolbook for 4x4x3
    # This computes C[:, 4:7] = A @ B[:, 4:7]
    # B_sub is 4x3.
    # U_sb is 16x48.
    # V_sb is 12x48.
    # W_sb is 12x48.
    
    U_sb = np.zeros((16, 48))
    V_sb = np.zeros((12, 48))
    W_sb = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # U index: i*4+j
                U_sb[i * 4 + j, col] = 1
                # V index: j*3+k (within the 4x3 submatrix)
                # Global V index: j*7 + (4+k)
                V_global_idx = j * 7 + (4 + k)
                V_total[V_global_idx, rank_s + col] = 1
                # W index: k*4+i (within the 4x3 submatrix)
                # Global W index: (4+k)*4 + i
                W_global_idx = (4 + k) * 4 + i
                W_total[W_global_idx, rank_s + col] = 1
                col += 1
                
    # U_sb is standard schoolbook U for 4x4 input.
    # We can just fill U_total[:, rank_s:] with the same pattern as U_sb.
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_total[i * 4 + j, rank_s + col] = 1
                col += 1
                
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # Try hybrid decomposition
    try:
        result = _build_hybrid_447()
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
