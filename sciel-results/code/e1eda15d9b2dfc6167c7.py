"""Matrix multiplication 3x4x8 with rank-88 decomposition using Strassen blocks."""
import time

DIMENSIONS = [3, 4, 8]

# Deadline guard: return best result found before wall-clock limit
_DEADLINE = time.time() + 140.0  # 140s safety margin out of 150s

def _strassen_2x2x2(U, V, W, u_base, v_base, w_base, rank):
    """
    Place a Strassen 2x2x2 decomposition at given base indices.
    
    For C = A * B where A is 2x2, B is 2x2:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    U indices for A elements:
      A00 -> u_base + 0, A01 -> u_base + 1
      A10 -> u_base + 2, A11 -> u_base + 3
    V indices for B elements:
      B00 -> v_base + 0, B01 -> v_base + 1
      B10 -> v_base + 2, B11 -> v_base + 3
    W indices for C elements:
      C00 -> w_base + 0, C01 -> w_base + 1
      C10 -> w_base + 2, C11 -> w_base + 3
    """
    # M1 = (A00 + A11) * (B00 + B11)
    U[u_base + 0][rank] = 1; V[v_base + 0][rank] = 1; W[w_base + 0][rank] = 1
    U[u_base + 3][rank] = 1; V[v_base + 0][rank] = 1; W[w_base + 0][rank] = 1
    U[u_base + 0][rank] = 1; V[v_base + 3][rank] = 1; W[w_base + 0][rank] = 1
    U[u_base + 3][rank] = 1; V[v_base + 3][rank] = 1; W[w_base + 0][rank] = 1
    # Wait - Strassen uses 7 scalar multiplications, not separate terms per element.
    # Let me reconsider. The tensor for 2x2x2 has rank 7.
    # The standard Strassen decomposition gives 7 rank-1 tensors.
    # Each rank-1 tensor contributes to U, V, W.
    
    # Actually, let me use the standard rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    # The 7 terms are:
    # t0: A00*B00 -> contributes to C00, C01 (via B01), C10 (via A10), C11 (via A10*B01)
    # Hmm, this is getting complicated. Let me just use the known rank-7 decomposition.
    
    # Standard Strassen: 7 scalar multiplications
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a scalar product. In the tensor decomposition, each Mi corresponds to ONE rank-1 term.
    # For Mi = (sum of A terms) * (sum of B terms), the U, V entries encode the coefficients.
    
    # M1: (A00 + A11) * (B00 + B11)
    # U: A00 coef 1, A11 coef 1; V: B00 coef 1, B11 coef 1
    # W: C00 coef 1, C01 coef 0, C10 coef 0, C11 coef 1
    U[u_base + 0][rank] = 1
    U[u_base + 3][rank] = 1
    V[v_base + 0][rank] = 1
    V[v_base + 3][rank] = 1
    W[w_base + 0][rank] = 1   # C00
    W[w_base + 3][rank] = 1   # C11
    
    rank += 1
    # M2: (A10 + A11) * B00
    # W: C10 coef 1, C11 coef -1
    U[u_base + 2][rank] = 1
    U[u_base + 3][rank] = 1
    V[v_base + 0][rank] = 1
    W[w_base + 2][rank] = 1   # C10
    W[w_base + 3][rank] = -1  # C11
    
    rank += 1
    # M3: A00 * (B01 - B11)
    # W: C00 coef 0, C01 coef 1, C10 coef 0, C11 coef 1
    U[u_base + 0][rank] = 1
    V[v_base + 1][rank] = 1
    V[v_base + 3][rank] = -1
    W[w_base + 1][rank] = 1   # C01
    W[w_base + 3][rank] = 1   # C11
    
    rank += 1
    # M4: A11 * (B10 - B00)
    # W: C00 coef 1, C10 coef 1
    U[u_base + 3][rank] = 1
    V[v_base + 2][rank] = 1
    V[v_base + 0][rank] = -1
    W[w_base + 0][rank] = 1   # C00
    W[w_base + 2][rank] = 1   # C10
    
    rank += 1
    # M5: (A00 + A01) * B11
    # W: C00 coef -1, C01 coef 1
    U[u_base + 0][rank] = 1
    U[u_base + 1][rank] = 1
    V[v_base + 3][rank] = 1
    W[w_base + 0][rank] = -1  # C00
    W[w_base + 1][rank] = 1   # C01
    
    rank += 1
    # M6: (A10 - A00) * (B00 + B01)
    # W: C11 coef 1
    U[u_base + 2][rank] = 1
    U[u_base + 0][rank] = -1
    V[v_base + 0][rank] = 1
    V[v_base + 1][rank] = 1
    W[w_base + 3][rank] = 1   # C11
    
    rank += 1
    # M7: (A01 - A11) * (B10 + B11)
    # W: C00 coef 1
    U[u_base + 1][rank] = 1
    U[u_base + 3][rank] = -1
    V[v_base + 2][rank] = 1
    V[v_base + 3][rank] = 1
    W[w_base + 0][rank] = 1   # C00
    
    rank += 1
    
    return rank


def _schoolbook_1x2x2(U, V, W, u_base, v_base, w_base, rank):
    """
    Schoolbook 1x2x2: C (1x2) = A (1x2) * B (2x2)
    A has 2 elements: A0, A1 -> u_base+0, u_base+1
    B has 4 elements: B00,B01,B10,B11 -> v_base+0..3
    C has 2 elements: C0,C1 -> w_base+0, w_base+1
    
    C0 = A0*B00 + A1*B10
    C1 = A0*B01 + A1*B11
    
    4 rank-1 terms.
    """
    # A0 * B00 -> C0
    U[u_base + 0][rank] = 1
    V[v_base + 0][rank] = 1
    W[w_base + 0][rank] = 1
    rank += 1
    # A0 * B01 -> C1
    U[u_base + 0][rank] = 1
    V[v_base + 1][rank] = 1
    W[w_base + 1][rank] = 1
    rank += 1
    # A1 * B10 -> C0
    U[u_base + 1][rank] = 1
    V[v_base + 2][rank] = 1
    W[w_base + 0][rank] = 1
    rank += 1
    # A1 * B11 -> C1
    U[u_base + 1][rank] = 1
    V[v_base + 3][rank] = 1
    W[w_base + 1][rank] = 1
    rank += 1
    
    return rank


def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly represents 3x4x8 matrix multiplication."""
    import numpy as np
    n, m, p = 3, 4, 8
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal delta (always 1 for matrix multiplication)
    # Actually for matrix mult, the tensor is: T[i,j,k] = 1 if the corresponding
    # entry exists, i.e., it's the structural tensor.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait, no. The tensor for matrix multiplication has T[i,j,k] = 1 for all valid i,j,k
    # (since each term A[i,j]*B[j,k] appears once in C[i,k]).
    
    # Check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    
    return max_err < 1e-9, max_err


def _build_88_decomposition():
    """
    Build rank-88 decomposition for 3x4x8 matrix multiplication.
    
    Tiling strategy:
    - Split A (3x4) into A_top (2x4) and A_bot (1x4)
    - Split B (4x8) into B_left (4x4) and B_right (4x4) -- actually let's keep B as 4x8
    
    Better tiling:
    - For i in {0,1} (top 2 rows of A):
      - j in {0,1} (left half of A cols): A_top_left (2x2), B_top_left (2x8)
        -> Split B_top_left into 4 (2x2) blocks for k groups {0,1},{2,3},{4,5},{6,7}
        -> Each 2x2x2 uses Strassen: rank 7, total 28
      - j in {2,3} (right half of A cols): A_top_right (2x2), B_bot_left (2x8)  
        Wait, I need to be more careful.
    
    Let me use a cleaner decomposition:
    A is 3x4: rows i=0,1,2; cols j=0,1,2,3
    B is 4x8: rows j=0,1,2,3; cols k=0..7
    C is 3x8: rows i=0,1,2; cols k=0..7
    
    Block decomposition by (i-group, j-group, k-group):
    i-groups: {0,1} (size 2), {2} (size 1)
    j-groups: {0,1} (size 2), {2,3} (size 2)
    k-groups: {0,1}, {2,3}, {4,5}, {6,7} (each size 2)
    
    For each combination (i_grp, j_grp, k_grp):
    - If |i_grp|=2, |j_grp|=2, |k_grp|=2: 2x2x2 -> Strassen rank 7
    - If |i_grp|=1, |j_grp|=2, |k_grp|=2: 1x2x2 -> Schoolbook rank 4
    
    Number of 2x2x2 blocks: 1 (i_grp={0,1}) * 2 (j_grp) * 4 (k_grp) = 8
    Number of 1x2x2 blocks: 1 (i_grp={2}) * 2 (j_grp) * 4 (k_grp) = 8
    
    Total rank = 8*7 + 8*4 = 56 + 32 = 88
    """
    n, m, p = 3, 4, 8
    total_rank = 88
    
    U = [[0] * total_rank for _ in range(n * m)]   # 12 x 88
    V = [[0] * total_rank for _ in range(m * p)]   # 32 x 88
    W = [[0] * total_rank for _ in range(p * n)]   # 24 x 88
    
    rank = 0
    
    i_groups = [(0, 1), (2,)]
    j_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for i_grp in i_groups:
        for j_grp in j_groups:
            for k_grp in k_groups:
                if len(i_grp) == 2:
                    # 2x2x2 Strassen block
                    # U indices: for i in i_grp, j in j_grp -> U[i*m + j]
                    # V indices: for j in j_grp, k in k_grp -> V[j*p + k]
                    # W indices: for k in k_grp, i in i_grp -> W[k*n + i]
                    
                    i0, i1 = i_grp
                    j0, j1 = j_grp
                    k0, k1 = k_grp
                    
                    # U base: u_base such that:
                    #   A00 -> U[i0*m + j0]
                    #   A01 -> U[i0*m + j1]
                    #   A10 -> U[i1*m + j0]
                    #   A11 -> U[i1*m + j1]
                    # We need a mapping. Let's define:
                    # u_base points to U[i0*m + j0], and the 4 A elements are at:
                    #   A00: U[i0*m + j0]
                    #   A01: U[i0*m + j1]
                    #   A10: U[i1*m + j0]
                    #   A11: U[i1*m + j1]
                    
                    # V base: v_base such that:
                    #   B00: V[j0*p + k0]
                    #   B01: V[j0*p + k1]
                    #   B10: V[j1*p + k0]
                    #   B11: V[j1*p + k1]
                    
                    # W base: w_base such that:
                    #   C00: W[k0*n + i0]
                    #   C01: W[k0*n + i1]  -- wait, C is i x k, so C[i,k] -> W[k*n+i]
                    #   C10: W[k1*n + i0]
                    #   C11: W[k1*n + i1]
                    
                    # For Strassen, we need U elements in order A00,A01,A10,A11
                    # and V elements B00,B01,B10,B11 and W elements C00,C01,C10,C11
                    # But they're not contiguous in memory.
                    
                    # Let me write a helper that places Strassen with explicit index lists.
                    u_idx = [i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1]  # A00,A01,A10,A11
                    v_idx = [j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1]  # B00,B01,B10,B11
                    w_idx = [k0*n+i0, k0*n+i1, k1*n+i0, k1*n+i1]  # C00,C01,C10,C11
                    
                    # Strassen 7 terms
                    # M1: (A00+A11)*(B00+B11) -> C00 + C11
                    U[u_idx[0]][rank] += 1
                    U[u_idx[3]][rank] += 1
                    V[v_idx[0]][rank] += 1
                    V[v_idx[3]][rank] += 1
                    W[w_idx[0]][rank] += 1   # C00
                    W[w_idx[3]][rank] += 1   # C11
                    rank += 1
                    
                    # M2: (A10+A11)*B00 -> C10 - C11
                    U[u_idx[2]][rank] += 1
                    U[u_idx[3]][rank] += 1
                    V[v_idx[0]][rank] += 1
                    W[w_idx[2]][rank] += 1   # C10
                    W[w_idx[3]][rank] += -1  # C11
                    rank += 1
                    
                    # M3: A00*(B01-B11) -> C01 + C11
                    U[u_idx[0]][rank] += 1
                    V[v_idx[1]][rank] += 1
                    V[v_idx[3]][rank] += -1
                    W[w_idx[1]][rank] += 1   # C01
                    W[w_idx[3]][rank] += 1   # C11
                    rank += 1
                    
                    # M4: A11*(B10-B00) -> C00 + C10
                    U[u_idx[3]][rank] += 1
                    V[v_idx[2]][rank] += 1
                    V[v_idx[0]][rank] += -1
                    W[w_idx[0]][rank] += 1   # C00
                    W[w_idx[2]][rank] += 1   # C10
                    rank += 1
                    
                    # M5: (A00+A01)*B11 -> -C00 + C01
                    U[u_idx[0]][rank] += 1
                    U[u_idx[1]][rank] += 1
                    V[v_idx[3]][rank] += 1
                    W[w_idx[0]][rank] += -1  # C00
                    W[w_idx[1]][rank] += 1   # C01
                    rank += 1
                    
                    # M6: (A10-A00)*(B00+B01) -> C11
                    U[u_idx[2]][rank] += 1
                    U[u_idx[0]][rank] += -1
                    V[v_idx[0]][rank] += 1
                    V[v_idx[1]][rank] += 1
                    W[w_idx[3]][rank] += 1   # C11
                    rank += 1
                    
                    # M7: (A01-A11)*(B10+B11) -> C00
                    U[u_idx[1]][rank] += 1
                    U[u_idx[3]][rank] += -1
                    V[v_idx[2]][rank] += 1
                    V[v_idx[3]][rank] += 1
                    W[w_idx[0]][rank] += 1   # C00
                    rank += 1
                    
                else:
                    # 1x2x2 Schoolbook block
                    # A is 1x2: A0, A1 -> U[i0*m+j0], U[i0*m+j1]
                    # B is 2x2: B00,B01,B10,B11 -> V[j0*p+k0], V[j0*p+k1], V[j1*p+k0], V[j1*p+k1]
                    # C is 1x2: C0, C1 -> W[k0*n+i0], W[k1*n+i0]
                    
                    i0 = i_grp[0]
                    j0, j1 = j_grp
                    k0, k1 = k_grp
                    
                    u0 = i0*m+j0  # A0
                    u1 = i0*m+j1  # A1
                    v00 = j0*p+k0
                    v01 = j0*p+k1
                    v10 = j1*p+k0
                    v11 = j1*p+k1
                    w0 = k0*n+i0  # C0
                    w1 = k1*n+i0  # C1
                    
                    # A0*B00 -> C0
                    U[u0][rank] += 1
                    V[v00][rank] += 1
                    W[w0][rank] += 1
                    rank += 1
                    
                    # A0*B01 -> C1
                    U[u0][rank] += 1
                    V[v01][rank] += 1
                    W[w1][rank] += 1
                    rank += 1
                    
                    # A1*B10 -> C0
                    U[u1][rank] += 1
                    V[v10][rank] += 1
                    W[w0][rank] += 1
                    rank += 1
                    
                    # A1*B11 -> C1
                    U[u1][rank] += 1
                    V[v11][rank] += 1
                    W[w1][rank] += 1
                    rank += 1
    
    assert rank == total_rank, f"Expected rank {total_rank}, got {rank}"
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    result = _build_88_decomposition()
    return result
