"""Program for 2x5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

DEADLINE = 140.0

def _verify(U, V, W, n, m, p):
    """Verify exact tensor decomposition using Fraction arithmetic."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _strassen_2x2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, U, V, W, base_col, n, m, p):
    """
    Add 7 Strassen terms for 2x2x2 block multiplication.
    A_rows: [r1, r2], A_cols: [c1, c2]
    B_rows: [r1, r2], B_cols: [k1, k2]
    C_rows: [r1, r2], C_cols: [k1, k2]
    """
    r1, r2 = A_rows
    c1, c2 = A_cols
    b1, b2 = B_rows
    k1, k2 = B_cols
    cr1, cr2 = C_rows
    ck1, ck2 = C_cols
    
    # Strassen's 7 multiplications for 2x2x2:
    # m1 = (A11 + A22)(B11 + B22)
    # m2 = (A21 + A22)B11
    # m3 = A11(B21 - B22)
    # m4 = A22(B12 - B11)
    # m5 = (A11 + A12)B22
    # m6 = (A21 - A11)(B11 + B12)
    # m7 = (A12 - A22)(B21 + B22)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        col = base_col[0]
        U[u_row][col] += u_val
        V[v_row][col] += v_val
        W[w_row][col] += w_val
        base_col[0] += 1
    
    # m1 = (A[r1,c1] + A[r2,c2]) * (B[b1,k1] + B[b2,k2])
    # This is a product of sums, so we need 4 terms
    add_term(r1 * m + c1, b1 * p + k1, ck1 * n + cr1, 1, 1, 1)
    add_term(r1 * m + c1, b2 * p + k2, ck1 * n + cr1, 1, 1, 1)
    add_term(r2 * m + c2, b1 * p + k1, ck1 * n + cr1, 1, 1, 1)
    add_term(r2 * m + c2, b2 * p + k2, ck1 * n + cr1, 1, 1, 1)
    # m1 contributes to C11 and C22 with coefficient 1
    # We need to track which output each term contributes to
    # Actually, Strassen works differently - each m_i is a scalar product
    # Let me reconsider.
    
    # Actually, for tensor decomposition, each "multiplication" in Strassen
    # corresponds to a rank-1 tensor term. The 7 Strassen multiplications
    # are:
    # m1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    # m2 = (A21+A22)B11 -> contributes to C21, C22
    # etc.
    
    # But in tensor decomposition, we decompose the tensor as sum of rank-1 terms.
    # Each rank-1 term is u_i * v_j * w_k for some indices.
    # The Strassen algorithm gives us 7 scalar multiplications, but each
    # multiplication involves sums of A and B entries, which means each
    # "multiplication" expands to multiple rank-1 terms.
    
    # Let me just directly construct the 7 Strassen rank-1 terms properly.
    # Actually, Strassen's algorithm for 2x2x2 uses 7 multiplications where
    # each multiplication is of the form (linear combo of A entries) * (linear combo of B entries)
    # and the result is a linear combo of C entries.
    
    # For the tensor M_{2,2,2}, the rank is 7. The 7 rank-1 tensors are:
    # T1: (e1+e2)(e1+e2)(e1+e2) -> but this isn't quite right either.
    
    # Let me think about this more carefully. The 2x2x2 multiplication tensor is:
    # sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # = sum_{i,j,k} A[i,j] * B[j,k] * delta_{ik} (where delta is the output)
    
    # Strassen shows this can be done with 7 rank-1 tensors. The 7 terms are:
    # 1. (A00+A11)(B00+B11) -> C00 += this, C11 += this
    #    As rank-1 tensors: A00*B00*C00, A00*B11*C00, A11*B00*C00, A11*B11*C00
    #    Wait, that's 4 terms, not 1.
    
    # I think I'm confusing the algorithm with the tensor decomposition.
    # Let me just use the known rank-7 decomposition of M_{2,2,2}.
    
    # The rank-7 decomposition of M_{2,2,2} has these 7 rank-1 terms:
    # (using 0-indexed: A rows 0,1; cols 0,1; B rows 0,1; cols 0,1; C rows 0,1; cols 0,1)
    
    # Term 1: A[0,0]*B[0,0]*C[0,0] + ... no, each rank-1 term is a single product.
    
    # Actually, a rank-1 tensor is u ⊗ v ⊗ w where u, v, w are vectors.
    # For M_{2,2,2}, we need 7 such tensors that sum to the multiplication tensor.
    
    # Let me just directly write the 7 terms based on Strassen's algorithm:
    # m1 = (A00+A11)(B00+B11)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #   C00 += m1, C11 += m1
    #   So rank-1 terms: (A00,B00,C00), (A00,B11,C00), (A11,B00,C00), (A11,B11,C00),
    #                    (A00,B00,C11), (A00,B11,C11), (A11,B00,C11), (A11,B11,C11)
    #   That's 8 terms, not 1.
    
    # I see the issue. Strassen's algorithm uses 7 scalar multiplications, but
    # each multiplication is of sums, so in terms of rank-1 tensor decomposition,
    # it's not directly 7 rank-1 terms.
    
    # Let me just fall back to schoolbook for the 2x2x2 blocks and see if the
    # overall structure gives a better rank.
    # 2x2x2 schoolbook rank = 8
    # 4 blocks of 2x2x2 = 32 terms
    # Plus cross terms...
    
    # Actually, let me just implement a direct structured decomposition.
    # For 2x5x5, let me try to find a decomposition with rank < 50.
    pass

def _construct_structured(n, m, p):
    """Construct a structured decomposition for 2x5x5."""
    # Let me try a different approach: directly construct rank-1 terms
    # that sum to the multiplication tensor.
    
    # The multiplication tensor is:
    # sum_{i=0}^{1} sum_{j=0}^{4} sum_{k=0}^{4} A[i,j] * B[j,k] * C[i,k]
    
    # A rank-1 term is: (sum_a u_a A[a]) * (sum_b v_b B[b]) * (sum_c w_c C[c])
    # where u, v, w are vectors.
    
    # For the tensor to decompose, we need:
    # sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 if (i,j,k) is a valid triple, 0 otherwise
    # where u_r, v_r, w_r are the r-th rank-1 tensor's components.
    
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k], 0 otherwise
    # But the tensor is really: T[i,j,k] = delta_{jk} ... no.
    
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is:
    # M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of equations. Let me try to find a low-rank solution.
    
    # For now, let me just return schoolbook and try to improve via search.
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    start_time = time.time()
    
    # Try to find lower rank via random search or structured approach
    # For now, just return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
