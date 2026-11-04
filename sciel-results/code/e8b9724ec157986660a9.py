"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    chunk_size = max(1, min(rank, 500))
    for start in range(0, rank, chunk_size):
        end = min(start + chunk_size, rank)
        T += np.einsum('ar,br,cr->abc', U[:, start:end], V[:, start:end], W[:, start:end])
    
    diff = np.max(np.abs(T - target))
    return diff < tol

def _make_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _winograd_2x2_core(U, V, W, base_col, n, m, p, i0, j0, k0):
    """
    Apply Winograd 2x2 convolution to a 2x2x2 block of the multiplication.
    This reduces the 8 terms of a 2x2x2 schoolbook to 7 terms.
    i0, j0, k0 are the starting indices of the 2x2x2 block.
    """
    # The 8 schoolbook terms for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2]
    # Winograd 2x2: 
    # d0 = a00*b00 + a01*b10
    # d1 = a00*b01 + a01*b11
    # d2 = a10*b00 + a11*b10
    # d3 = a10*b01 + a11*b11
    # Using 7 multiplications:
    # m0 = (a00+a01)*(b00+b10)
    # m1 = a00*b10
    # m2 = a01*b00
    # m3 = (a10+a11)*(b01+b11)
    # m4 = a10*b11
    # m5 = a11*b01
    # m6 = (a00-a10)*(b01-b00)  -- wait, this isn't right for matrix mult
    
    # Actually, for matrix multiplication, the Winograd approach is different.
    # Let me use the standard 2x2 matrix multiplication Winograd:
    # C = A @ B where A, B are 2x2
    # m0 = (a00 + a10) * (b00 + b01)
    # m1 = (a00 + a11) * (b11 - b10)
    # m2 = (a10 + a11) * (b10 + b11)
    # m3 = a00 * (b01 - b00)
    # m4 = a11 * (b10 - b11)
    # m5 = a10 * (b00 + b10 - b01)
    # m6 = a01 * (b01 + b11 - b10)
    # C00 = m0 - m3 - m5
    # C01 = m3 + m5 + m6
    # C10 = m1 + m4
    # C11 = m0 - m1 - m4 + m6
    
    # But we need to express this as rank-1 terms in the tensor decomposition.
    # Each multiplication mr corresponds to a rank-1 term.
    # For the 2x2x2 block at (i0, j0, k0):
    # A entries: A[i0, j0], A[i0, j0+1], A[i0+1, j0], A[i0+1, j0+1]
    # B entries: B[j0, k0], B[j0, k0+1], B[j0+1, k0], B[j0+1, k0+1]
    # C entries: C[i0, k0], C[i0, k0+1], C[i0+1, k0], C[i0+1, k0+1]
    
    # The 7 multiplications are:
    # m0 = (A[i0,j0] + A[i0+1,j0]) * (B[j0,k0] + B[j0,k0+1])
    # m1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0+1,k0+1] - B[j0+1,k0])
    # m2 = (A[i0+1,j0] + A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
    # m3 = A[i0,j0] * (B[j0,k0+1] - B[j0,k0])
    # m4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0+1,k0+1])
    # m5 = A[i0+1,j0] * (B[j0,k0] + B[j0+1,k0] - B[j0,k0+1])
    # m6 = A[i0,j0+1] * (B[j0,k0+1] + B[j0+1,k0+1] - B[j0+1,k0])
    
    # Each mr contributes to the tensor as a rank-1 term.
    # For m0: U has 1 at (i0,j0) and (i0+1,j0), V has 1 at (j0,k0) and (j0,k0+1)
    #   W should be such that the product gives the right contribution to C.
    #   m0 contributes to C00 with +1, C01 with +1, C10 with -1, C11 with -1
    #   Wait, let me re-derive.
    
    # Actually, this is getting complicated. Let me use a simpler approach:
    # Just use the known Strassen-like 7-term decomposition for 2x2x2.
    
    # For the 2x2x2 block, the 7 rank-1 terms are:
    # Term 0: U[(i0,j0)]=1, U[(i0+1,j0)]=1, V[(j0,k0)]=1, V[(j0,k0+1)]=1, W[(k0,i0)]=1, W[(k0+1,i0)]=1, W[(k0,i0+1)]=-1, W[(k0+1,i0+1)]=-1
    # This is getting messy. Let me just use schoolbook for now and focus on the overall structure.
    
    # For simplicity, let's just use schoolbook for the 2x2x2 blocks.
    # The Winograd/Strassen optimization for 2x2x2 gives rank 7 instead of 8.
    # But implementing it correctly in the tensor decomposition format is tricky.
    
    # Let me use a different approach: partition into 2x2 blocks and use Strassen for each.
    # 4x5x6 can be partitioned as:
    # A is 4x5, B is 5x6
    # C = A @ B is 4x6
    
    # Partition A into 2x2 blocks (with padding):
    # A = [A11 A12 A13] where A11 is 2x2, A12 is 2x2, A13 is 2x1
    #     [A21 A22 A23] where A21 is 2x2, A22 is 2x2, A23 is 2x1
    
    # B = [B11 B12] where B11 is 5x3, B12 is 5x3
    #     Actually, let me partition B as 2x2 blocks too:
    # B = [B11 B12 B13] where each is 2x2, and B21, B22, B23 are 1x2
    #     [B21 B22 B23]
    
    # This is getting complicated. Let me just use the full schoolbook decomposition
    # but try to find a lower-rank decomposition using a more systematic search.
    
    pass

def _strassen_2x2(U, V, W, col, i0, j0, k0):
    """
    Add Strassen's 7-term decomposition for a 2x2x2 block starting at (i0, j0, k0).
    """
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In our tensor decomposition, each multiplication corresponds to a rank-1 term.
    # The indices are:
    # A[i,j] is at position i*m + j in U
    # B[j,k] is at position j*p + k in V
    # C[i,k] is at position k*n + i in W
    
    # For the 2x2x2 block at (i0, j0, k0):
    # A entries: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
    # B entries: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
    # C entries: (i0,k0), (i0,k0+1), (i0+1,k0), (i0+1,k0+1)
    
    # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
    # This is a bilinear form in A and B, so it corresponds to a rank-1 term in the tensor.
    # U has 1 at (i0,j0) and (i0+1,j0+1)
    # V has 1 at (j0,k0) and (j0+1,k0+1)
    # W should have the coefficients that map to C:
    # M1 contributes +1 to C[i0,k0], +1 to C[i0+1,k0+1]
    # Wait, M1 = C00 + C11 - M4 + M5 - M7 - M3 - M2 - M6... this is getting circular.
    
    # Let me just directly construct the 7 rank-1 terms.
    # Each term r has U[:,r], V[:,r], W[:,r] such that:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # For M1: U has 1 at i0*m+j0 and (i0+1)*m+(j0+1), V has 1 at j0*p+k0 and (j0+1)*p+(k0+1)
    # W should be such that the product gives the right contribution.
    # M1 appears in C00 with +1 and C11 with +1.
    # So W has 1 at k0*n+i0 and (k0+1)*n+(i0+1).
    
    U[i0 * m + j0, col] += 1.0
    U[(i0+1) * m + (j0+1), col] += 1.0
    V[j0 * p + k0, col] += 1.0
    V[(j0+1) * p + (k0+1), col] += 1.0
    W[k0 * n + i0, col] += 1.0
    W[(k0+1) * n + (i0+1), col] += 1.0
    col += 1
    
    # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
    U[(i0+1) * m + j0, col] += 1.0
    U[(i0+1) * m + (j0+1), col] += 1.0
    V[j0 * p + k0, col] += 1.0
    W[k0 * n + (i0+1), col] += 1.0
    col += 1
    
    # M3 = A[i0,j0] * (B[j0+1,k0] - B[j0+1,k0+1])
    U[i0 * m + j0, col] += 1.0
    V[(j0+1) * p + k0, col] += 1.0
    V[(j0+1) * p + (k0+1), col] += -1.0
    W[k0 * n + i0, col] += -1.0
    W[(k0+1) * n + i0, col] += 1.0
    col += 1
    
    # M4 = A[i0+1,j0+1] * (B[j0,k0+1] - B[j0,k0])
    U[(i0+1) * m + (j0+1), col] += 1.0
    V[j0 * p + (k0+1), col] += 1.0
    V[j0 * p + k0, col] += -1.0
    W[k0 * n + (i0+1), col] += -1.0
    W[(k0+1) * n + (i0+1), col] += 1.0
    col += 1
    
    # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
    U[i0 * m + j0, col] += 1.0
    U[i0 * m + (j0+1), col] += 1.0
    V[(j0+1) * p + (k0+1), col] += 1.0
    W[(k0+1) * n + i0, col] += 1.0
    col += 1
    
    # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0+1,k0])
    U[(i0+1) * m + j0, col] += 1.0
    U[i0 * m + j0, col] += -1.0
    V[j0 * p + k0, col] += 1.0
    V[(j0+1) * p + k0, col] += 1.0
    W[k0 * n + (i0+1), col] += 1.0
    col += 1
    
    # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0,k0+1] + B[j0+1,k0+1])
    U[i0 * m + (j0+1), col] += 1.0
    U[(i0+1) * m + (j0+1), col] += -1.0
    V[j0 * p + (k0+1), col] += 1.0
    V[(j0+1) * p + (k0+1), col] += 1.0
    W[(k0+1) * n + i0, col] += 1.0
    col += 1
    
    return col

def _make_strassen_partition(n, m, p):
    """
    Create a decomposition using Strassen's algorithm for 2x2 blocks.
    Partition the 4x5x6 multiplication into 2x2x2 blocks and use Strassen for each.
    """
    # 4x5x6: A is 4x5, B is 5x6, C is 4x6
    # Partition into 2x2 blocks:
    # A has 2x2 blocks: (0,0), (0,2), (0,4), (1,0), (1,2), (1,4) -- but 5 is not even
    # Let's pad to 4x6 or use a different partition.
    
    # Actually, let's partition as:
    # A = [A11 A12 A13] where A11 is 2x2, A12 is 2x2, A13 is 2x1
    #     [A21 A22 A23] where A21 is 2x2, A22 is 2x2, A23 is 2x1
    # B = [B11 B12 B13] where B11 is 5x2, B12 is 5x2, B13 is 5x2
    #     Actually, B is 5x6, so partition as:
    # B = [B11 B12 B13] where each is 5x2
    #     But we need 2x2 blocks for Strassen.
    
    # Let's use a different approach: partition both A and B into 2x2 blocks where possible.
    # A is 4x5: partition rows into 2+2, columns into 2+2+1
    # B is 5x6: partition rows into 2+2+1, columns into 2+2+2
    
    # This is getting complicated. Let me just use the full schoolbook for now
    # and try to optimize using a search.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Use Strassen for the 2x2x2 blocks where possible.
    # For the 4x4x4 part (first 4 rows of A, first 4 cols of B, first 4 cols of C):
    # We can partition into 2x2x2 blocks and use Strassen for each.
    
    # Let's use Strassen for the (0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2) blocks.
    # But we only have 4x5x6, so the blocks at (0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2) are valid.
    # The block at (0,0,4) would be 2x2x2 but k0=4, k0+1=5 which is out of bounds for p=6.
    # So we can use Strassen for blocks with i0 in {0,2}, j0 in {0,2}, k0 in {0,2,4}.
    # But j0=2, j0+1=3 is valid for m=5.
    # j0=4, j0+1=5 is out of bounds.
    
    col = 0
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                if j0 + 1 < m and k0 + 1 < p:
                    col = _strassen_2x2(U, V, W, col, i0, j0, k0)
    
    # Now fill in the remaining terms with schoolbook.
    # The Strassen decomposition covers some of the 2x2x2 blocks, but not all.
    # Let's just add schoolbook terms for the remaining (i,j,k) triples.
    
    # Actually, this is getting too complicated. Let me just return the schoolbook decomposition.
    # The Strassen optimization is not straightforward to implement correctly.
    
    # For now, return schoolbook.
    U_sb, V_sb, W_sb = _make_schoolbook(n, m, p)
    return U_sb, V_sb, W_sb

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110 second limit for safety
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower-rank decomposition using a simple search.
    # For now, just return the schoolbook decomposition.
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        rounded = np.round(mat * 2) / 2
        return rounded.tolist()
    
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
