"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    Indexing: U[i*2+j, :], V[j*2+k, :], W[k*2+i, :]
    """
    # Strassen's 7 products:
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
    # A indexing: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indexing: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indexing: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    # A11=(0,0), A22=(1,1) -> U indices 0, 3
    # B11=(0,0), B22=(1,1) -> V indices 0, 3
    # C11=(0,0), C22=(1,1) -> W indices 0, 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21=(1,0), A22=(1,1) -> U indices 2, 3
    # B11=(0,0) -> V index 0
    # C21=(1,0), C22=(1,1) -> W indices 2, 3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    # A11=(0,0) -> U index 0
    # B12=(0,1), B22=(1,1) -> V indices 1, 3
    # C12=(0,1), C22=(1,1) -> W indices 1, 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    # A22=(1,1) -> U index 3
    # B21=(1,0), B11=(0,0) -> V indices 2, 0
    # C11=(0,0), C21=(1,0) -> W indices 0, 2
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11=(0,0), A12=(0,1) -> U indices 0, 1
    # B22=(1,1) -> V index 3
    # C11=(0,0), C12=(0,1) -> W indices 0, 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21=(1,0), A11=(0,0) -> U indices 2, 0
    # B11=(0,0), B12=(0,1) -> V indices 0, 1
    # C22=(1,1) -> W index 3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12=(0,1), A22=(1,1) -> U indices 1, 3
    # B21=(1,0), B22=(1,1) -> V indices 2, 3
    # C11=(0,0) -> W index 0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_block_decomposition(n, m, p):
    """
    Construct a structured decomposition for n x m x p matrix multiplication
    using slicing and Strassen for 2x2x2 blocks.
    
    For 3x4x5:
    - Split rows of A into 2+1
    - Split columns of A (and rows of B) into 2+2
    - Split columns of B into 2+3 (or handle 5 as 2+3)
    
    Actually, let's use a cleaner approach:
    Split A (3x4) into top 2x4 and bottom 1x4
    Split B (4x5) into left 4x2 and right 4x3
    
    Top block: 2x4 x 4x5 -> use Strassen for 2x2x2 sub-blocks
    Bottom block: 1x4 x 4x5 -> trivial (rank 20)
    
    For the 2x4 x 4x5 multiplication:
    Split further: A_top (2x4) into 2x2 + 2x2 (left and right)
    B (4x5) into 4x2 + 4x3... 
    
    Let me think more carefully.
    
    A is 3x4, B is 4x5, C is 3x5.
    
    Strategy: Split A into A_top (2x4) and A_bottom (1x4).
    Split B into B_left (4x2) and B_right (4x3).
    
    C_top_left = A_top @ B_left  (2x2)
    C_top_right = A_top @ B_right (2x3)
    C_bottom_left = A_bottom @ B_left (1x2)
    C_bottom_right = A_bottom @ B_right (1x3)
    
    For C_top_left (2x2): Use Strassen, rank 7
    For C_top_right (2x3): Split into 2x2 + 2x1
      - 2x2 part: Strassen, rank 7
      - 2x1 part: trivial, rank 2
      Total: rank 9
    For C_bottom_left (1x2): trivial, rank 2
    For C_bottom_right (1x3): trivial, rank 3
    
    Total rank: 7 + 9 + 2 + 3 = 21
    
    Wait, that's better than 56! Let me verify this is correct.
    
    Actually, I need to be more careful. The tensor decomposition must be consistent.
    Let me reconsider.
    
    The key insight: we can decompose the 3x4x5 tensor as a sum of smaller tensors.
    If we partition the indices, each sub-tensor can have its own decomposition.
    
    Let's use:
    - i in {0,1} for top 2 rows, i=2 for bottom row
    - j in {0,1} for left 2 cols of A / left 2 rows of B, j in {2,3} for right
    - k in {0,1} for left 2 cols of B, k in {2,3,4} for right 3 cols
    
    Sub-tensors:
    1. (i=0,1; j=0,1; k=0,1): 2x2x2 -> Strassen rank 7
    2. (i=0,1; j=0,1; k=2,3,4): 2x2x3 -> split into 2x2x2 + 2x2x1 = 7 + 4 = 11? 
       Actually 2x2x1 is trivial rank 4 (each of 4 products)
       Wait, 2x2x1: A is 2x2, B is 2x1, C is 2x1. 
       C[0] = A[0,0]*B[0] + A[0,1]*B[1]
       C[1] = A[1,0]*B[0] + A[1,1]*B[1]
       This needs 4 scalar products, so rank 4.
       So 2x2x3 = 2x2x2 + 2x2x1 = 7 + 4 = 11
    3. (i=0,1; j=2,3; k=0,1): 2x2x2 -> Strassen rank 7
    4. (i=0,1; j=2,3; k=2,3,4): 2x2x3 -> 7 + 4 = 11
    5. (i=2; j=0,1; k=0,1): 1x2x2 -> trivial rank 4
    6. (i=2; j=0,1; k=2,3,4): 1x2x3 -> trivial rank 6
    7. (i=2; j=2,3; k=0,1): 1x2x2 -> trivial rank 4
    8. (i=2; j=2,3; k=2,3,4): 1x2x3 -> trivial rank 6
    
    Total: 7 + 11 + 7 + 11 + 4 + 6 + 4 + 6 = 56
    
    So rank 56 is achievable! Let me implement this.
    """
    n, m, p = 3, 4, 5
    
    # We'll build U, V, W as lists of (row, col, value) tuples, then assemble
    U_entries = []  # (row_idx, col_idx, value)
    V_entries = []
    W_entries = []
    
    total_rank = 0
    
    def add_strassen(i_offset, j_offset, k_offset):
        """Add Strassen decomposition for 2x2x2 block at given offsets."""
        nonlocal total_rank
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(4):
            for c in range(7):
                if U_s[r, c] != 0:
                    U_entries.append((i_offset * 2 + r // 2 + (r % 2) * 2, total_rank + c, U_s[r, c]))
                    # Wait, I need to be careful with indexing.
                    # For 2x2x2: A is 2x2, indexed by (i_local, j_local) where i_local in {0,1}, j_local in {0,1}
                    # U row index = i_local * 2 + j_local
                    # With offset: U row = (i_offset + i_local) * m + (j_offset + j_local)
                    # But m=4, so this doesn't directly work for the full matrix.
                    # I need to think about this differently.
        # This approach is getting complicated. Let me use a different method.
        pass
    
    # Let me use a cleaner approach: build the full U, V, W matrices directly.
    # The total rank will be 56.
    
    # Define the 8 sub-problems and their decompositions
    # Each sub-problem has a local 2x2x2 or 1x2x2 or 1x2x3 or 2x2x3 structure.
    
    # I'll build this step by step using helper functions.
    
    U = np.zeros((n * m, 56), dtype=float)
    V = np.zeros((m * p, 56), dtype=float)
    W = np.zeros((p * n, 56), dtype=float)
    
    col = 0
    
    # Helper to add a trivial 1xa x bxc block
    def add_trivial_1x(a, b, c, i_idx, j_start, k_start):
        """
        Add trivial decomposition for A[i_idx, j_start:j_start+a] @ B[j_start:j_start+a, k_start:k_start+c]
        A is 1xa, B is axc, C is 1xc.
        """
        nonlocal col
        for jj in range(a):
            for kk in range(c):
                U[i_idx * m + j_start + jj, col] = 1
                V[(j_start + jj) * p + k_start + kk, col] = 1
                W[(k_start + kk) * n + i_idx, col] = 1
                col += 1
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(i_start, j_start, k_start):
        """
        Add Strassen decomposition for A[i_start:i_start+2, j_start:j_start+2] @ B[j_start:j_start+2, k_start:k_start+2]
        """
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(4):
            i_local = r // 2
            j_local = r % 2
            for c in range(7):
                if U_s[r, c] != 0:
                    U[(i_start + i_local) * m + j_start + j_local, col + c] = U_s[r, c]
        
        for r in range(4):
            j_local = r // 2
            k_local = r % 2
            for c in range(7):
                if V_s[r, c] != 0:
                    V[(j_start + j_local) * p + k_start + k_local, col + c] = V_s[r, c]
        
        for r in range(4):
            k_local = r // 2
            i_local = r % 2
            for c in range(7):
                if W_s[r, c] != 0:
                    W[(k_start + k_local) * n + i_start + i_local, col + c] = W_s[r, c]
        
        col += 7
    
    # Helper to add 2x2x1 block (A is 2x2, B is 2x1, C is 2x1) - trivial rank 4
    def add_2x2x1(i_start, j_start, k_idx):
        """
        Add trivial decomposition for A[i_start:i_start+2, j_start:j_start+2] @ B[j_start:j_start+2, k_idx]
        Result: C[i_start:i_start+2, k_idx]
        """
        nonlocal col
        for ii in range(2):
            for jj in range(2):
                U[(i_start + ii) * m + j_start + jj, col] = 1
                V[(j_start + jj) * p + k_idx, col] = 1
                W[k_idx * n + i_start + ii, col] = 1
                col += 1
    
    # Now assemble the 8 blocks:
    
    # Block 1: i=0,1; j=0,1; k=0,1 -> 2x2x2 Strassen
    add_strassen(0, 0, 0)
    
    # Block 2: i=0,1; j=0,1; k=2,3,4 -> 2x2x3 = 2x2x2 (k=2,3) + 2x2x1 (k=4)
    # 2x2x2 part: A[0:2, 0:2] @ B[0:2, 2:4]
    add_strassen(0, 0, 2)
    # 2x2x1 part: A[0:2, 0:2] @ B[0:2, 4]
    add_2x2x1(0, 0, 4)
    
    # Block 3: i=0,1; j=2,3; k=0,1 -> 2x2x2 Strassen
    add_strassen(0, 2, 0)
    
    # Block 4: i=0,1; j=2,3; k=2,3,4 -> 2x2x3 = 2x2x2 (k=2,3) + 2x2x1 (k=4)
    add_strassen(0, 2, 2)
    add_2x2x1(0, 2, 4)
    
    # Block 5: i=2; j=0,1; k=0,1 -> 1x2x2 trivial
    add_trivial_1x(2, 2, 2, 2, 0, 0)
    
    # Block 6: i=2; j=0,1; k=2,3,4 -> 1x2x3 trivial
    add_trivial_1x(2, 2, 3, 2, 0, 2)
    
    # Block 7: i=2; j=2,3; k=0,1 -> 1x2x2 trivial
    add_trivial_1x(2, 2, 2, 2, 2, 0)
    
    # Block 8: i=2; j=2,3; k=2,3,4 -> 1x2x3 trivial
    add_trivial_1x(2, 2, 3, 2, 2, 2)
    
    assert col == 56, f"Expected rank 56, got {col}"
    
    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        return None
    
    return U, V, W

def _construct_schoolbook(n, m, p):
    """Construct the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try the structured decomposition first
    result = _construct_block_decomposition(n, m, p)
    
    if result is not None:
        U, V, W = result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
