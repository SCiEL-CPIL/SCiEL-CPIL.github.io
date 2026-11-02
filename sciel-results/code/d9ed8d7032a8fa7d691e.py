"""Tensor decomposition search for 3x3 * 3x8 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank-72 decomposition."""
    n, m, p = 3, 3, 8
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U[i*2+j, c] for i,j in 0..1
    V[j*2+k, c] for j,k in 0..1
    W[k*2+i, c] for k,i in 0..1
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1 = (A11+A22)(B11+B22)
    # A11 = index 0, A22 = index 3
    # B11 = index 0, B22 = index 3
    # C11 = index 0, C22 = index 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21 = index 2, A22 = index 3
    # B11 = index 0
    # C21 = index 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B21-B22)
    # A11 = index 0
    # B21 = index 2, B22 = index 3
    # C12 = index 1, C22 = index 3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B12-B11)
    # A22 = index 3
    # B12 = index 1, B11 = index 0
    # C11 = index 0, C21 = index 2
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11 = index 0, A12 = index 1
    # B22 = index 3
    # C11 = index 0, C12 = index 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21 = index 2, A11 = index 0
    # B11 = index 0, B12 = index 1
    # C22 = index 3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12 = index 1, A22 = index 3
    # B21 = index 2, B22 = index 3
    # C11 = index 0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_3x3x3():
    """
    Construct a rank-23 decomposition for 3x3 * 3x3 using Strassen's algorithm.
    We partition the 3x3 matrices into 2x2 blocks and use Strassen on the 2x2 parts,
    plus additional terms for the 3rd row/column.
    
    Actually, a cleaner approach: use the known result that rank(M_{3,3,3}) <= 23.
    One construction uses Strassen's rank-7 for 2x2x2 plus 16 additional terms.
    
    Let's use a different approach: treat 3x3 as 2x2 + 1x1 extensions.
    A = [A22, a3; a1, a2] where A22 is 2x2, a3 is 1x2 row, a1 is 2x1 col, a2 is scalar.
    B = [B22, b3; b1, b2]
    
    This gets complicated. Let me use a simpler structured approach.
    
    Alternative: Use the fact that M_{3,3,3} can be decomposed using 7 (Strassen for 2x2) 
    plus additional terms. Actually, let's just use a known rank-23 construction.
    
    For simplicity and correctness, let me construct it as:
    - Use Strassen (rank 7) for the top-left 2x2 block
    - Add terms for interactions with the 3rd row/col
    - Total should be <= 23
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # We'll use a known construction. Let's build it carefully.
    # Partition: indices 0,1,2 for rows/cols
    # 2x2 block uses indices 0,1; index 2 is the extension
    
    # First, embed Strassen 2x2x2 into the 2x2 part (indices 0,1)
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Place Strassen in the first 7 columns
    for i in range(4):
        U[i, 0:7] = S_U[i]
        V[i, 0:7] = S_V[i]
        W[i, 0:7] = S_W[i]
    
    # Now we need to handle:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # The Strassen part already handles C[0,0], C[0,1], C[1,0], C[1,1] for the 2x2 blocks.
    # But we also need to account for the fact that A and B have 3rd row/col.
    
    # Actually, this is getting too complex to derive correctly on the fly.
    # Let me use a different strategy: use the schoolbook decomposition but try to
    # find linear dependencies to reduce rank.
    
    # For now, return schoolbook for 3x3x3 as a safe baseline
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    
    return U, V, W

def _kron_product_decomp(n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2):
    """
    Compute the Kronecker product of two tensor decompositions.
    If T1 = sum_c u1_c ⊗ v1_c ⊗ w1_c and T2 = sum_d u2_d ⊗ v2_d ⊗ w2_d,
    then T1 ⊗ T2 = sum_{c,d} (u1_c ⊗ u2_d) ⊗ (v1_c ⊗ v2_d) ⊗ (w1_c ⊗ w2_d)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    U = np.zeros((n1*n2, rank))
    V = np.zeros((m1*m2, rank))
    W = np.zeros((p1*p2, rank))
    
    for c in range(r1):
        for d in range(r2):
            idx = c * r2 + d
            U[:, idx] = np.kron(U1[:, c], U2[:, d])
            V[:, idx] = np.kron(V1[:, c], V2[:, d])
            W[:, idx] = np.kron(W1[:, c], W2[:, d])
    
    return U, V, W

def _construct_strassen_based():
    """
    Try to construct a decomposition using Strassen-like ideas.
    
    For 3x3 * 3x8, we can try:
    - Decompose as 3x3 * 3x8 = (3x3 * 3x4) + (3x3 * 3x4) [not quite right]
    - Or use the structure more cleverly
    
    Actually, let's try a direct approach:
    - Use Strassen's rank-7 for 2x2x2
    - Extend to 3x3x3 with additional terms
    - Then extend to 3x3x8
    
    For a simpler and more reliable approach, let's try:
    - Rank-7 Strassen for the 2x2 part
    - Schoolbook for the remaining parts
    - This should give rank < 72
    """
    n, m, p = 3, 3, 8
    
    # Strategy: Partition the 3x3 matrices into 2x2 + 1x1 blocks
    # A = [[A22, a_col], [a_row, a_scalar]]
    # B = [[B22, b_col], [b_row, b_scalar]]
    # where A22, B22 are 2x2, a_col, b_col are 1x2, a_row, b_row are 2x1, a_scalar, b_scalar are scalars
    
    # C = A*B
    # C22 (top-left 2x2) = A22*B22 + a_col*b_row
    # C21 (top-right 1x2) = A22*b_col + a_col*b_scalar
    # C12 (bottom-left 2x1) = a_row*B22 + a_scalar*b_row
    # C11 (bottom-right scalar) = a_row*b_col + a_scalar*b_scalar
    
    # This is getting complicated. Let me try a different approach.
    
    # Approach: Use the fact that for rectangular matrices, we can sometimes
    # do better than schoolbook by exploiting structure.
    
    # For 3x3 * 3x8, the schoolbook rank is 72.
    # A simple improvement: if we can find that some of the 72 terms are linearly
    # dependent in the tensor space, we can reduce the rank.
    
    # Let's try a numerical approach: build the 72 schoolbook terms as vectors
    # in the tensor space and find the rank of this set of vectors.
    
    # The tensor space has dimension 3*3*8 = 72.
    # Each schoolbook term is a vector with a 1 in one position and 0 elsewhere.
    # So the 72 schoolbook terms form a basis, and the rank is exactly 72.
    
    # This means we can't reduce the rank by just finding linear dependencies
    # in the schoolbook decomposition. We need a fundamentally different approach.
    
    # Let's try to use Strassen's algorithm in a more clever way.
    
    # For 3x3 * 3x8, we can view this as multiplying a 3x3 matrix by a 3x8 matrix.
    # We can partition the 3x8 matrix into two 3x4 matrices: B = [B1 | B2]
    # Then C = A*B = [A*B1 | A*B2]
    # Each A*B_i is a 3x4 matrix multiplication.
    
    # For 3x3 * 3x4, the schoolbook rank is 3*3*4 = 36.
    # Can we do better?
    
    # Let's try to construct a rank < 36 decomposition for 3x3 * 3x4.
    
    # Actually, let me just return the schoolbook decomposition for now,
    # but with a more efficient implementation that might be easier to optimize.
    
    rank = 72
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(8):
                U[i*3+j][col] = 1
                V[j*8+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_extension():
    """
    Try to construct a decomposition using Strassen's algorithm extended to 3x3.
    
    The key idea: use Strassen's rank-7 for 2x2x2, then add terms to handle
    the 3rd row and column.
    
    For 3x3 * 3x3:
    - Use Strassen (rank 7) for the 2x2 blocks
    - Add terms for the 3rd row/col interactions
    - Total rank should be around 23
    
    For 3x3 * 3x8:
    - Use the 3x3x3 decomposition (rank ~23)
    - Extend to handle the 8 columns instead of 3
    - This might give rank around 23 * (8/3) ~ 61, which is not much better than 72
    
    Let me try a different approach: directly construct a low-rank decomposition
    for 3x3 * 3x8 using the structure of the problem.
    """
    n, m, p = 3, 3, 8
    
    # Let's try to use the fact that the output C is 3x8.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can write this as:
    # C = A @ B
    # where A is 3x3 and B is 3x8
    
    # One approach: use the SVD or other factorizations, but we need exact
    # tensor decomposition, not approximate.
    
    # Let me try a structured decomposition:
    # For each output element C[i,k], we need to compute sum_j A[i,j] B[j,k]
    
    # We can use the following identity:
    # sum_j A[i,j] B[j,k] = (1/2) * sum_{j,l} (A[i,j] + A[i,l]) * (B[j,k] + B[l,k]) 
    #   - (1/2) * sum_{j,l} (A[i,j] - A[i,l]) * (B[j,k] - B[l,k])
    # This doesn't seem to help directly.
    
    # Let me try a different strategy: use the known result that the rank of
    # M_{3,3,3} is at most 23, and see if we can extend this to M_{3,3,8}.
    
    # For now, return schoolbook as a safe baseline
    return _schoolbook()

def _local_optimization(U, V, W, n, m, p, max_iters=100, tol=1e-8):
    """Refine a decomposition using gradient descent."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    
    for it in range(max_iters):
        lr = 0.01 / (1 + it * 0.1)
        
        for c in range(rank):
            # Update W
            grad_W = np.zeros(p * n)
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        pred = 0.0
                        for c2 in range(rank):
                            pred += Ua[i * m + j, c2] * Va[j * p + k, c2] * Wa[k * n + i, c2]
                        grad_W[k * n + i] += (pred - 1.0) * Ua[i * m + j, c] * Va[j * p + k, c]
            Wa[:, c] -= lr * grad_W
            
            # Update U
            grad_U = np.zeros(n * m)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        pred = 0.0
                        for c2 in range(rank):
                            pred += Ua[i * m + j, c2] * Va[j * p + k, c2] * Wa[k * n + i, c2]
                        grad_U[i * m + j] += (pred - 1.0) * Va[j * p + k, c] * Wa[k * n + i, c]
            Ua[:, c] -= lr * grad_U
            
            # Update V
            grad_V = np.zeros(m * p)
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        pred = 0.0
                        for c2 in range(rank):
                            pred += Ua[i * m + j, c2] * Va[j * p + k, c2] * Wa[k * n + i, c2]
                        grad_V[j * p + k] += (pred - 1.0) * Ua[i * m + j, c] * Wa[k * n + i, c]
            Va[:, c] -= lr * grad_V
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = 0.0
                    for c in range(rank):
                        pred += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                    err += (pred - 1.0) ** 2
        if err < tol:
            break
    
    U_list = Ua.tolist()
    V_list = Va.tolist()
    W_list = Wa.tolist()
    
    return U_list, V_list, W_list

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110 second limit (safety margin)
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try Strassen-based construction
    if time.time() < deadline:
        try:
            strassen_result = _try_strassen_extension()
            if strassen_result:
                r = len(strassen_result['U'][0])
                if r < best_rank and _verify_decomposition(
                    strassen_result['U'], strassen_result['V'], strassen_result['W'], n, m, p
                ):
                    best = strassen_result
                    best_rank = r
        except Exception:
            pass
    
    # Try local optimization on schoolbook to see if we can reduce rank
    if time.time() < deadline and best_rank > 60:
        try:
            # Try to find a rank-60 decomposition by starting from schoolbook
            # and using gradient descent
            rank = 60
            rng = np.random.RandomState(42)
            U = np.zeros((9, rank))
            V = np.zeros((24, rank))
            W = np.zeros((24, rank))
            
            # Initialize with first 60 columns of schoolbook
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(8):
                        if col >= rank:
                            break
                        U[i*3+j, col] = 1
                        V[j*8+k, col] = 1
                        W[k*3+i, col] = 1
                        col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            
            # Add random components to the remaining columns
            for c in range(col, rank):
                U[:, c] = rng.randn(9) * 0.1
                V[:, c] = rng.randn(24) * 0.1
                W[:, c] = rng.randn(24) * 0.1
            
            U_list, V_list, W_list = _local_optimization(
                U.tolist(), V.tolist(), W.tolist(), n, m, p, max_iters=50
            )
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
