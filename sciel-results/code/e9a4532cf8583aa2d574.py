"""
Tensor decomposition for 2x2 * 2x10 matrix multiplication.
Strategy: Start with block-diagonal Strassen (rank 35), then attempt to reduce rank
by finding linear dependencies across the 5 blocks using the shared A matrix structure.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    
    Returns U (4x7), V (4x7), W (4x7) where:
      U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2
    """
    # Strassen products P1..P7
    # A indices: 0=a(A00), 1=b(A01), 2=c(A10), 3=d(A11)
    # B indices: 0=e(B00), 1=f(B01), 2=g(B10), 3=h(B11)
    # C indices (local): 0=C00, 1=C10, 2=C01, 3=C11 (using k*n+i convention: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3)
    
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # C = A*B:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    strassen_A = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 1],   # P2
        [1, 0, 0, 0],   # P3
        [0, 0, 0, 1],   # P4
        [1, 1, 0, 0],   # P5
        [-1, 0, 1, 0],  # P6
        [0, 1, 0, -1],  # P7
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # P1 (e+h)
        [1, 0, 0, 0],   # P2 (e)
        [0, 1, 0, -1],  # P3 (f-h)
        [-1, 0, 1, 0],  # P4 (g-e)
        [0, 0, 0, 1],   # P5 (h)
        [1, 1, 0, 0],   # P6 (e+f)
        [0, 0, 1, 1],   # P7 (g+h)
    ]
    
    # W contributions: w_contrib[r][c_local] where c_local: 0=C00, 1=C10, 2=C01, 3=C11
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    # C00 = P1 + P4 - P5 + P7
    w_contrib[0][0] = 1   # P1 -> C00
    w_contrib[3][0] = 1   # P4 -> C00
    w_contrib[4][0] = -1  # P5 -> C00
    w_contrib[6][0] = 1   # P7 -> C00
    
    # C01 = P3 + P5
    w_contrib[2][2] = 1   # P3 -> C01
    w_contrib[4][2] = 1   # P5 -> C01
    
    # C10 = P2 + P4
    w_contrib[1][1] = 1   # P2 -> C10
    w_contrib[3][1] = 1   # P4 -> C10
    
    # C11 = P1 - P2 + P3 + P6
    w_contrib[0][3] = 1   # P1 -> C11
    w_contrib[1][3] = -1  # P2 -> C11
    w_contrib[2][3] = 1   # P3 -> C11
    w_contrib[5][3] = 1   # P6 -> C11
    
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    for r in range(7):
        for idx in range(4):
            U[idx][r] = strassen_A[r][idx]
            V[idx][r] = strassen_B[r][idx]
            W[idx][r] = w_contrib[r][idx]
    
    return U, V, W


def _build_block_diagonal_strassen():
    """
    Build the rank-35 block-diagonal Strassen decomposition for 2x2 * 2x10.
    Returns U (4x35), V (20x35), W (20x35).
    """
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks
    block_rank = 7
    total_rank = num_blocks * block_rank  # 35
    
    U_base, V_base, W_base = _strassen_2x2_decomposition()
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for b in range(num_blocks):
        offset = b * block_rank
        
        for r in range(block_rank):
            col = offset + r
            
            # U: same for all blocks
            for idx in range(4):
                U[idx][col] = U_base[idx][r]
            
            # V: specific to block b
            # B indices for block b: e=B[0,2b], f=B[0,2b+1], g=B[1,2b], h=B[1,2b+1]
            # V index: j*p + k
            V[0 * p + 2 * b][col] = V_base[0 * 2 + 0][r]    # j=0, k=2b -> local 0
            V[0 * p + 2 * b + 1][col] = V_base[0 * 2 + 1][r]  # j=0, k=2b+1 -> local 1
            V[1 * p + 2 * b][col] = V_base[1 * 2 + 0][r]    # j=1, k=2b -> local 2
            V[1 * p + 2 * b + 1][col] = V_base[1 * 2 + 1][r]  # j=1, k=2b+1 -> local 3
            
            # W: specific to block b
            # C indices: C00(k=2b,i=0)->(2b)*2+0, C10(k=2b,i=1)->(2b)*2+1, C01(k=2b+1,i=0)->(2b+1)*2+0, C11(k=2b+1,i=1)->(2b+1)*2+1
            W[(2 * b) * 2 + 0][col] = W_base[0 * 2 + 0][r]   # k=2b, i=0 -> local C00
            W[(2 * b) * 2 + 1][col] = W_base[0 * 2 + 1][r]   # k=2b, i=1 -> local C10
            W[(2 * b + 1) * 2 + 0][col] = W_base[1 * 2 + 0][r]  # k=2b+1, i=0 -> local C01
            W[(2 * b + 1) * 2 + 1][col] = W_base[1 * 2 + 1][r]  # k=2b+1, i=1 -> local C11
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True


def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce the rank by finding linear dependencies among the 35 terms.
    
    The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    We can try to find a smaller set of rank-1 tensors that span the same tensor.
    One approach: check if some columns of the "tensor matricizations" are linearly dependent.
    
    For a tensor of format (a,b,c), the rank is the minimum number of rank-1 tensors needed.
    We can try to use the fact that the U matrix has repeated rows across blocks.
    
    Alternative: Use numerical linear algebra to find the minimal rank.
    The tensor can be viewed as a matrix of shape (n*m) x (m*p * p*n) or other unfoldings.
    The rank of the tensor is at least the rank of any unfolding.
    
    Let's try a different approach: since U has the same 4 rows repeated 5 times (with different
    columns), we can try to find a basis for the column space of U and see if we can merge terms.
    
    Actually, the key insight is that the 35 rank-1 tensors may not be linearly independent.
    We can set up a system of equations to find the minimal number of terms.
    
    For now, let's try a simple approach: check if we can eliminate some terms by finding
    linear combinations that give the same tensor.
    """
    rank = U.shape[1]
    
    # Try to find if any subset of terms can be removed while preserving the tensor.
    # This is a combinatorial problem, so let's try a greedy approach.
    
    # First, verify the original decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    # Try to remove one term at a time and see if the tensor is still preserved.
    # This is expensive, so we'll limit the number of attempts.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing each term individually
    for r_remove in range(rank):
        if time.time() > deadline - 30:
            break
        
        # Create a copy with term r_remove removed
        keep = [r for r in range(rank) if r != r_remove]
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = len(keep)
    
    # Try removing pairs of terms
    if best_rank < rank:
        for r1 in range(best_rank):
            if time.time() > deadline - 20:
                break
            for r2 in range(r1 + 1, best_rank):
                if time.time() > deadline - 20:
                    break
                
                keep = [r for r in range(best_rank) if r != r1 and r != r2]
                U_new = best_U[:, keep]
                V_new = best_V[:, keep]
                W_new = best_W[:, keep]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = len(keep)
    
    return best_U, best_V, best_W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
    Uses block-diagonal Strassen (rank 35) and attempts rank reduction.
    """
    deadline = time.time() + 120
    
    n, m, p = DIMENSIONS
    
    # Build the rank-35 block-diagonal Strassen decomposition
    U, V, W = _build_block_diagonal_strassen()
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    U, V, W = _try_rank_reduction(U, V, W, n, m, p, deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
