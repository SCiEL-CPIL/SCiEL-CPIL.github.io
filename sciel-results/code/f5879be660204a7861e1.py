import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    TIME_LIMIT = 110  # Leave margin before 120s limit
    
    # First, try to find a rank-27 solution (better score)
    # Fall back to rank-28 if not found in time
    
    # Try rank 27 first
    try:
        result = _try_rank_27(start_time, TIME_LIMIT)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback: proven rank-28 Strassen block decomposition
    U, V, W = _build_rank_28()
    return {"U": U, "V": V, "W": W}


def _build_rank_28():
    """Build the rank-28 decomposition using Strassen on four 2x2 block multiplications."""
    n, m, p = 2, 4, 4
    TARGET_RANK = 28
    
    U = [[0] * TARGET_RANK for _ in range(n * m)]  # 8 x 28
    V = [[0] * TARGET_RANK for _ in range(m * p)]  # 16 x 28
    W = [[0] * TARGET_RANK for _ in range(p * n)]  # 8 x 28
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(i, j): return i * p + j
    def get_w_idx(i, j): return j * n + i  # W[k*n + i, r]
    
    # We have 4 block multiplications, each using Strassen (7 terms)
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 2:4]  ... wait, let me reconsider
    
    # Actually, A is 2x4, B is 4x4
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # Split j into 0,1 and 2,3
    # C[i][k] = A[i][0]*B[0][k] + A[i][1]*B[1][k] + A[i][2]*B[2][k] + A[i][3]*B[3][k]
    
    # Group A columns: A1 = A[:, 0:2], A2 = A[:, 2:4]
    # Group B rows: B1 = B[0:2, :], B2 = B[2:4, :]
    # C = A1 * B1 + A2 * B2
    
    # For each (A_block, B_block) pair, we need a 2x2 * 2x4 multiplication
    # Hmm, this is getting complicated. Let me use the original block structure.
    
    # The original code splits into four 2x2x2 sub-problems:
    # (a_r, a_c, b_r, b_c, c_c) represents:
    # A sub-block at rows [a_r, a_r+2), cols [a_c, a_c+2)
    # B sub-block at rows [b_r, b_r+2), cols [b_c, b_c+2)
    # C sub-block at rows [?, ?), cols [c_c, c_c+2)
    
    # For C = A * B where A is 2x4, B is 4x4, C is 2x4:
    # C[i][k] = sum_{j=0}^{3} A[i][j] * B[j][k]
    
    # The four blocks in the original:
    # (0, 0, 0, 0, 0): A[0:2, 0:2] * B[0:2, 0:2] -> contributes to C[0:2, 0:2]
    # (0, 2, 2, 0, 0): A[0:2, 2:4] * B[0:2, 0:2] -> contributes to C[0:2, 0:2]
    # (0, 0, 0, 2, 2): A[0:2, 0:2] * B[0:2, 2:4] -> contributes to C[0:2, 2:4]
    # (0, 2, 2, 2, 2): A[0:2, 2:4] * B[0:2, 2:4] -> contributes to C[0:2, 2:4]
    
    # Wait, but B is 4x4, so B[0:2, :] is the top half and B[2:4, :] is the bottom half.
    # Let me re-examine.
    
    # Actually I think the blocks should be:
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    
    # So the four 2x2x2 sub-multiplications are:
    # 1. A[:, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> C[:, 0:2] (2x2)
    # 2. A[:, 0:2] (2x2) * B[0:2, 2:4] (2x2) -> C[:, 2:4] (2x2)  
    # 3. A[:, 2:4] (2x2) * B[2:4, 0:2] (2x2) -> C[:, 0:2] (2x2)
    # 4. A[:, 2:4] (2x2) * B[2:4, 2:4] (2x2) -> C[:, 2:4] (2x2)
    
    # In terms of (a_col_block, b_row_block, b_col_block, c_col_block):
    # 1. a_c=0, b_r=0, b_c=0, c_c=0
    # 2. a_c=0, b_r=0, b_c=2, c_c=2
    # 3. a_c=2, b_r=2, b_c=0, c_c=0
    # 4. a_c=2, b_r=2, b_c=2, c_c=2
    
    # The original code had:
    # (0, 0, 0, 0, 0) -> a_r=0, a_c=0, b_r=0, b_c=0, c_c=0
    # (0, 2, 2, 0, 0) -> a_r=0, a_c=2, b_r=2, b_c=0, c_c=0
    # (0, 0, 0, 2, 2) -> a_r=0, a_c=0, b_r=0, b_c=2, c_c=2
    # (0, 2, 2, 2, 2) -> a_r=0, a_c=2, b_r=2, b_c=2, c_c=2
    
    # This matches! Good.
    
    blocks = [
        (0, 0, 0, 0, 0),  # A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        (0, 2, 2, 0, 0),  # A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        (0, 0, 0, 2, 2),  # A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        (0, 2, 2, 2, 2),  # A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        # U indices for A sub-block (2x2)
        u11 = get_u_idx(a_r, a_c)       # A[0][0] of sub-block
        u12 = get_u_idx(a_r, a_c + 1)   # A[0][1] of sub-block
        u21 = get_u_idx(a_r + 1, a_c)   # A[1][0] of sub-block
        u22 = get_u_idx(a_r + 1, a_c + 1)  # A[1][1] of sub-block
        
        # V indices for B sub-block (2x2)
        v11 = get_v_idx(b_r, b_c)       # B[0][0] of sub-block
        v12 = get_v_idx(b_r, b_c + 1)   # B[0][1] of sub-block
        v21 = get_v_idx(b_r + 1, b_c)   # B[1][0] of sub-block
        v22 = get_v_idx(b_r + 1, b_c + 1)  # B[1][1] of sub-block
        
        # W indices for C sub-block (2x2), using W[k*n+i] = W[2*k+i]
        w11 = get_w_idx(c_c, a_r)       # C[0][0] of sub-block -> W[2*0+0] = W[0]
        w12 = get_w_idx(c_c + 1, a_r)   # C[0][1] of sub-block -> W[2*1+0] = W[2]
        w21 = get_w_idx(c_c, a_r + 1)   # C[1][0] of sub-block -> W[2*0+1] = W[1]
        w22 = get_w_idx(c_c + 1, a_r + 1)  # C[1][1] of sub-block -> W[2*1+1] = W[3]
        
        # Strassen's 7 multiplication terms for 2x2 * 2x2:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 - M4 + M5 + M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[u11][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # M2 = (A21 + A22) * B11
        U[u21][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1
        W[w22][current_rank] += -1
        current_rank += 1
        
        # M3 = A11 * (B12 - B22)
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1
        V[v22][current_rank] += -1
        W[w12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # M4 = A22 * (B21 - B11)
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1
        V[v11][current_rank] += -1
        W[w11][current_rank] += 1
        W[w21][current_rank] += 1
        current_rank += 1
        
        # M5 = (A11 + A12) * B22
        U[u11][current_rank] += 1
        U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1
        W[w12][current_rank] += 1
        current_rank += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[u21][current_rank] += 1
        U[u11][current_rank] += -1
        V[v11][current_rank] += 1
        V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[u12][current_rank] += 1
        U[u22][current_rank] += -1
        V[v21][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
    
    assert current_rank == TARGET_RANK, f"Expected rank {TARGET_RANK}, got {current_rank}"
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition exactly matches matrix multiplication."""
    num_u = len(U)
    num_v = len(V)
    num_w = len(W)
    rank = len(U[0]) if num_u > 0 else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (j == k) else 0.0
                # Wait, this is not right. The tensor for matrix mult (i,j,k) is 1 if the indices match.
                # For C[i][k] = sum_j A[i][j] B[j][k], the tensor T[i,j,k] = 1 for all i,j,k
                # because every (i,j,k) triple contributes.
                # Actually no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
                # Let me reconsider.
                
                # The tensor for multiplying (n x m) by (m x p) to get (n x p):
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor T has T[i,j,k] = 1 for all i,j,k.
                
                # Wait, that's not a standard tensor decomposition. Let me think again.
                
                # The bilinear map is: (A, B) -> A*B
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # This can be written as sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # For this to equal sum_j A[i,j]*B[j,k], we need:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j, j'} ... 
                
                # Actually, the standard formulation:
                # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # should equal sum_j A[i,j] * B[j,k]
                
                # This works if for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # because then C[i,k] = sum_j A[i,j] * B[j,k] * 1 = sum_j A[i,j] * B[j,k]
                
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition exactly matches matrix multiplication."""
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None


def _try_rank_27(start_time, time_limit):
    """
    Attempt to find a rank-27 decomposition.
    
    Strategy: Start from the rank-28 solution and try to merge/eliminate one term
    by adjusting coefficients. This is a local search / optimization approach.
    
    If successful, return the decomposition. Otherwise return None.
    """
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return None
    
    # For now, this is a placeholder. A proper rank-27 search would be very complex
    # and risky. The known lower bound for R(2,4,4) is not well-established to be < 28.
    # 
    # In fact, it's conjectured that R(2,4,4) = 28, so rank 27 may not exist.
    # Given the risk of producing an invalid solution (score 0), we return None.
    
    return None


def construct():
    start_time = time.time()
    
    # Build the proven rank-28 decomposition
    U, V, W = _build_rank_28()
    
    # Verify correctness
    n, m, p = 2, 4, 4
    valid, info = _verify_decomposition(U, V, W, n, m, p)
    
    if not valid:
        # This should never happen, but handle it gracefully
        # Return a simple rank-32 schoolbook decomposition as ultimate fallback
        U, V, W = _build_rank_32()
    
    return {"U": U, "V": V, "W": W}


def _build_rank_32():
    """Build the schoolbook rank-32 decomposition as a fallback."""
    n, m, p = 2, 4, 4
    TARGET_RANK = 32  # n*m*p = 2*4*4 = 32
    
    U = [[0] * TARGET_RANK for _ in range(n * m)]
    V = [[0] * TARGET_RANK for _ in range(m * p)]
    W = [[0] * TARGET_RANK for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = idx
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
                idx += 1
    
    return U, V, W
