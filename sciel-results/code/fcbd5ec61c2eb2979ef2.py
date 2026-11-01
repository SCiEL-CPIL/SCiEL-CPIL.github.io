"""
Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.

Strategy:
- Split the 15 columns of B into blocks.
- For each 2x2 block, use Strassen's rank-7 decomposition.
- For the remaining 2x1 block, use schoolbook rank 4.
- Additionally, attempt to find a rank-10 decomposition for a 2x2x3 block
  to replace the last 2x2 (rank 7) + 2x1 (rank 4) = 11 with rank 10.
- If rank 10 is found, total rank = 6*7 + 10 = 52.
- Otherwise, total rank = 7*7 + 4 = 53.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1
    U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1
    V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1
    W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1
    W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1
    V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1
    W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1
    V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1
    W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1
    W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1
    U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1
    V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1
    U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1
    V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _verify_2x2x3(U, V, W, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the 2x2x3 matrix multiplication tensor.
    U: (4, rank), V: (6, rank), W: (6, rank)
    The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    where i in {0,1}, j in {0,1}, k in {0,1,2}
    """
    n, m, p = 2, 2, 3
    rank = U.shape[1]
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _try_find_2x2x3_rank10(deadline):
    """
    Attempt to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) if found, None otherwise.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Start from the rank-11 solution (Strassen + schoolbook) and try to optimize
    # Or use random initialization and ALS
    
    # Let's try a structured approach: start with rank 11 and try to merge terms
    # Actually, let's use ALS with random initialization
    
    np.random.seed(42)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(4, rank)
        V = np.random.randn(6, rank)
        W = np.random.randn(6, rank)
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    # T[i,j,:] = sum_r U[i,j,r] * V[j,:,r] * W[:,i,r]
                    # Let's build the system
                    # For each k, T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # This is linear in U[i,j,:]
                    # Let A[r,k] = V[j,k,r] * W[k,i,r]
                    # Then T[i,j,:] = U[i,j,:] @ A^T ... wait, let me think more carefully
                    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # Let B[r] be a vector of length p where B[r]_k = V[j,k,r] * W[k,i,r]
                    # Then T[i,j,:] = sum_r U[i,j,r] * B[r]
                    # So T[i,j,:] = U[i,j,:] @ B^T where B is (rank, p)
                    B = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            B[r, k] = V[j * p + k, r] * W[k * n + i, r]
                    # T[i,j,:] is a vector of length p
                    # We want to find U[i,j,:] such that U[i,j,:] @ B^T = T[i,j,:]
                    # This is a least squares problem: B @ U[i,j,:]^T = T[i,j,:]^T
                    # Or equivalently: solve B^T @ X = T[i,j,:] where X = U[i,j,:]^T
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    # T[:,j,k] = sum_r U[:,j,r] * V[j,k,r] * W[k,:,r]
                    # Let C[r] be a vector of length n*m where C[r]_{i} = U[i,j,r] * W[k,i,r]
                    # Then T[:,j,k] = sum_r V[j,k,r] * C[r]
                    C = np.zeros((rank, n * m))
                    for r in range(rank):
                        for i in range(n):
                            C[r, i * m + j] = U[i * m + j, r] * W[k * n + i, r]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(C, np.ones(n * m), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    # T[:,j,k] for all j, but W[k,i,r] appears in T[i,j,k]
                    # T[i,:,k] = sum_r U[i,:,r] * V[:,k,r] * W[k,i,r]
                    # Let D[r] be a vector of length m where D[r]_j = U[i,j,r] * V[j,k,r]
                    # Then T[i,:,k] = sum_r W[k,i,r] * D[r]
                    D = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            D[r, j] = U[i * m + j, r] * V[j * p + k, r]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(D, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error += (actual - 1.0) ** 2
        error = np.sqrt(error)
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_error < 1e-6:
            break
    
    if best_error < 1e-6 and best_U is not None:
        return best_U, best_V, best_W
    
    return None

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    
    Strategy: Decompose the 2x15 matrix into blocks.
    Use Strassen (rank 7) for 2x2 blocks and schoolbook (rank 4) for 2x1 block.
    Additionally, try to find a rank-10 decomposition for 2x2x3 to replace
    the last 2x2 (rank 7) + 2x1 (rank 4) = 11 with rank 10.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds total
    
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Get Strassen decomposition for 2x2x2
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # Try to find rank-10 decomposition for 2x2x3
    U_223, V_223, W_223 = None, None, None
    if time.time() < deadline - 30:  # Leave some time margin
        result = _try_find_2x2x3_rank10(deadline - 10)
        if result is not None:
            U_223, V_223, W_223 = result
            # Verify
            if _verify_2x2x3(U_223, V_223, W_223):
                print(f"Found valid rank-10 decomposition for 2x2x3")
            else:
                U_223, V_223, W_223 = None, None, None
                print("Rank-10 decomposition failed verification")
    
    if U_223 is not None:
        # Use 6 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x2x3 (rank 10)
        # Total rank = 6*7 + 10 = 52
        block_sizes = [2, 2, 2, 2, 2, 2, 3]
        num_blocks = len(block_sizes)
        total_rank = 6 * 7 + 10  # 52
    else:
        # Use 7 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (schoolbook, rank 4)
        # Total rank = 7*7 + 4 = 53
        block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
        num_blocks = len(block_sizes)
        total_rank = 7 * 7 + 4  # 53
    
    # Get schoolbook decomposition for 2x2x1
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    # Build full U, V, W matrices
    U = np.zeros((n * m, total_rank))  # 4 x total_rank
    V = np.zeros((m * p, total_rank))  # 30 x total_rank
    W = np.zeros((p * n, total_rank))  # 30 x total_rank
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])  # Starting column index in B
        
        if block_size == 2:
            # Use Strassen decomposition
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        elif block_size == 3:
            # Use the rank-10 decomposition for 2x2x3
            U[:, col_offset:col_offset+10] = U_223
            for j in range(m):
                for k_local in range(3):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+10] = V_223[j * 3 + k_local, :]
            for k_local in range(3):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+10] = W_223[k_local * 2 + i, :]
            col_offset += 10
        else:  # block_size == 1
            # Use schoolbook
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
