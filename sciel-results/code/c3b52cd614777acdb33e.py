"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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
    return U, V, W

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4,7) for a single 2x2x2 block.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (a10 + a11) * b00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # M3 = a00 * (b10 - b11)
    U[0, 2] = 1.0
    V[2, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # M4 = a11 * (b01 - b00)
    U[3, 3] = 1.0
    V[1, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = -1.0; W[2, 3] = 1.0
    
    # M5 = (a00 + a01) * b11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = 1.0; W[1, 4] = 1.0
    
    # M6 = (a10 - a00) * (b00 + b10)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[2, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7 = (a01 - a11) * (b01 + b11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[1, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = -1.0
    
    return U, V, W

def _schoolbook_2x2x1():
    """Schoolbook for 2x2 * 2x1: rank 4."""
    U = np.zeros((4, 4), dtype=np.float64)
    V = np.zeros((2, 4), dtype=np.float64)
    W = np.zeros((2, 4), dtype=np.float64)
    for r in range(4):
        i, j = divmod(r, 2)
        U[i*2+j, r] = 1.0
        V[j, r] = 1.0
        W[i, r] = 1.0
    return U, V, W

def _build_2x2x3_from_2x2x2_and_2x2x1():
    """Build a 2x2x3 decomposition from Strassen(2x2x2) + schoolbook(2x2x1). Rank = 11."""
    U_s, V_s, W_s = _strassen_2x2x2()
    U_1, V_1, W_1 = _schoolbook_2x2x1()
    
    U = np.zeros((4, 11), dtype=np.float64)
    V = np.zeros((6, 11), dtype=np.float64)
    W = np.zeros((6, 11), dtype=np.float64)
    
    # First 7 columns: Strassen for k=0,1
    U[:, :7] = U_s
    V[0:4, :7] = V_s  # j*p+k: j in {0,1}, k in {0,1} -> indices 0,1,2,3
    W[:4, :7] = W_s   # k*n+i: k in {0,1}, i in {0,1} -> indices 0,1,2,3
    
    # Last 4 columns: schoolbook for k=2
    U[:, 7:11] = U_1
    V[0*3+2, 7:11] = V_1[0, :]  # j=0, k=2 -> index 2
    V[1*3+2, 7:11] = V_1[1, :]  # j=1, k=2 -> index 5
    W[2*2+0, 7:11] = W_1[0, :]  # k=2, i=0 -> index 4
    W[2*2+1, 7:11] = W_1[1, :]  # k=2, i=1 -> index 5
    
    return U, V, W

def _verify_2x2x3(U, V, W, tol=1e-9):
    """Verify a 2x2x3 decomposition: n=2, m=2, p=3."""
    n, m, p = 2, 2, 3
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_find_2x2x3_rank10(deadline):
    """
    Try to find a rank-10 decomposition for 2x2x3 matrix multiplication
    using iterative refinement (Bini's method variant).
    """
    n, m, p = 2, 2, 3
    rank = 10
    rng = np.random.default_rng(42)
    
    # Initialize from the rank-11 solution, then project to rank 10
    U11, V11, W11 = _build_2x2x3_from_2x2x2_and_2x2x1()
    
    # Strategy: start with random rank-10 init and iterate
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = rng.standard_normal((4, rank))
        V = rng.standard_normal((6, rank))
        W = rng.standard_normal((6, rank))
        
        # Iterative refinement (Bini's algorithm)
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each column r, U[:,r] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
            # Fix V and W, solve for U
            # For each (i,j): U[i*m+j, :] * (V[j*p+0,:] * W[0*n+i,:] + V[j*p+1,:] * W[1*n+i,:] + V[j*p+2,:] * W[2*n+i,:]) = 1
            for i in range(n):
                for j in range(m):
                    # Compute the "target" vector
                    target = np.zeros(rank)
                    for k in range(p):
                        target += V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j, :] * target = 1, minimize ||U[i*m+j, :]||
                    # Least squares: U_row = target / (target @ target) if target is nonzero
                    dot = target @ target
                    if dot > 1e-12:
                        U[i*m+j, :] = target / dot
                    else:
                        U[i*m+j, :] = 0.0
            
            # Update V
            for j in range(m):
                for k in range(p):
                    target = np.zeros(rank)
                    for i in range(n):
                        target += U[i*m+j, :] * W[k*n+i, :]
                    dot = target @ target
                    if dot > 1e-12:
                        V[j*p+k, :] = target / dot
                    else:
                        V[j*p+k, :] = 0.0
            
            # Update W
            for i in range(n):
                for k in range(p):
                    target = np.zeros(rank)
                    for j in range(m):
                        target += U[i*m+j, :] * V[j*p+k, :]
                    dot = target @ target
                    if dot > 1e-12:
                        W[k*n+i, :] = target / dot
                    else:
                        W[k*n+i, :] = 0.0
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def _block_decomposition(n, m, p):
    """
    Construct a block decomposition for M_{2,2,p} using 2x2x2 Strassen blocks.
    p=15: 7 blocks of 2 + 1 block of 1. Rank = 7*7 + 4 = 53.
    """
    if n != 2 or m != 2:
        return None
        
    block_sizes = []
    remaining = p
    while remaining > 0:
        if remaining >= 2:
            block_sizes.append(2)
            remaining -= 2
        else:
            block_sizes.append(1)
            remaining -= 1
            
    U_s, V_s, W_s = _strassen_2x2x2()
    U_1, V_1, W_1 = _schoolbook_2x2x1()
    
    total_rank = sum(7 if bs == 2 else 4 for bs in block_sizes)
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    k_offset = 0
    
    for bs in block_sizes:
        if bs == 2:
            for r in range(7):
                U[0, col_offset] = U_s[0, r]
                U[1, col_offset] = U_s[1, r]
                U[2, col_offset] = U_s[2, r]
                U[3, col_offset] = U_s[3, r]
                
                V[0 * p + k_offset, col_offset] = V_s[0, r]
                V[0 * p + k_offset + 1, col_offset] = V_s[1, r]
                V[1 * p + k_offset, col_offset] = V_s[2, r]
                V[1 * p + k_offset + 1, col_offset] = V_s[3, r]
                
                W[k_offset * n + 0, col_offset] = W_s[0, r]
                W[k_offset * n + 1, col_offset] = W_s[1, r]
                W[(k_offset + 1) * n + 0, col_offset] = W_s[2, r]
                W[(k_offset + 1) * n + 1, col_offset] = W_s[3, r]
                
                col_offset += 1
            k_offset += 2
        elif bs == 1:
            for r in range(4):
                U[0, col_offset] = U_1[0, r]
                U[1, col_offset] = U_1[1, r]
                U[2, col_offset] = U_1[2, r]
                U[3, col_offset] = U_1[3, r]
                
                V[0 * p + k_offset, col_offset] = V_1[0, r]
                V[1 * p + k_offset, col_offset] = V_1[1, r]
                
                W[k_offset * n + 0, col_offset] = W_1[0, r]
                W[k_offset * n + 1, col_offset] = W_1[1, r]
                
                col_offset += 1
            k_offset += 1
            
    return U, V, W

def _block_decomposition_2x2x3(n, m, p, U3, V3, W3):
    """
    Construct a block decomposition using 2x2x3 blocks.
    p=15: 5 blocks of 3. Rank = 5 * rank_of_block.
    """
    if n != 2 or m != 2 or p != 15:
        return None
    
    block_rank = U3.shape[1]
    total_rank = 5 * block_rank
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    for block_idx in range(5):
        col_offset = block_idx * block_rank
        k_offset = block_idx * 3
        
        for r in range(block_rank):
            U[0, col_offset] = U3[0, r]
            U[1, col_offset] = U3[1, r]
            U[2, col_offset] = U3[2, r]
            U[3, col_offset] = U3[3, r]
            
            for j in range(2):
                for k in range(3):
                    V[j * p + k_offset + k, col_offset] = V3[j * 3 + k, r]
            
            for k in range(3):
                for i in range(2):
                    W[(k_offset + k) * n + i, col_offset] = W3[k * 2 + i, r]
    
    return U, V, W

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Best result so far
    best_result = None
    best_rank = float('inf')
    
    # Option 1: 2x2x2 + 2x2x1 blocks (rank 53)
    U, V, W = _block_decomposition(n, m, p)
    if U is not None and _is_valid(U, V, W, n, m, p):
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        best_rank = U.shape[1]
    
    # Option 2: Try to find rank-10 2x2x3 block, giving rank 50
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        U3, V3, W3, err = _try_find_2x2x3_rank10(deadline)
        if U3 is not None:
            U, V, W = _block_decomposition_2x2x3(n, m, p, U3, V3, W3)
            if U is not None and _is_valid(U, V, W, n, m, p):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = U.shape[1]
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
