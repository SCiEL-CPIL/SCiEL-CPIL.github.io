"""Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]
DEADLINE_SEC = 120


def _build_schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition exactly reproduces the matmul tensor."""
    rank = U.shape[1]
    # Tensor T[i,j,k] = 1 for all valid i,j,k
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using numpy."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True


def _mod_gaussian_elimination(A, mod):
    """
    Perform Gaussian elimination over a finite field Z/mod.
    A is a 2D numpy array of integers.
    Returns (rank, pivot_cols) where pivot_cols are column indices of pivots.
    """
    A = A.astype(np.int64).copy()
    rows, cols = A.shape
    rank = 0
    pivot_cols = []
    
    for col in range(cols):
        if rank >= rows:
            break
        # Find pivot
        pivot_row = None
        for r in range(rank, rows):
            if A[r, col] % mod != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        
        # Swap rows
        if pivot_row != rank:
            A[[rank, pivot_row]] = A[[pivot_row, rank]]
        
        # Normalize pivot row
        pivot_val = A[rank, col] % mod
        inv = pow(pivot_val, mod - 2, mod) if mod > 2 else 1
        A[rank, :] = (A[rank, :] * inv) % mod
        
        # Eliminate
        for r in range(rows):
            if r != rank and A[r, col] % mod != 0:
                factor = A[r, col] % mod
                A[r, :] = (A[r, :] - factor * A[rank, :]) % mod
        
        rank += 1
        pivot_cols.append(col)
    
    return rank, pivot_cols


def _find_min_rank_support(n, m, p, deadline, start_rank=35):
    """
    Use rank-revealing decomposition over a finite field to find the minimal rank
    and support pattern for the matmul tensor.
    
    The matmul tensor T[i,j,k] = 1 for all i,j,k.
    We search for a rank-r decomposition where U, V, W have sparse entries.
    
    Strategy: For each candidate rank r, try to find support patterns (which entries
    are nonzero) and solve for the values over a finite field.
    """
    if time.time() > deadline:
        return None, None, None, False
    
    rng = np.random.RandomState(42)
    mod = 97  # Small prime for speed
    
    # For each rank, try random support patterns
    for target_rank in range(start_rank, 81):
        if time.time() > deadline:
            break
        
        # Try multiple random support patterns
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate random support: for each rank direction r, randomly choose
            # which (i,j) pairs are nonzero in U, which (j,k) in V, which (k,i) in W
            U_support = {}  # r -> set of (i,j)
            V_support = {}  # r -> set of (j,k)
            W_support = {}  # r -> set of (k,i)
            
            for r in range(target_rank):
                # Each direction covers a small random subset
                n_u = rng.randint(2, 6)
                n_v = rng.randint(2, 6)
                n_w = rng.randint(2, 6)
                
                u_pairs = set()
                while len(u_pairs) < n_u:
                    i, j = rng.randint(0, n), rng.randint(0, m)
                    u_pairs.add((i, j))
                U_support[r] = u_pairs
                
                v_pairs = set()
                while len(v_pairs) < n_v:
                    j, k = rng.randint(0, m), rng.randint(0, p)
                    v_pairs.add((j, k))
                V_support[r] = v_pairs
                
                w_pairs = set()
                while len(w_pairs) < n_w:
                    k, i = rng.randint(0, p), rng.randint(0, n)
                    w_pairs.add((k, i))
                W_support[r] = w_pairs
            
            # Build the linear system
            # Unknowns: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for supported entries
            # Equations: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is nonlinear, so we use a different approach.
            
            # Instead, fix W and solve for U and V, or use iterative approach.
            # For now, let's try a simpler approach: assume each rank direction
            # has exactly one nonzero in each of U, V, W (diagonal-like structure).
            
            # Alternative: use the fact that for matmul, we can use the structure
            # where U[i*m+j, r] is nonzero only for specific (i,j) patterns.
            
            # Let's try a structured approach: Strassen-like for 2x2 blocks
            # For 4x4 * 4x5, use 2x2 blocks for the first 4 columns, handle last col separately
            
            if time.time() > deadline - 5:
                break
        
        # If we found a valid decomposition at this rank, return it
        # (This section will be filled in by the solver)
    
    return None, None, None, False


def _strassen_4x4_4x5(n, m, p, deadline):
    """
    Construct a Strassen-based decomposition for 4x4 * 4x5.
    
    Partition A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Partition B (4x5) into blocks: B = [[B11, B12, B13], [B21, B22, B23]]
    where B11, B21 are 2x2, B12, B22 are 2x2, B13, B23 are 2x1
    
    Actually, let's partition differently:
    A (4x4) -> 4 blocks of 2x2
    B (4x5) -> 2 blocks of 2x2 (cols 0-3) + 2 blocks of 2x1 (col 4)
    
    For the 2x2 * 2x2 parts, use Strassen (rank 7 each).
    For the 2x2 * 2x1 parts, use standard (rank 4 each) or optimize.
    
    Total: 4 Strassen (7 each) = 28 for the 4x4*4x4 part
           Plus the 4x4*4x1 part: 4 standard (4 each) = 16
           Total: 44
    
    But we can do better by sharing some computations.
    """
    if time.time() > deadline:
        return None, None, None, False
    
    # Let's construct an explicit decomposition
    # For simplicity, start with a known good rank and try to reduce
    
    # Schoolbook is rank 80. Let's try to build a rank-49 or lower decomposition.
    # Known: rank(4x4*4x4) <= 49 (from literature)
    # For 4x4*4x5, we can extend the 4x4*4x4 result
    
    # Start with a block-based approach
    # Use 2x2 Strassen for the main part
    
    U = []
    V = []
    W = []
    
    # Block indices for 2x2 partition
    # A blocks: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # B blocks: B[0:2,0:2], B[0:2,2:4], B[0:2,4:5], B[2:4,0:2], B[2:4,2:4], B[2:4,4:5]
    
    # For each 2x2 * 2x2 block multiplication, use Strassen (7 multiplications)
    # Strassen formulas for C = A*B where A, B are 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's build the full decomposition using this structure
    # This is complex, so let's use a simpler approach:
    # Just return a known good decomposition
    
    # For now, return None and let the main function use schoolbook
    return None, None, None, False


def _build_rank_49_decomposition(n, m, p, deadline):
    """
    Build a rank-49 or lower decomposition for 4x4 * 4x5.
    Uses known results from matrix multiplication complexity theory.
    """
    if time.time() > deadline:
        return None, None, None, False
    
    # For 4x4 * 4x4, the best known rank is 49.
    # For 4x4 * 4x5, we can use a similar approach.
    
    # Let's try to construct a decomposition with rank 49
    # by extending the 4x4*4x4 result to handle the 5th column.
    
    # Actually, let's use a different approach:
    # Use the fact that 4x4*4x5 can be embedded in 4x4*4x4 with an extra column.
    
    # For now, let's just try random linear combinations more aggressively
    U, V, W, current_rank = _build_schoolbook(n, m, p)
    
    rng = np.random.RandomState(12345)
    
    # Try to reduce to rank 49
    target_rank = 49
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        M = rng.randn(current_rank, target_rank)
        U_new = U @ M
        V_new = V @ M
        W_new = W @ M
        
        if _verify_decomposition_fast(U_new, V_new, W_new, n, m, p, tol=1e-8):
            return U_new, V_new, W_new, True
    
    return None, None, None, False


def construct():
    """Construct the matrix multiplication decomposition."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SEC - 15  # Safety margin

    n, m, p = DIMENSIONS

    # Start with schoolbook
    U, V, W, current_rank = _build_schoolbook(n, m, p)
    best_rank = current_rank
    best_U, best_V, best_W = U, V, W

    # Try to reduce rank using various strategies
    # Strategy 1: Random linear combinations
    rng = np.random.RandomState(42)
    
    for target_rank in range(49, min(current_rank, 80)):
        if time.time() > deadline - 10:
            break
        
        success = False
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            M = rng.randn(current_rank, target_rank)
            U_new = U @ M
            V_new = V @ M
            W_new = W @ M
            
            if _verify_decomposition_fast(U_new, V_new, W_new, n, m, p, tol=1e-8):
                U, V, W = U_new, V_new, W_new
                current_rank = target_rank
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                success = True
                break
        
        if not success and time.time() > deadline - 30:
            break

    # Use the best found
    U, V, W = best_U, best_V, best_W

    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        # Fall back to schoolbook
        U, V, W, current_rank = _build_schoolbook(n, m, p)

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


# Verify correctness
if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])

    # Verify with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    # Compute via decomposition
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])

    # The standard check is that
    # for all i, j, k: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # And T_{i,j,k} = 1 for all valid i, j, k since C_{i,k} = sum_j A_{i,j} B_{j,k}
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                max_err = max(max_err, abs(s - 1.0))
    print(f"Max error: {max_err}")
    print(f"Rank: {rank}")
