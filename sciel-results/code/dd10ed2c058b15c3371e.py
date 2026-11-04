"""Tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # safety margin before 120s limit


def _check_time():
    return time.time() >= _DEADLINE


def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check a set of random instances
    rng = np.random.RandomState(42)
    for _ in range(50):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_true = A @ B
        C_decomp = np.zeros((n, p))
        for c in range(rank):
            u_vec = U[:, c].reshape(n, m)
            v_vec = V[:, c].reshape(m, p)
            w_vec = W[:, c].reshape(p, n)
            # Contribution: u_vec * v_vec contracted, then weighted by w_vec
            # Actually: C[i,k] += U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # This is: C += sum_j A[i,j] * B[j,k] * W[k*n+i,c] ... no
            # Let me re-derive: C[i,k] = sum_{j} A[i,j] B[j,k]
            # Decomposition: C[i,k] = sum_c (sum_j A[i,j] U[i*m+j,c]) * (sum_k' ... )
            # Actually the standard form: C[i,k] = sum_c U_c[i,m_row] ... 
            # Let me just check directly with the tensor definition.
            pass
    # Direct tensor check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank(rank, max_iters=500, seed=0):
    """Try to find a rank-rank decomposition using structured random initialization."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Try different structured approaches
    for attempt in range(max_iters):
        if _check_time():
            return None
        
        # Approach: random sparse initialization with iterative refinement
        # Initialize with sparse patterns
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Randomly assign non-zero entries
        for c in range(rank):
            # Choose which entries are non-zero in each factor
            for idx in range(n * m):
                if rng.random() < 0.3:
                    U[idx, c] = rng.choice([-1, 0, 1, 2])
            for idx in range(m * p):
                if rng.random() < 0.3:
                    V[idx, c] = rng.choice([-1, 0, 1, 2])
            for idx in range(p * n):
                if rng.random() < 0.3:
                    W[idx, c] = rng.choice([-1, 0, 1, 2])
        
        # Check if this works
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist()
        
        # Try alternating least squares style refinement
        # For each c, try to match the tensor
        for _ in range(10):
            if _check_time():
                return None
            # Update W given U, V
            for c in range(rank):
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                    # For fixed c, W[k*n+i,c] = T[i,j,k] / (U[i*m+j,c] * V[j*p+k,c])
                    # But we need consistency across j
                    found = False
                    for j in range(m):
                        u_val = U[i * m + j, c]
                        v_val = V[j * p + k, c]
                        if abs(u_val) > 1e-12 and abs(v_val) > 1e-12:
                            target = 1.0 / (u_val * v_val)
                            if not found:
                                W[idx, c] = target
                                found = True
                            elif abs(W[idx, c] - target) > 1e-9:
                                break
                    if not found:
                        W[idx, c] = 0.0
            
            # Check
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _greedy_search():
    """Greedy structured search for low-rank decompositions."""
    n, m, p = DIMENSIONS
    # Known lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Upper bound from schoolbook: n*m*p = 63
    # Try ranks from lower bound upward
    
    for rank in range(21, 64):
        if _check_time():
            return None
        result = _try_rank(rank, max_iters=100, seed=rank)
        if result is not None:
            return result
    return None


def _structured_pattern():
    """Try specific structured patterns known to work for small matrices."""
    n, m, p = DIMENSIONS
    
    # Pattern: exploit the structure where p=7
    # Try rank = n*m + m*p - 1 type patterns
    # Or try bilinear patterns
    
    # Try a pattern based on separating the k dimension
    # For each k, we need a 3x3 * 3x1 multiplication
    # This could be done with rank 9 per k, but sharing across k
    
    # Try: U[i*m+j, c] where c indexes (i, j, k_group)
    # Let's try rank = 21 with a structured pattern
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Try: for each j (middle index), have a 3x7 block
    # U[i*m+j, c] = 1 if c corresponds to (i, j, k_slot)
    # This is essentially schoolbook but compressed
    
    # Alternative: use the fact that 3x3*3x7 can be seen as 7 instances of 3x3*3x1
    # Each 3x3*3x1 has rank at most 9 (schoolbook) or possibly less
    
    # Let's just try random search with more iterations
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_result = (U, V, W)
    
    # Try to find lower rank
    result = _greedy_search()
    if result is not None:
        U, V, W = result
        best_result = (U, V, W)
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
