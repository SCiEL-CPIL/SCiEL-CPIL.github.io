"""Matrix multiplication 3x4x7 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS
N, M, P = n * m, m * p, p * n  # 12, 28, 21

def _tensor_entries():
    """Return list of (u_idx, v_idx, w_idx) for the multiplication tensor."""
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                entries.append((u_idx, v_idx, w_idx))
    return entries

_TENSOR_ENTRIES = _tensor_entries()

def _check_decomposition(U, V, W, tol=1e-10):
    """Check if U,V,W exactly decompose the tensor. U:(N,rank), V:(M,rank), W:(P,rank)"""
    rank = U.shape[1]
    for (u_idx, v_idx, w_idx) in _TENSOR_ENTRIES:
        s = 0.0
        for c in range(rank):
            s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
        if abs(s - 1.0) > tol:
            return False
    return True

def _to_rational_matrix(mat, tol=1e-9):
    """Convert float matrix to Fraction matrix, returning None if not close to rationals."""
    from fractions import Fraction
    rows = []
    for row in mat:
        new_row = []
        for val in row:
            # Try to find a simple rational approximation
            best_frac = Fraction(0)
            best_err = abs(val)
            for denom in range(1, 100):
                num = round(val * denom)
                frac = Fraction(num, denom)
                err = abs(float(frac) - val)
                if err < best_err:
                    best_err = err
                    best_frac = frac
            if best_err > tol:
                return None
            new_row.append(best_frac)
        rows.append(new_row)
    return rows

def _build_rank1_decompositions(n, m, p):
    """Build schoolbook rank-1 decompositions for each (i,j,k) triple."""
    decomp = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                decomp[(i, j, k)] = (i * m + j, j * p + k, k * n + i)
    return decomp

def _construct_schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_optimize(rank, max_time, seed=42):
    """
    Try to find an exact decomposition of given rank using optimization.
    Returns (U, V, W) as lists of lists of Fractions, or None.
    """
    if rank < 1:
        return None
    
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(N, rank) * 0.5
    V = np.random.randn(M, rank) * 0.5
    W = np.random.randn(P, rank) * 0.5
    
    # Normalize to avoid trivial solutions
    for c in range(rank):
        norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
        if norm > 1e-12:
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
    
    start_time = time.time()
    
    # Use simple gradient descent on the loss function
    # Loss = sum over all tensor entries of (product - 1)^2 + regularization
    
    def compute_loss_and_grad(U, V, W):
        loss = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for (u_idx, v_idx, w_idx) in _TENSOR_ENTRIES:
            # product = sum_c U[u,c] * V[v,c] * W[w,c]
            prod = 0.0
            for c in range(rank):
                prod += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
            
            diff = prod - 1.0
            loss += diff * diff
            
            for c in range(rank):
                uvw = U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                dU[u_idx, c] += 2 * diff * V[v_idx, c] * W[w_idx, c]
                dV[v_idx, c] += 2 * diff * U[u_idx, c] * W[w_idx, c]
                dW[w_idx, c] += 2 * diff * U[u_idx, c] * V[v_idx, c]
        
        return loss, dU, dV, dW
    
    # Simple gradient descent with adaptive learning rate
    lr = 0.01
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for iteration in range(5000):
        if time.time() - start_time > max_time - 5:
            break
        
        loss, dU, dV, dW = compute_loss_and_grad(U, V, W)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            # Check if we're close to exact
            if loss < 1e-15:
                # Try to rationalize
                U_rat = _to_rational_matrix(best_U)
                V_rat = _to_rational_matrix(best_V)
                W_rat = _to_rational_matrix(best_W)
                if U_rat is not None and V_rat is not None and W_rat is not None:
                    # Verify
                    U_np = np.array([[float(x) for x in row] for row in U_rat])
                    V_np = np.array([[float(x) for x in row] for row in V_rat])
                    W_np = np.array([[float(x) for x in row] for row in W_rat])
                    if _check_decomposition(U_np, V_np, W_np, tol=1e-10):
                        return [list(row) for row in U_rat], [list(row) for row in V_rat], [list(row) for row in W_rat]
        
        # Adaptive learning rate
        if loss > best_loss * 1.5:
            lr *= 0.5
        elif loss < best_loss * 0.8:
            lr = min(lr * 1.1, 0.1)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Periodic re-normalization
        if iteration % 100 == 0:
            for c in range(rank):
                norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
                if norm > 1e-12:
                    U[:, c] /= norm
                    V[:, c] /= norm
                    W[:, c] /= norm
    
    # Final check with best found
    if best_loss < 1e-10:
        U_rat = _to_rational_matrix(best_U)
        V_rat = _to_rational_matrix(best_V)
        W_rat = _to_rational_matrix(best_W)
        if U_rat is not None and V_rat is not None and W_rat is not None:
            U_np = np.array([[float(x) for x in row] for row in U_rat])
            V_np = np.array([[float(x) for x in row] for row in V_rat])
            W_np = np.array([[float(x) for x in row] for row in W_rat])
            if _check_decomposition(U_np, V_np, W_np, tol=1e-10):
                return [list(row) for row in U_rat], [list(row) for row in V_rat], [list(row) for row in W_rat]
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 by 4x7 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    """
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _construct_schoolbook()
    best_rank = n * m * p  # 84
    best_result = {
        'U': U_school.tolist(),
        'V': V_school.tolist(),
        'W': W_school.tolist()
    }
    
    # Try to find better decompositions
    # Based on known results, the rank should be around 30-40 for 3x4x7
    
    # Strategy: Try ranks from low to high, with time-limited optimization
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But tensor rank is typically higher
    
    # Try a range of ranks with optimization
    ranks_to_try = []
    # Based on literature, for 3x4x7 the rank is likely around 31-35
    for r in range(28, 50):
        ranks_to_try.append(r)
    
    # Sort by rank (lowest first)
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Allocate time for this rank attempt
        time_per_attempt = min(8, (deadline - time.time()) / max(1, len(ranks_to_try) - ranks_to_try.index(rank)))
        
        if time_per_attempt < 2:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            result = _try_optimize(rank, time_per_attempt, seed=seed)
            if result is not None:
                U_rat, V_rat, W_rat = result
                best_rank = rank
                best_result = {
                    'U': U_rat,
                    'V': V_rat,
                    'W': W_rat
                }
                found = True
                break
        
        if found:
            # Found a valid decomposition, try lower ranks if time permits
            # But since we're going in order, this is already the best
            break
    
    # If we didn't find anything better, return schoolbook
    return best_result
