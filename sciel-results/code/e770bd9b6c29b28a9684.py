"""Matrix multiplication 2x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact for matrix multiplication."""
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    # This is the condition for the tensor to represent matrix multiplication
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _block_decompose(n, m, p, split_m=None):
    """
    Decompose M_{n,m,p} by splitting the middle dimension.
    If m = m1 + m2, then we can combine decompositions of M_{n,m1,p} and M_{n,m2,p}.
    """
    if split_m is None:
        # Try all splits
        best_rank = n * m * p
        best_UVW = None
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = _block_decompose(n, m1, p)
            U2, V2, W2 = _block_decompose(n, m2, p)
            r1, r2 = U1.shape[1], U2.shape[1]
            total_rank = r1 + r2
            if total_rank < best_rank:
                best_rank = total_rank
                # Combine: U has blocks [U1; U2] but with different row structures
                # Actually, for M_{n,m,p}, U has shape (n*m, rank)
                # When splitting m into m1+m2:
                # U[i*m+j, c] where j < m1 uses U1, j >= m1 uses U2
                U = np.zeros((n*m, total_rank))
                V = np.zeros((m*p, total_rank))
                W = np.zeros((p*n, total_rank))
                
                # Fill U: for i in [0,n), j in [0,m1): use U1[i*m1+j, :]
                # for j in [m1, m): use U2[i*m2+(j-m1), :]
                for i in range(n):
                    for j in range(m1):
                        U[i*m+j, :r1] = U1[i*m1+j, :]
                    for j in range(m1, m):
                        U[i*m+j, r1:] = U2[i*m2+(j-m1), :]
                
                # Fill V: for j in [0,m1): use V1[j*p+k, :]
                # for j in [m1, m): use V2[(j-m1)*p+k, :]
                for j in range(m1):
                    for k in range(p):
                        V[j*p+k, :r1] = V1[j*p+k, :]
                for j in range(m1, m):
                    for k in range(p):
                        V[j*p+k, r1:] = V2[(j-m1)*p+k, :]
                
                # W is the same for both: W[k*n+i, c]
                # But we need to combine them. Actually W should be identical structure.
                # W1 has shape (p*n, r1), W2 has shape (p*n, r2)
                # We can just concatenate: W[:, :r1] = W1, W[:, r1:] = W2
                W[:, :r1] = W1
                W[:, r1:] = W2
                
                best_UVW = (U, V, W)
        if best_UVW is not None:
            return best_UVW
        return _schoolbook(n, m, p)
    else:
        # Base case or specific split
        if m == 1:
            return _schoolbook(n, m, p)
        m1 = split_m
        m2 = m - m1
        U1, V1, W1 = _block_decompose(n, m1, p)
        U2, V2, W2 = _block_decompose(n, m2, p)
        r1, r2 = U1.shape[1], U2.shape[1]
        total_rank = r1 + r2
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        for i in range(n):
            for j in range(m1):
                U[i*m+j, :r1] = U1[i*m1+j, :]
            for j in range(m1, m):
                U[i*m+j, r1:] = U2[i*m2+(j-m1), :]
        
        for j in range(m1):
            for k in range(p):
                V[j*p+k, :r1] = V1[j*p+k, :]
        for j in range(m1, m):
            for k in range(p):
                V[j*p+k, r1:] = V2[(j-m1)*p+k, :]
        
        W[:, :r1] = W1
        W[:, r1:] = W2
        
        return U, V, W

def _try_optimize(U, V, W, n, m, p, max_iter=100, tol=1e-6):
    """
    Try to reduce the rank by merging columns.
    We look for linear dependencies among the rank-1 tensors.
    """
    rank = U.shape[1]
    best_rank = rank
    best_UVW = (U.copy(), V.copy(), W.copy())
    
    # Try to remove columns one by one by checking if the tensor can still be represented
    # without that column, by adjusting other columns.
    # This is a heuristic approach.
    
    for col_to_remove in range(rank):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        # Try removing column col_to_remove
        # We need to check if the remaining columns can still span the tensor
        # This is complex, so we use a simpler heuristic:
        # Check if the column is "redundant" by seeing if its contribution
        # can be absorbed by other columns.
        
        # For now, just try a simple approach: if two columns are similar, merge them.
        # This is a placeholder for a more sophisticated method.
        pass
    
    return best_UVW

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' (list-of-lists) and rank = number of columns.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_UVW = (U_sb, V_sb, W_sb)
    
    # Try block decomposition
    U_bd, V_bd, W_bd = _block_decompose(n, m, p)
    if U_bd.shape[1] < best_rank:
        best_rank = U_bd.shape[1]
        best_UVW = (U_bd, V_bd, W_bd)
    
    # Verify the best decomposition
    if not _verify(best_UVW[0], best_UVW[1], best_UVW[2], n, m, p):
        # Fallback to schoolbook if verification fails
        best_UVW = (U_sb, V_sb, W_sb)
        best_rank = n * m * p
    
    # Try optimization
    if time.time() - _START_TIME < _DEADLINE - 10:
        U_opt, V_opt, W_opt = _try_optimize(best_UVW[0], best_UVW[1], best_UVW[2], n, m, p)
        if U_opt.shape[1] < best_rank and _verify(U_opt, V_opt, W_opt, n, m, p):
            best_rank = U_opt.shape[1]
            best_UVW = (U_opt, V_opt, W_opt)
    
    return {
        "U": best_UVW[0].tolist(),
        "V": best_UVW[1].tolist(),
        "W": best_UVW[2].tolist()
    }
# EVOLVE-BLOCK-END
