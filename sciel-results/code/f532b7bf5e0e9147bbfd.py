"""Rewritten program: direct algebraic rank search for 3x4 * 4x6 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Return U, V, W as numpy arrays for schoolbook decomposition, shape (rows, rank)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    # The tensor T_{i,j,k} should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each (i,j,k), the product of the three entries should sum to 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy broadcasting."""
    # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
    # For each c, the outer product structure:
    # U_c is a vector of length n*m, V_c of length m*p, W_c of length p*n
    # The contribution to T[i,j,k] from component c is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We can check by reconstructing the full tensor and comparing.
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # For fixed c, the contribution to T[i,j,k] is uc[i,j]*vc[j,k]*wc[k,i]
        # This is not a simple outer product; we need to iterate or use einsum-like logic
        # Actually: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # Let's just do it directly for verification
        pass
    # Direct check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_attempts=50):
    """
    Try to reduce the rank from current to target_rank by finding linear dependencies.
    Uses iterative peeling: at each step, try to express one column as a combination
    of others and eliminate it.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    deadline = time.time() + 100  # Safety margin
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        if U.shape[1] <= target_rank:
            break
        
        # Try to find a column that is linearly dependent on others
        # We look for a column c such that there exist coefficients alpha_d for d != c
        # where for all (i,j,k): sum_d alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        # This is a system of n*m*p equations in (current_rank-1) unknowns.
        # Let's pick a column to eliminate and solve.
        
        eliminated = False
        # Try eliminating each column
        for c_elim in range(U.shape[1]):
            if time.time() > deadline:
                break
            
            # Build the system: for each (i,j,k), 
            # sum_{d != c_elim} alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = U[i*m+j,c_elim]*V[j*p+k,c_elim]*W[k*n+i,c_elim]
            
            # Number of unknowns: current_rank - 1
            # Number of equations: n*m*p
            n_eq = n * m * p
            n_unk = U.shape[1] - 1
            
            # Only solvable if n_unk <= n_eq (which is usually true)
            # Build A matrix (n_eq x n_unk) and b vector (n_eq,)
            A = np.zeros((n_eq, n_unk), dtype=float)
            b = np.zeros(n_eq, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b[eq_idx] = U[i*m+j, c_elim] * V[j*p+k, c_elim] * W[k*n+i, c_elim]
                        unk_idx = 0
                        for d in range(U.shape[1]):
                            if d == c_elim:
                                continue
                            A[eq_idx, unk_idx] = U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d]
                            unk_idx += 1
                        eq_idx += 1
            
            # Solve the linear system (least squares if overdetermined)
            try:
                alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if the solution actually satisfies the equations
            if np.max(np.abs(A @ alpha - b)) > 1e-8:
                continue
            
            # If valid, eliminate column c_elim by subtracting the combination
            for d in range(U.shape[1]):
                if d == c_elim:
                    continue
                # Find the index of d in the alpha vector
                # d maps to position: number of columns before d that are not c_elim
                pos = d if d < c_elim else d - 1
                if abs(alpha[pos]) < 1e-15:
                    continue
                U[:, d] -= alpha[pos] * U[:, c_elim]
                V[:, d] -= alpha[pos] * V[:, c_elim]
                W[:, d] -= alpha[pos] * W[:, c_elim]
            
            # Remove column c_elim
            mask = [d for d in range(U.shape[1]) if d != c_elim]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            eliminated = True
            break
        
        if not eliminated:
            break
    
    return U, V, W

def _greedy_rank_reduction(U, V, W, n, m, p, time_limit=100):
    """
    Aggressive rank reduction: repeatedly try to find any linear dependency
    among the rank-1 components and eliminate it.
    """
    deadline = time.time() + time_limit
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    while U.shape[1] > 1 and time.time() < deadline:
        current_rank = U.shape[1]
        eliminated = False
        
        # Try each column as a candidate for elimination
        for c_elim in range(current_rank):
            if time.time() > deadline:
                return U, V, W
            
            # Check if column c_elim is in the span of the others in the tensor sense
            # Build the linear system
            n_eq = n * m * p
            n_unk = current_rank - 1
            
            if n_unk >= n_eq:
                # Underdetermined or square - use lstsq
                pass
            
            A = np.zeros((n_eq, n_unk), dtype=float)
            b = np.zeros(n_eq, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b[eq_idx] = U[i*m+j, c_elim] * V[j*p+k, c_elim] * W[k*n+i, c_elim]
                        unk_idx = 0
                        for d in range(current_rank):
                            if d == c_elim:
                                continue
                            A[eq_idx, unk_idx] = U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d]
                            unk_idx += 1
                        eq_idx += 1
            
            try:
                alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if np.max(np.abs(A @ alpha - b)) > 1e-8:
                continue
            
            # Eliminate
            for d in range(current_rank):
                if d == c_elim:
                    continue
                pos = d if d < c_elim else d - 1
                if abs(alpha[pos]) > 1e-15:
                    U[:, d] -= alpha[pos] * U[:, c_elim]
                    V[:, d] -= alpha[pos] * V[:, c_elim]
                    W[:, d] -= alpha[pos] * W[:, c_elim]
            
            mask = [d for d in range(current_rank) if d != c_elim]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            eliminated = True
            break
        
        if not eliminated:
            break
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Start with schoolbook decomposition
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify schoolbook
    assert _verify_fast(U, V, W, n, m, p), "Schoolbook verification failed"
    
    # Try greedy rank reduction
    U, V, W = _greedy_rank_reduction(U, V, W, n, m, p, time_limit=100)
    
    # Verify result
    if not _verify_fast(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                # Round to nearest half-integer if close
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-8:
                    r.append(rounded)
                else:
                    r.append(round(val, 10))
            result.append(r)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
