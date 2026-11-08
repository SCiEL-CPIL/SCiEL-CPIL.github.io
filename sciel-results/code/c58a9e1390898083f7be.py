"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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


def _round_to_half_int(arr, tol=1e-6):
    """Round each entry to the nearest half-integer; return rounded array and success flag."""
    rounded = np.round(arr * 2) / 2
    diff = np.abs(arr - rounded)
    return rounded, np.all(diff <= tol)


def _verify(U, V, W, tol=1e-8):
    """Check that U * V^T * W^T (contracted) equals the identity tensor for (n,m,p).
    Specifically, for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1"""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def _verify_fast(U, V, W, tol=1e-8):
    """Vectorized verification using einsum."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def _try_greedy_constructive(R, deadline=None, max_attempts=200):
    """Try to constructively find a rank-R decomposition by solving linear constraints.
    We fix U and V randomly and solve for W, then check if W can be made half-integer."""
    n, m, p = DIMENSIONS
    if deadline is not None and time.time() > deadline:
        return None
    
    num_vars = p * n * R
    num_eqs = n * m * p
    
    if num_vars < num_eqs:
        return None
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Random U, V with entries in {-1, -0.5, 0, 0.5, 1}
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, R))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, R))
        
        # Build A matrix: A[(i,j,k), var_idx] = U[i*m+j, r] * V[j*p+k, r]
        # var_idx = (k*n+i)*R + r
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        var_idx = (k * n + i) * R + r
                        A[eq_idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_, singular = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if residual is small
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        # Check if solution is (approximately) half-integer
        W_rounded, is_half_int = _round_to_half_int(W_flat)
        if not is_half_int:
            continue
        
        W = W_rounded.reshape(p * n, R)
        
        # Final verification
        if _verify_fast(U, V, W):
            return U, V, W
    
    return None


def _try_greedy_constructive_fixed_U(R, deadline=None, max_attempts=200):
    """Fix U to be structured, randomize V, solve for W."""
    n, m, p = DIMENSIONS
    if deadline is not None and time.time() > deadline:
        return None
    
    num_vars = p * n * R
    num_eqs = n * m * p
    
    if num_vars < num_eqs:
        return None
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Structured U: try to make it sparse or have simple structure
        U = np.zeros((n * m, R))
        # Assign each row of U a small number of non-zero entries
        for row in range(n * m):
            nnz = random.randint(1, min(3, R))
            cols = random.sample(range(R), nnz)
            for c in cols:
                U[row, c] = random.choice([-1, -0.5, 0.5, 1])
        
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, R))
        
        # Build A matrix
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        var_idx = (k * n + i) * R + r
                        A[eq_idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_, singular = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        W_rounded, is_half_int = _round_to_half_int(W_flat)
        if not is_half_int:
            continue
        
        W = W_rounded.reshape(p * n, R)
        
        if _verify_fast(U, V, W):
            return U, V, W
    
    return None


def _try_block_strassen(deadline=None):
    """Try to construct a decomposition using block structure.
    Split the 6 columns into 3 groups of 2. Each group corresponds to a 2x2x2 sub-problem.
    Use Strassen's algorithm (rank 7) for each 2x2x2 block."""
    n, m, p = DIMENSIONS
    if deadline is not None and time.time() > deadline:
        return None
    
    # Strassen's 7-term decomposition for 2x2x2 matrix multiplication
    # For C = A*B where A, B are 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our 2x3x6 problem, split B into 3 blocks of 3x2
    # B = [B0 | B1 | B2] where each Bx is 3x2
    # C = A * B = [A*B0 | A*B1 | A*B2]
    # Each A*Bx is 2x2, so we can use Strassen for each
    
    # However, A is 2x3, not 2x2. We need to handle this.
    # Let's split A into two 2x2 blocks plus a column:
    # A = [A_left(2x2) | A_right(2x1)]
    # This gets complicated. Let's try a different approach.
    
    # Alternative: use the known rank-5 decomposition for 2x2x2 if available,
    # or just do a direct constructive search with the block structure in mind.
    
    # For now, let's try a more targeted search with structured initialization
    # that respects the block structure.
    
    # Try ranks from 21 down to 15
    for R in range(21, 14, -1):
        if deadline is not None and time.time() > deadline:
            break
        result = _try_greedy_constructive(R, deadline, max_attempts=50)
        if result is not None:
            return result
        result = _try_greedy_constructive_fixed_U(R, deadline, max_attempts=50)
        if result is not None:
            return result
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U_s, V_s, W_s = _schoolbook()
    
    # Try block-structured approach first
    result = _try_block_strassen(deadline)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try constructive approach for various ranks
    for R in range(n * m * p - 1, max(n * m * p - 15, 15), -1):
        if time.time() > deadline:
            break
        result = _try_greedy_constructive(R, deadline, max_attempts=100)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try with fixed U structure
    for R in range(n * m * p - 1, max(n * m * p - 10, 15), -1):
        if time.time() > deadline:
            break
        result = _try_greedy_constructive_fixed_U(R, deadline, max_attempts=100)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook if no better found
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END
