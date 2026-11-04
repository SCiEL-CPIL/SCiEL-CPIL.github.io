"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
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

def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should be 1 if (i,j,k) is a valid triple, else 0.
    # Actually, the tensor for matmul A(n,m) * B(m,p) -> C(n,p) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    
    # Efficient check using numpy
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can compute this as a batched operation.
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, -1)  # (n, m, rank)
    V_reshaped = V.reshape(m, p, -1)  # (m, p, rank)
    W_reshaped = W.reshape(p, n, -1)  # (p, n, rank)
    
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # This is a sum over c of a product of three arrays.
    # We can compute this as:
    # For each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Use einsum: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But W is indexed by (k, i, c), so we need W_reshaped[k,i,c]
    # Let's use np.einsum with the appropriate indices.
    
    # Actually, let's do it more carefully.
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # We can compute this by:
    # 1. For each c, compute the outer product U_c[i,j] * V_c[j,k] * W_c[k,i]
    # 2. Sum over c
    
    # But this is expensive. Let's use einsum.
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Let's try: np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    # Wait, W_reshaped has shape (p, n, rank) = (k, i, c), so W_reshaped[k,i,c] is correct.
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    
    # T should be all ones for valid (i,j,k)
    # For matrix multiplication, all (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid.
    # So T should be all ones.
    if T.shape != (n, m, p):
        return False
    
    return np.allclose(T, 1.0, atol=tol, rtol=tol)

def _to_integer_or_half(U, V, W, tol=1e-6):
    """Convert entries to integer or half-integer if possible."""
    def snap(arr):
        # Check if entries are close to integers or half-integers
        # Multiply by 2 and check if close to integers
        arr2 = 2.0 * arr
        rounded = np.round(arr2)
        if np.allclose(arr2, rounded, atol=tol):
            return (rounded / 2.0).astype(float)
        return arr
    
    U = snap(U)
    V = snap(V)
    W = snap(W)
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = n * m * p
    
    # Try to reduce rank using random initialization and gradient-free search
    # Strategy: Try various rank values and see if we can find a valid decomposition
    
    # For small matrices, we can try to find decompositions with rank < n*m*p
    # Known bounds: For 2x2x2, rank is 7 (Strassen). For other sizes, bounds are less clear.
    # Let's try to find a decomposition with rank significantly below schoolbook.
    
    # Approach: Use a random initialization and try to optimize
    # But since we need exact decomposition, let's try a different approach.
    
    # For the specific case of 4x5 * 5x6, let's try to construct a decomposition
    # using known algebraic identities or by solving a system of equations.
    
    # Actually, let's try a simpler approach: 
    # 1. Start with schoolbook
    # 2. Try to merge terms or find linear dependencies
    # 3. Use SVD or other linear algebra techniques to reduce rank
    
    # Let's try to find a low-rank decomposition by solving for U, V, W
    # given a target rank r.
    
    # For each target rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # This is a system of n*m*p equations in n*m*r + m*p*r + p*n*r unknowns.
    # For small r, this might be solvable.
    
    # Let's try ranks from low to high and see if we can find a solution.
    
    # For efficiency, let's try a few candidate ranks.
    # Known: For square matrices, the rank is at most n^3 and at least n^2.
    # For rectangular, the rank is at most n*m*p and at least max(n*m, m*p, p*n).
    
    # Let's try to find a decomposition with rank around max(n*m, m*p, p*n) = max(20, 30, 24) = 30.
    
    # Actually, let's just try to verify the schoolbook decomposition and return it.
    # If we have time, we can try to improve it.
    
    if time.time() > deadline - 5.0:
        U, V, W = _to_integer_or_half(best_U, best_V, best_W)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better decomposition using random search
    # This is a heuristic approach
    
    np.random.seed(42)
    
    # Try a few random initializations
    for trial in range(10):
        if time.time() > deadline - 10.0:
            break
        
        # Try rank = n*m*p - trial*5 (reduce rank gradually)
        target_rank = min(n * m * p - trial * 3, max(n*m, m*p, p*n) + 10)
        if target_rank < max(n*m, m*p, p*n):
            target_rank = max(n*m, m*p, p*n)
        
        # Random initialization
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Verify
        if _verify(U, V, W, tol=1e-4):
            U, V, W = _to_integer_or_half(U, V, W)
            if _verify(U, V, W, tol=1e-6):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                break
    
    # Convert to list and return
    U, V, W = _to_integer_or_half(best_U, best_V, best_W)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
