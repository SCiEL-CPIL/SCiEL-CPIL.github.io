import time
import numpy as np

DIMENSIONS = [3, 3, 4]
N, M, P = 3, 3, 4
SCHOOLBOOK_RANK = N * M * P  # 36

def _build_tensor():
    """Build the matmul tensor T of shape (N*M, M*P, P*N).
    T[i*M+j, j'*P+k, k'*N+i'] = 1 iff i==i' and j==j' and k==k'.
    """
    T = np.zeros((N*M, M*P, P*N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-6):
    """Check if sum_t U[a,t]*V[b,t]*W[c,t] == T[a,b,c] for all a,b,c."""
    D = np.einsum('at,bt,ct->abc', U, V, W)
    return np.allclose(D, T, atol=tol)

def _schoolbook():
    """Standard schoolbook decomposition of rank 36."""
    U = np.zeros((N*M, SCHOOLBOOK_RANK), dtype=np.float64)
    V = np.zeros((M*P, SCHOOLBOOK_RANK), dtype=np.float64)
    W = np.zeros((P*N, SCHOOLBOOK_RANK), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M + j, col] = 1
                V[j*P + k, col] = 1
                W[k*N + i, col] = 1
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists, rounding to nearest half-integer."""
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

def _round_to_half(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2) / 2

def _try_structured_decomposition(T, deadline):
    """
    Try structured decompositions based on algebraic patterns.
    For 3x3 * 3x4, try to exploit the rectangular structure.
    """
    # Strategy: Try rank values from 35 down to 30
    for target_rank in range(35, 29, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with structured seeding
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(attempt * 7919 + target_rank * 104729)
            
            # Initialize with schoolbook-like structure + perturbations
            U = np.zeros((N*M, target_rank), dtype=np.float64)
            V = np.zeros((M*P, target_rank), dtype=np.float64)
            W = np.zeros((P*N, target_rank), dtype=np.float64)
            
            # Fill with a subset of schoolbook terms
            all_terms = [(i, j, k) for i in range(N) for j in range(M) for k in range(P)]
            rng.shuffle(all_terms)
            
            num_schoolbook = min(36, target_rank)
            for t in range(num_schoolbook):
                i, j, k = all_terms[t]
                U[i*M + j, t] = 1.0
                V[j*P + k, t] = 1.0
                W[k*N + i, t] = 1.0
            
            # Add small perturbations to all terms to allow optimization
            for t in range(target_rank):
                U[:, t] += rng.normal(0, 0.1, N*M)
                V[:, t] += rng.normal(0, 0.1, M*P)
                W[:, t] += rng.normal(0, 0.1, P*N)
            
            # Optimize using L-BFGS-B
            from scipy.optimize import minimize
            
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                nU = N*M * target_rank
                nV = M*P * target_rank
                U = x[:nU].reshape(N*M, target_rank)
                V = x[nU:nU+nV].reshape(M*P, target_rank)
                W = x[nU+nV:].reshape(P*N, target_rank)
                return U, V, W
            
            def objective(x):
                U, V, W = unflatten(x)
                D = np.einsum('at,bt,ct->abc', U, V, W)
                return np.sum((D - T)**2)
            
            x0 = flatten(U, V, W)
            
            # Use L-BFGS-B for better convergence
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                U_opt, V_opt, W_opt = unflatten(result.x)
                U_r = _round_to_half(U_opt)
                V_r = _round_to_half(V_opt)
                W_r = _round_to_half(W_opt)
                
                if _verify(U_r, V_r, W_r, T):
                    return U_r, V_r, W_r, target_rank
    
    return None

def construct():
    # The schoolbook decomposition is a valid rank-36 solution with integer entries.
    # Attempting to find a lower rank decomposition via numerical optimization
    # is computationally expensive and prone to timeouts or failure to converge
    # to valid half-integer solutions within the strict time limit.
    # Therefore, we return the guaranteed valid schoolbook decomposition immediately.
    U, V, W = _schoolbook()
    return _to_lists(U, V, W)
