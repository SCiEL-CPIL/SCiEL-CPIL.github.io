"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    dim_u = n * m  # 4
    dim_v = m * p  # 12
    dim_w = p * n  # 12

    # Non-zero entries of T:
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_set.add((a, b, c))

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    expected = 1.0 if (a, b, c) in nonzero_set else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[a, r] * V[b, r] * W[c, r]
                    if abs(actual - expected) > tol:
                        return False
        return True

    def try_rank(target_rank, seed=42):
        """Try to find a rank decomposition of given rank."""
        np.random.seed(seed)
        
        # Structured initialization based on the tensor structure
        # For each k (0..5), we have 4 entries:
        # (0, k, 2k), (1, 6+k, 2k), (2, k, 2k+1), (3, 6+k, 2k+1)
        # Try to use 3 terms per k = 18 terms, plus 2 shared terms = 20 total
        
        U = np.zeros((dim_u, target_rank))
        V = np.zeros((dim_v, target_rank))
        W = np.zeros((dim_w, target_rank))
        
        if target_rank >= 24:
            # Schoolbook: one term per nonzero entry
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, col] = 1
                        V[b, col] = 1
                        W[c, col] = 1
                        col += 1
            return U, V, W, verify(U, V, W)
        
        # For rank 20: use structured ansatz
        # Try 3 terms per k-value (6*3=18) + 2 additional terms
        # Initialize with small random values for the structured part
        
        # First, try a specific algebraic construction
        # For each k, use 3 terms with the following pattern:
        # Term 1: handles (0,k,2k) and (2,k,2k+1) partially
        # Term 2: handles (1,6+k,2k) and (3,6+k,2k+1) partially  
        # Term 3: correction term
        
        for k in range(p):
            base = k * 3
            # Term base+0
            U[0, base] = 1.0
            U[2, base] = 1.0
            V[k, base] = 1.0
            W[2*k, base] = 1.0
            W[2*k+1, base] = 1.0
            
            # Term base+1
            U[1, base+1] = 1.0
            U[3, base+1] = 1.0
            V[6+k, base+1] = 1.0
            W[2*k, base+1] = 1.0
            W[2*k+1, base+1] = 1.0
            
            # Term base+2: correction
            U[0, base+2] = 1.0
            U[1, base+2] = 1.0
            U[2, base+2] = 1.0
            U[3, base+2] = 1.0
            V[k, base+2] = 1.0
            V[6+k, base+2] = 1.0
            W[2*k, base+2] = 1.0
            W[2*k+1, base+2] = 1.0
        
        # This gives rank 18, but likely incorrect. Let's try numerical refinement.
        # Use gradient-free optimization: minimize the error
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        for attempt in range(50):
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Add small random perturbation
            noise = 0.1 * (attempt + 1) / 50
            U_try += np.random.randn(dim_u, target_rank) * noise
            V_try += np.random.randn(dim_v, target_rank) * noise
            W_try += np.random.randn(dim_w, target_rank) * noise
            
            # Compute error
            err = 0.0
            for a in range(dim_u):
                for b in range(dim_v):
                    for c in range(dim_w):
                        expected = 1.0 if (a, b, c) in nonzero_set else 0.0
                        actual = 0.0
                        for r in range(target_rank):
                            actual += U_try[a, r] * V_try[b, r] * W_try[c, r]
                        err += (actual - expected) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            
            if best_err < 1e-12:
                break
        
        valid = verify(best_U, best_V, best_W)
        return best_U, best_V, best_W, valid

    # Try rank 20 first
    for target_rank in [20, 21, 22, 23, 24]:
        U, V, W, valid = try_rank(target_rank)
        if valid:
            best_rank = target_rank
            best_decomp = (U, V, W)
            break
    else:
        # Fallback: schoolbook rank 24
        rank = 24
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        best_decomp = (U, V, W)
    
    U, V, W = best_decomp
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    row_list.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row_list.append(round(val * 2) / 2)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
