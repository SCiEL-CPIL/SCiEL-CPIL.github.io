"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try to find a low-rank exact decomposition
    # The (2,3,3) tensor should have rank 12
    for target_rank in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, target_rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank n*m*p
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
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank):
    """Try to find an exact rank decomposition."""
    import numpy as np
    import time
    
    start = time.time()
    deadline = start + 130
    
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # This is a nonlinear system. Let's use a randomized approach:
    # Initialize randomly, then try to refine.
    
    # Actually, let's use a different approach: solve the polynomial system
    # by fixing some variables and solving linearly for others.
    
    # Strategy: Use the fact that if we fix U and V, then for each (i,j,k),
    # we have a linear equation in W. But W is shared across equations.
    
    # Better strategy: Use a tensor decomposition approach.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want T = sum_t u_t ⊗ v_t ⊗ w_t where u_t is n×m, v_t is m×p, w_t is p×n.
    
    # Let's try a random initialization and then use gradient descent or similar.
    # But we need EXACT solutions, so this is tricky.
    
    # Alternative: Use the fact that the system is polynomial and use a Groebner basis
    # or similar. But that's expensive.
    
    # Let me try a specific construction for rank 12.
    if rank == 12:
        result = _construct_rank12(n, m, p)
        if result is not None:
            return result
    
    # For other ranks, use schoolbook if rank >= n*m*p
    if rank >= n * m * p:
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
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_rank12(n, m, p):
    """Try to construct a rank-12 decomposition for (2,3,3)."""
    # This is based on known results that rank(2,3,3) = 12.
    # Let me try to find it numerically.
    import numpy as np
    import time
    
    start = time.time()
    
    # The tensor T has T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We need U (6x12), V (9x12), W (6x12) such that:
    # For all i,j,k: sum_t U[i*3+j,t] * V[j*3+k,t] * W[k*2+i,t] = 1
    
    # Let's use a random initialization and then try to solve.
    # Actually, let's use a structured approach.
    
    # I'll try to find a solution by solving the system of polynomial equations.
    # Use a simple approach: random init, then Newton's method or similar.
    
    np.random.seed(42)
    
    # Initialize with small random values
    U = np.random.randn(n*m, 12) * 0.1
    V = np.random.randn(m*p, 12) * 0.1
    W = np.random.randn(p*n, 12) * 0.1
    
    # Define the residual
    def residual(U, V, W):
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(12):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    res[i,j,k] = s - 1
        return res
    
    # Try gradient descent
    lr = 0.01
    for iteration in range(5000):
        if time.time() > start + 120:
            break
        r = residual(U, V, W)
        err = np.sum(r**2)
        if err < 1e-12:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(12):
                        dU[i*m+j, t] += 2*r[i,j,k] * V[j*p+k, t] * W[k*n+i, t]
                        dV[j*p+k, t] += 2*r[i,j,k] * U[i*m+j, t] * W[k*n+i, t]
                        dW[k*n+i, t] += 2*r[i,j,k] * U[i*m+j, t] * V[j*p+k, t]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we got close
    r = residual(U, V, W)
    if np.max(np.abs(r)) < 1e-6:
        # Try to round to rationals
        # Scale to get nicer numbers
        # Find a common scale
        max_val = max(np.max(np.abs(U)), np.max(np.abs(V)), np.max(np.abs(W)))
        if max_val > 0:
            U = U / max_val
            V = V / max_val
            W = W / max_val
        
        # Try to round to fractions with small denominators
        def to_fraction_array(arr, max_denom=100):
            from fractions import Fraction
            frac_arr = []
            for val in arr.flatten():
                # Try to find a good rational approximation
                best_frac = Fraction(val).limit_denominator(max_denom)
                frac_arr.append(best_frac)
            return frac_arr
        
        # Check if the rounded version works
        U_frac = to_fraction_array(U)
        V_frac = to_fraction_array(V)
        W_frac = to_fraction_array(W)
        
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for t in range(12):
                        s += U_frac[i*m*12 + j*12 + t] * V_frac[j*p*12 + k*12 + t] * W_frac[k*n*12 + i*12 + t]
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Convert to lists of integers or fractions
            U_list = [[int(f) if f.denominator == 1 else float(f) for f in U_frac[i*12:(i+1)*12]] for i in range(n*m)]
            V_list = [[int(f) if f.denominator == 1 else float(f) for f in V_frac[j*12:(j+1)*12]] for j in range(m*p)]
            W_list = [[int(f) if f.denominator == 1 else float(f) for f in W_frac[k*12:(k+1)*12]] for k in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
