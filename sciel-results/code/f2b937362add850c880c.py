"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np

def construct():
    """Find rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook decomposition
    rank_school = n * m * p  # 90
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Verify a decomposition is exact
    def verify(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Try to find lower-rank decompositions using random search
    # Fix U and V randomly, solve for W, check exactness
    
    np.random.seed(123)
    
    def to_fraction(x):
        """Convert a float to a Fraction with denominator 2 (half-integer)."""
        x2 = round(x * 2)
        return Fraction(int(x2), 2)
    
    def try_rank(r, max_trials=100):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Random U and V with half-integer entries
            U_rand = np.zeros((n*m, r), dtype=float)
            V_rand = np.zeros((m*p, r), dtype=float)
            
            # Use small half-integers
            for _ in range(30):
                U_rand = np.random.randint(-4, 5, size=(n*m, r)) * 0.5
                V_rand = np.random.randint(-4, 5, size=(m*p, r)) * 0.5
                
                # Solve for W
                W_sol = np.zeros((p*n, r), dtype=float)
                feasible = True
                
                for i in range(n):
                    for k in range(p):
                        # Build m x r system for W[k*n+i, :]
                        A_sub = np.zeros((m, r), dtype=float)
                        for j in range(m):
                            for r_idx in range(r):
                                A_sub[j, r_idx] = U_rand[i*m+j, r_idx] * V_rand[j*p+k, r_idx]
                        
                        # Solve A_sub @ W_row = ones(m)
                        try:
                            W_row, _, _, _ = np.linalg.lstsq(A_sub, np.ones(m), rcond=None)
                            # Check exactness
                            if np.allclose(A_sub @ W_row, np.ones(m), atol=1e-10):
                                W_sol[k*n+i, :] = W_row
                            else:
                                feasible = False
                                break
                        except:
                            feasible = False
                            break
                    
                    if not feasible:
                        break
                
                if not feasible:
                    continue
                
                # Convert to exact fractions
                U_frac = [[to_fraction(U_rand[i, r]) for r in range(r)] for i in range(n*m)]
                V_frac = [[to_fraction(V_rand[i, r]) for r in range(r)] for i in range(m*p)]
                W_frac = [[to_fraction(W_sol[i, r]) for r in range(r)] for i in range(p*n)]
                
                # Verify with exact arithmetic
                if verify(U_frac, V_frac, W_frac, n, m, p):
                    return U_frac, V_frac, W_frac
        
        return None
    
    # Try ranks from 89 down to 70
    for r in range(best_rank - 1, 70, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_trials=50)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
