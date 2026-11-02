"""Matrix multiplication 3x5x7 - rank minimization via algebraic search."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - target) > 1e-12:
                    return False
    return True

def _verify_np(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    n, m, p = 3, 5, 7
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > 1e-10:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Use the bilinear structure to reduce the problem.
    # For fixed V and W, the constraints on U are linear.
    # We can try to find solutions with integer/half-integer entries.
    
    # Key insight: The multiplication tensor has a specific structure.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This means we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Approach: Try to find low-rank decompositions using random initialization
    # and solving linear systems.
    
    def try_rank(r, num_attempts=20):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        random.seed(42 + r)
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random V and W with small integer entries
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(p*n)]
            
            # For each i, solve for U[i*m+j, :] for j=0..m-1
            # The system: for each (j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            U = [[0]*r for _ in range(n*m)]
            
            valid = True
            for i in range(n):
                # Build the system for this i
                # Unknowns: U[i*m+j, c] for j in range(m), c in range(r) -> m*r unknowns
                # Equations: for j in range(m), k in range(p):
                #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = V[j*p+k][c] * W[k*n+i][c]
                
                # Solve least squares
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    valid = False
                    break
                
                # Check if solution is close to integer/half-integer
                sol_rounded = np.round(sol * 2) / 2
                if np.max(np.abs(sol - sol_rounded)) > 1e-6:
                    # Try to find nearby half-integers
                    valid = False
                    break
                
                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(sol_rounded[j*r + c])
            
            if not valid:
                continue
            
            # Verify the full solution
            if _verify_np(U, V, W, n, m, p):
                # Convert to exact fractions
                U_frac = [[Fraction(U[i][j]).limit_denominator(2) for j in range(r)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(2) for j in range(r)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(2) for j in range(r)] for i in range(p*n)]
                
                # Double check with fractions
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(r))
                            if val != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        return None
    
    # Try ranks from a reasonable lower bound
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But matrix multiplication requires more. Known: M(2,2,2)=7, M(3,3,3) >= 19
    # For 3x5x7, let's try from ~40 upward
    
    # Actually, let's be smarter. Try specific ranks.
    for r in range(35, 106):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
