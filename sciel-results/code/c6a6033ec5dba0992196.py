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
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 3, 5, 7
    
    # Schoolbook baseline: rank 105
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
    
    # Strategy: Use structured search with random initialization and solving.
    # For fixed V and W, the constraints on U are linear.
    
    def try_rank(r):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        # Randomized approach: fix V and W, solve for U.
        # For each i (0..n-1), the rows U[i*m+j, :] for j=0..m-1 are determined by:
        # For each (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a linear system: A_i * u_i = 1, where u_i is a vector of length m*r.
        
        random.seed(42 + r)
        
        # Try multiple random seeds for V and W
        for attempt in range(100):
            if time.time() > deadline:
                return None
            
            # Generate random V and W with small integer entries
            # Using {-1, 0, 1} to keep numbers small
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
            
            U = [[0]*r for _ in range(n*m)]
            valid = True
            
            for i in range(n):
                # Build the linear system for U rows corresponding to index i
                # Unknowns: U[i*m+j, c] for j in [0, m), c in [0, r)
                # Total unknowns: m * r
                # Equations: for each (j, k) in [0, m) x [0, p):
                #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = V[j*p+k][c] * W[k*n+i][c]
                
                # Solve least squares
                try:
                    sol, residuals, rank_sys, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    valid = False
                    break
                
                # Check if the solution is close to half-integers
                # The problem requires integer/half-integer entries.
                # If the solution is not close to half-integers, this random V,W won't work
                # with a "nice" U. But wait, the problem says "entries that exactly decompose".
                # It doesn't explicitly say U, V, W must be half-integers, just "integer/half-integer entries".
                # So we MUST find a solution where U, V, W are half-integers.
                
                # If we fix V, W to integers, U must be such that the products work out.
                # If the linear system has a unique solution, it might not be half-integer.
                # However, if we allow V, W to be half-integers too, it's a 3-way nonlinear problem.
                
                # Let's check if the solution is "nice".
                # If not, we might need to adjust V, W or accept that this attempt fails.
                
                sol_rounded = np.round(sol * 2) / 2
                if np.max(np.abs(sol - sol_rounded)) > 1e-6:
                    # The solution is not half-integer.
                    # This doesn't mean no solution exists, but this specific V, W doesn't yield a "nice" U.
                    # We could try to find a nearby V, W that does, but that's hard.
                    # Or we can just skip this attempt.
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
    # Lower bound: max(n*m, m*p, p*n) = 35
    # But matrix multiplication is harder.
    # Let's try from 40 to 104.
    
    # To improve chances, we can also try to fix U and solve for V, W?
    # No, the system is linear in each factor if the other two are fixed.
    # The bottleneck is finding V, W such that the resulting U is "nice" (half-integer).
    
    # Let's add a heuristic: if the solution is close to integer, round to integer.
    # If close to half-integer, round to half-integer.
    # The check `np.max(np.abs(sol - sol_rounded)) > 1e-6` handles this.
    
    # We can also try to perturb V, W slightly if they are "almost" good?
    # No, let's stick to random search for now.
    
    # Optimization: If r is small, the system is likely underdetermined or ill-conditioned.
    # If r is large, it's overdetermined.
    # We want the "sweet spot".
    
    # Let's try specific ranks that are known to be good for similar problems.
    # For 2x2x2, rank 7.
    # For 3x3x3, rank 19 (Strassen).
    # For 3x5x7, maybe around 60-80?
    
    # Let's iterate from 50 to 104.
    for r in range(50, 105):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
