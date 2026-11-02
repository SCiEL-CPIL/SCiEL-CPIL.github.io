"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Precompute consistent triples
    consistent = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                consistent.append((i*m+j, j*p+k, k*n+i))
    consistent_set = set(consistent)
    
    # Build index arrays for vectorized computation
    cons_a = np.array([t[0] for t in consistent])
    cons_b = np.array([t[1] for t in consistent])
    cons_c = np.array([t[2] for t in consistent])
    
    incons = []
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a,b,c) not in consistent_set:
                    incons.append((a,b,c))
    incons_a = np.array([t[0] for t in incons])
    incons_b = np.array([t[1] for t in incons])
    incons_c = np.array([t[2] for t in incons])
    
    def compute_residual(U, V, W, r):
        """Vectorized residual computation."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For consistent: sum_c U[cons_a, c] * V[cons_b, c] * W[cons_c, c]
        # This is a batched operation
        
        # Reshape for vectorized computation
        # U[cons_a, :] has shape (len(cons_a), r)
        # V[cons_b, :] has shape (len(cons_b), r)
        # W[cons_c, :] has shape (len(cons_c), r)
        
        res_cons = np.sum(U[cons_a] * V[cons_b] * W[cons_c], axis=1) - 1.0
        res_incons = np.sum(U[incons_a] * V[incons_b] * W[incons_c], axis=1)
        
        return np.concatenate([res_cons, res_incons])
    
    def verify_exact(U, V, W, r):
        for (a,b,c) in consistent:
            s = Fraction(0)
            for col in range(r):
                s += Fraction(U[a][col]) * Fraction(V[b][col]) * Fraction(W[c][col])
            if s != 1:
                return False
        for (a,b,c) in incons:
            s = Fraction(0)
            for col in range(r):
                s += Fraction(U[a][col]) * Fraction(V[b][col]) * Fraction(W[c][col])
            if s != 0:
                return False
        return True
    
    def try_solve_rank(r, max_attempts=3):
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                return None
            
            np.random.seed(attempt * 7919 + r * 104729)
            
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            from scipy.optimize import least_squares
            
            def obj(params):
                U_f = params[:n*m*r].reshape(n*m, r)
                V_f = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_f = params[n*m*r+m*p*r:].reshape(p*n, r)
                return compute_residual(U_f, V_f, W_f, r)
            
            try:
                result = least_squares(obj, params, max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if result.cost < 1e-12:
                    U_sol = result.x[:n*m*r].reshape(n*m, r)
                    V_sol = result.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W_sol = result.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Check if entries are close to rational with small denominator
                    def to_rational(val, max_denom=4):
                        best = None
                        best_err = float('inf')
                        for d in range(1, max_denom+1):
                            num = round(val * d)
                            err = abs(val - num/d)
                            if err < best_err:
                                best_err = err
                                best = (num, d)
                        if best_err < 1e-8:
                            return best
                        return None
                    
                    U_r = [[0]*r for _ in range(n*m)]
                    V_r = [[0]*r for _ in range(m*p)]
                    W_r = [[0]*r for _ in range(p*n)]
                    
                    ok = True
                    for idx in range(n*m):
                        for c in range(r):
                            rr = to_rational(U_sol[idx,c])
                            if rr is None:
                                ok = False
                                break
                            U_r[idx][c] = rr[0]/rr[1]
                        if not ok:
                            break
                    if not ok:
                        continue
                    
                    for idx in range(m*p):
                        for c in range(r):
                            rr = to_rational(V_sol[idx,c])
                            if rr is None:
                                ok = False
                                break
                            V_r[idx][c] = rr[0]/rr[1]
                        if not ok:
                            break
                    if not ok:
                        continue
                    
                    for idx in range(p*n):
                        for c in range(r):
                            rr = to_rational(W_sol[idx,c])
                            if rr is None:
                                ok = False
                                break
                            W_r[idx][c] = rr[0]/rr[1]
                        if not ok:
                            break
                    if not ok:
                        continue
                    
                    # Exact verification
                    for (a,b,c) in consistent:
                        s = Fraction(0)
                        for col in range(r):
                            s += Fraction(U_r[a][col]) * Fraction(V_r[b][col]) * Fraction(W_r[c][col])
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    for (a,b,c) in incons:
                        s = Fraction(0)
                        for col in range(r):
                            s += Fraction(U_r[a][col]) * Fraction(V_r[b][col]) * Fraction(W_r[c][col])
                        if s != 0:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    U_out = [[int(x) if x == int(x) else x for x in row] for row in U_r]
                    V_out = [[int(x) if x == int(x) else x for x in row] for row in V_r]
                    W_out = [[int(x) if x == int(x) else x for x in row] for row in W_r]
                    return {"U": U_out, "V": V_out, "W": W_out}
            except Exception:
                pass
        
        return None
    
    # Try ranks from low to high
    for r in range(15, min(60, best_rank)):
        if time.time() > deadline - 30:
            break
        
        result = try_solve_rank(r, max_attempts=2)
        if result is not None:
            best_rank = r
            best_result = result
            continue
    
    return best_result
# EVOLVE-BLOCK-END
