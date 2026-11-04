import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    deadline = time.time() + 140
    
    def build_schoolbook():
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
        return U, V, W, rank
    
    U, V, W, rank = build_schoolbook()
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    def to_frac(x):
        if isinstance(x, Fraction):
            return x
        if isinstance(x, int):
            return Fraction(x)
        if isinstance(x, float):
            # Try to detect half-integers
            if abs(x * 2 - round(x * 2)) < 1e-9:
                return Fraction(int(round(x * 2)), 2)
            elif abs(x - round(x)) < 1e-9:
                return Fraction(int(round(x)))
            else:
                return Fraction(x)
        return Fraction(x)
    
    def verify_exact(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += to_frac(U[i*m+j][c]) * to_frac(V[j*p+k][c]) * to_frac(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    def solve_for_W(U, V, target_rank):
        num_eqs = n * m * p
        num_unk = p * n * target_rank
        
        A = np.zeros((num_eqs, num_unk), dtype=float)
        b = np.ones(num_eqs, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for c in range(target_rank):
                        coeff = U[i*m+j][c] * V[j*p+k][c]
                        if coeff != 0:
                            unk_idx = (k*n+i)*target_rank + c
                            A[eq, unk_idx] = float(coeff)
        
        try:
            w_sol, residuals, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
        
        # Try rounding to integers
        w_int = np.round(w_sol)
        if np.max(np.abs(w_sol - w_int)) < 1e-6:
            w_use = w_int
        else:
            # Try half-integers
            w_half = np.round(w_sol * 2) / 2.0
            if np.max(np.abs(w_sol - w_half)) < 1e-6:
                w_use = w_half
            else:
                return None
        
        W_mat = [[0] * target_rank for _ in range(p*n)]
        for k in range(p):
            for i in range(n):
                for c in range(target_rank):
                    val = w_use[(k*n+i)*target_rank + c]
                    if abs(val - round(val)) < 1e-9:
                        W_mat[k*n+i][c] = int(round(val))
                    elif abs(val*2 - round(val*2)) < 1e-9:
                        W_mat[k*n+i][c] = int(round(val*2)) / 2
                    else:
                        W_mat[k*n+i][c] = float(val)
        
        return W_mat
    
    random.seed(42)
    for target_rank in range(10, best_rank):
        if time.time() > deadline:
            break
        max_trials = 1000
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            
            W_rand = solve_for_W(U_rand, V_rand, target_rank)
            if W_rand is not None:
                if verify_exact(U_rand, V_rand, W_rand, target_rank):
                    best_rank = target_rank
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_rand
                    break
        if best_rank < rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
