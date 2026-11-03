"""Tensor rank search for 3x3 * 3x6 matrix multiplication.
W index convention: W[k*n+i, r]
"""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 6]
DEADLINE = 120
SAFETY_MARGIN = 10


def _validate(U, V, W, n, m, p, rank):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    rng = random.Random(42)
    
    # Schoolbook baseline
    rank_sch = n * m * p  # 54
    U_sch = [[0] * rank_sch for _ in range(n * m)]
    V_sch = [[0] * rank_sch for _ in range(m * p)]
    W_sch = [[0] * rank_sch for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j][col] = 1
                V_sch[j*p+k][col] = 1
                W_sch[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    best_rank = rank_sch
    
    def time_up():
        return time.time() - start > DEADLINE - SAFETY_MARGIN
    
    # Try ranks from lower bound upward
    # Lower bound = max(n*m, m*p, p*n) = 18
    for target_rank in range(18, min(rank_sch, 40)):
        if time_up():
            break
        
        max_trials = 5000
        for trial in range(max_trials):
            if time_up():
                break
            
            U = [[rng.randint(0, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[rng.randint(0, 2) for _ in range(target_rank)] for _ in range(m*p)]
            
            # Solve for W: for each (i,k), m equations in rank unknowns
            W = [[0] * target_rank for _ in range(p*n)]
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    # A is m x rank, A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # A * W_vec = [1, 1, ..., 1]
                    aug = []
                    for j in range(m):
                        row = []
                        for c in range(target_rank):
                            row.append(Fraction(U[i*m+j][c] * V[j*p+k][c]))
                        row.append(Fraction(1))
                        aug.append(row)
                    
                    # Gaussian elimination with full pivoting
                    n_eq = m
                    n_var = target_rank
                    pivot_cols = []
                    
                    for col in range(min(n_eq, n_var)):
                        # Find pivot
                        pivot_row = None
                        for row in range(col, n_eq):
                            if aug[row][col] != 0:
                                pivot_row = row
                                break
                        if pivot_row is None:
                            continue
                        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
                        pv = aug[col][col]
                        for cc in range(col, n_var + 1):
                            aug[col][cc] /= pv
                        for row in range(n_eq):
                            if row != col and aug[row][col] != 0:
                                factor = aug[row][col]
                                for cc in range(col, n_var + 1):
                                    aug[row][cc] -= factor * aug[col][cc]
                        pivot_cols.append(col)
                    
                    # Consistency check
                    ok = True
                    for row in range(n_eq):
                        if all(aug[row][cc] == 0 for cc in range(n_var)) and aug[row][n_var] != 0:
                            ok = False
                            break
                    if not ok:
                        feasible = False
                        break
                    
                    # Back-substitute, free vars = 0
                    W_vec = [Fraction(0)] * n_var
                    for idx in range(len(pivot_cols) - 1, -1, -1):
                        pc = pivot_cols[idx]
                        row = idx  # pivot at (idx, pc) after row swaps
                        val = aug[row][n_var]
                        for cc in range(pc + 1, n_var):
                            val -= aug[row][cc] * W_vec[cc]
                        W_vec[pc] = val
                    
                    W[k*n+i] = [int(w) if w.denominator == 1 else float(w) for w in W_vec]
            
            if feasible and _validate(U, V, W, n, m, p, target_rank):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
