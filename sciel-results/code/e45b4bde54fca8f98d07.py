import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # A is n x m, B is m x p, C is n x p
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    
    all_abc = list(product(range(n*m), range(m*p), range(p*n)))
    
    def verify(U, V, W, rank):
        """Verify exact decomposition."""
        for abc in all_abc:
            a, b, c = abc
            val = 0
            for r in range(rank):
                val += U[a][r] * V[b][r] * W[c][r]
            expected = T.get(abc, 0)
            if val != expected:
                return False
        return True
    
    def solve_for_V(U, W, rank):
        """Given U and W (lists of Fractions), solve for V. Returns None if inconsistent."""
        V = []
        for b in range(m * p):
            equations = []
            rhs = []
            for a in range(n * m):
                for c in range(p * n):
                    row = []
                    for r in range(rank):
                        row.append(U[a][r] * W[c][r])
                    equations.append(row)
                    rhs.append(Fraction(T.get((a, b, c), 0)))
            
            # Gaussian elimination over rationals
            aug = [eq + [rhs[i]] for i, eq in enumerate(equations)]
            n_eq = len(aug)
            n_var = rank
            rank_actual = 0
            for col in range(n_var):
                pivot = -1
                for row in range(rank_actual, n_eq):
                    if aug[row][col] != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                aug[rank_actual], aug[pivot] = aug[pivot], aug[rank_actual]
                for row in range(n_eq):
                    if row != rank_actual and aug[row][col] != 0:
                        factor = aug[row][col] / aug[rank_actual][col]
                        for j in range(col, n_var + 1):
                            aug[row][j] -= factor * aug[rank_actual][j]
                rank_actual += 1
            
            for row in range(rank_actual, n_eq):
                if all(aug[row][j] == 0 for j in range(n_var)) and aug[row][n_var] != 0:
                    return None
            
            sol = [Fraction(0)] * n_var
            for row in range(rank_actual - 1, -1, -1):
                lead_col = -1
                for col in range(n_var):
                    if aug[row][col] != 0:
                        lead_col = col
                        break
                if lead_col == -1:
                    continue
                val = aug[row][n_var]
                for col in range(lead_col + 1, n_var):
                    val -= aug[row][col] * sol[col]
                sol[lead_col] = val / aug[row][lead_col]
            
            V.append(sol)
        
        return V
    
    # Try ranks from 19 upward
    for target_rank in [19, 20, 21, 22, 23, 24]:
        if time.time() > deadline - 20:
            break
        
        np.random.seed(42 + target_rank)
        
        for attempt in range(300):
            if time.time() > deadline - 10:
                break
            
            # Random integer initialization for U and W
            U = [[Fraction(np.random.randint(-3, 4)) for _ in range(target_rank)] for _ in range(n*m)]
            W = [[Fraction(np.random.randint(-3, 4)) for _ in range(target_rank)] for _ in range(p*n)]
            
            V = solve_for_V(U, W, target_rank)
            if V is None:
                continue
            
            if verify(U, V, W, target_rank):
                U_out = [[float(x) for x in row] for row in U]
                V_out = [[float(x) for x in row] for row in V]
                W_out = [[float(x) for x in row] for row in W]
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # Local search on U and W
            for _ in range(80):
                if time.time() > deadline - 5:
                    break
                
                mat = np.random.choice([0, 1])
                if mat == 0:
                    idx = (np.random.randint(n*m), np.random.randint(target_rank))
                    old = U[idx[0]][idx[1]]
                    U[idx[0]][idx[1]] = old + Fraction(np.random.choice([-1, 1]))
                else:
                    idx = (np.random.randint(p*n), np.random.randint(target_rank))
                    old = W[idx[0]][idx[1]]
                    W[idx[0]][idx[1]] = old + Fraction(np.random.choice([-1, 1]))
                
                V = solve_for_V(U, W, target_rank)
                if V is not None and verify(U, V, W, target_rank):
                    U_out = [[float(x) for x in row] for row in U]
                    V_out = [[float(x) for x in row] for row in V]
                    W_out = [[float(x) for x in row] for row in W]
                    return {"U": U_out, "V": V_out, "W": W_out}
                
                if mat == 0:
                    U[idx[0]][idx[1]] = old
                else:
                    W[idx[0]][idx[1]] = old
    
    # Fallback: naive rank-24 decomposition
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U[i * m + j][idx] = 1.0
        V[j * p + k][idx] = 1.0
        W[k * n + i][idx] = 1.0
        
    return {"U": U, "V": V, "W": W}
