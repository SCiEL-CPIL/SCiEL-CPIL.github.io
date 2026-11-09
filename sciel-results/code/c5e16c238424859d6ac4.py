import numpy as np
from fractions import Fraction
import time
import random
import math

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 115

    def make_schoolbook(rank=None):
        if rank is None:
            rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def solve_linear_system_exact(A, b):
        m = len(A)
        R = len(A[0]) if m > 0 else 0
        M = [A[i][:] + [b[i]] for i in range(m)]
        
        rank = 0
        pivot_cols = []
        for col in range(R):
            pivot_row = None
            for row in range(rank, m):
                if M[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            
            M[rank], M[pivot_row] = M[pivot_row], M[rank]
            pivot_cols.append(col)
            
            pv = M[rank][col]
            M[rank] = [x / pv for x in M[rank]]
            
            for row in range(m):
                if row != rank and M[row][col] != 0:
                    factor = M[row][col]
                    M[row] = [M[row][c] - factor * M[rank][c] for c in range(R + 1)]
            
            rank += 1
            if rank == m:
                break
        
        for row in range(rank, m):
            if M[row][R] != 0:
                return None
        
        x = [Fraction(0) for _ in range(R)]
        for i in range(rank):
            x[pivot_cols[i]] = M[i][R]
        return x

    def try_rank(R, max_iters=500):
        if time.time() > deadline:
            return None
        
        values = [0, 1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            U = [[random.choice(values) for _ in range(R)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(R)] for _ in range(m*p)]
            
            W = [[0] * R for _ in range(p*n)]
            valid = True
            max_denom = 1
            
            for i in range(n):
                for k in range(p):
                    A = [[U[i*m+j][r] * V[j*p+k][r] for r in range(R)] for j in range(m)]
                    b = [Fraction(1), Fraction(1), Fraction(1)]
                    
                    w_sol = solve_linear_system_exact(A, b)
                    if w_sol is None:
                        valid = False
                        break
                    
                    W[k*n+i] = w_sol
                    for val in w_sol:
                        if val.denominator > max_denom:
                            max_denom = val.denominator
                        if val.denominator > 2:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # If we get here, W has denominators <= 2.
            # U, V are integers (0 or 1).
            # Thus U, V, W are all half-integers.
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            
            if verify(U_out, V_out, W_out):
                return {"U": U_out, "V": V_out, "W": W_out}
        
        return None

    best = make_schoolbook()
    best_rank = n * m * p
    
    for R in range(18, best_rank):
        if time.time() > deadline:
            break
        result = try_rank(R, max_iters=500)
        if result is not None:
            best = result
            best_rank = R
            break
    
    return best
