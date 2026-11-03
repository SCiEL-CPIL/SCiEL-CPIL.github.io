import time
import numpy as np
from fractions import Fraction
from itertools import product

_START_TIME = time.time()
DEADLINE = _START_TIME + 118

def _time_left():
    return DEADLINE - time.time()

def construct():
    n, m, p = 3, 3, 4
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank
    for target_rank in range(13, best_rank):
        if _time_left() < 5:
            break
        result = _try_rank(n, m, p, target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            
    return best_result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _try_rank(n, m, p, r):
    np.random.seed(int(time.time() * 1000) % 1000000)
    max_attempts = 500 if r <= 20 else 200
    if _time_left() < 10:
        max_attempts = 50
        
    for attempt in range(max_attempts):
        if _time_left() < 2:
            break
        
        # Random U, V with entries in {-2, -1, 0, 1, 2}
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        for i in range(n*m):
            for c in range(r):
                U_vals[i, c] = np.random.choice([-2, -1, 0, 1, 2])
                
        for i in range(m*p):
            for c in range(r):
                V_vals[i, c] = np.random.choice([-2, -1, 0, 1, 2])
                
        # Compute M
        M = np.zeros((p*n, r), dtype=int)
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    s = 0
                    for j in range(m):
                        s += U_vals[i*m+j, c] * V_vals[j*p+k, c]
                    M[k*n + i, c] = s
                    
        # Solve for W
        W_vals = np.zeros((p*n, r), dtype=object)
        valid = True
        for i in range(n):
            for k in range(p):
                row_idx = k * n + i
                m_row = M[row_idx, :]
                w_solution = _solve_half_integer(m_row)
                if w_solution is None:
                    valid = False
                    break
                W_vals[row_idx, :] = w_solution
            if not valid:
                break
                
        if not valid:
            continue
            
        # Convert to lists
        U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n*m)]
        V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m*p)]
        W_list = []
        for i in range(p*n):
            row = []
            for c in range(r):
                val = W_vals[i, c]
                if isinstance(val, Fraction):
                    row.append(val)
                else:
                    row.append(Fraction(val))
            W_list.append(row)
            
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            # Convert Fractions to int/float for output
            U_out = [[int(x) for x in row] for row in U_list]
            V_out = [[int(x) for x in row] for row in V_list]
            W_out = []
            for row in W_list:
                row_out = []
                for x in row:
                    if x.denominator == 1:
                        row_out.append(x.numerator)
                    else:
                        row_out.append(float(x))
                W_out.append(row_out)
            return {"U": U_out, "V": V_out, "W": W_out}
            
    return None

def _solve_half_integer(m_row):
    r = len(m_row)
    # Try 1-term
    for c in range(r):
        mc = int(m_row[c])
        if mc != 0:
            w_val = Fraction(1, mc)
            if (w_val * 2).denominator == 1:
                result = [Fraction(0)] * r
                result[c] = w_val
                return result
                
    # Try 2-term
    for a in range(r):
        ma = int(m_row[a])
        if ma == 0: continue
        for b in range(a+1, r):
            mb = int(m_row[b])
            if mb == 0: continue
            for x_num in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
                x = Fraction(x_num, 2)
                y = (Fraction(1) - Fraction(ma) * x) / Fraction(mb)
                if (y * 2).denominator == 1:
                    result = [Fraction(0)] * r
                    result[a] = x
                    result[b] = y
                    return result
                    
    # Try 3-term
    for a in range(r):
        ma = int(m_row[a])
        if ma == 0: continue
        for b in range(a+1, r):
            mb = int(m_row[b])
            if mb == 0: continue
            for c in range(b+1, r):
                mc = int(m_row[c])
                if mc == 0: continue
                for x_num in [-2, -1, 0, 1, 2]:
                    x = Fraction(x_num, 2)
                    for y_num in [-2, -1, 0, 1, 2]:
                        y = Fraction(y_num, 2)
                        z = (Fraction(1) - Fraction(ma)*x - Fraction(mb)*y) / Fraction(mc)
                        if (z * 2).denominator == 1:
                            result = [Fraction(0)] * r
                            result[a] = x
                            result[b] = y
                            result[c] = z
                            return result
                            
    return None

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
