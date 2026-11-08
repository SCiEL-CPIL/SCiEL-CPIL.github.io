"""Combined approach: structured low-rank ansatz + exact rational verification
for the 3x3 x 3x4 matrix-multiplication tensor."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def _check(U, V, W, n, m, p):
    """Verify sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != Fraction(1):
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, rng, deadline):
    """Attempt to reduce rank by 1 via random linear combinations.
    Returns (U2, V2, W2) with one fewer column if successful, else None."""
    r = len(U[0])
    if r <= 1:
        return None

    # Try random pairs and coefficients to zero out a column
    for _ in range(200):
        if time.time() > deadline:
            break
        a = rng.randrange(r)
        b = rng.randrange(r)
        if a == b:
            continue
        # Try to eliminate column a by adding beta * column b to it
        for _ in range(30):
            if time.time() > deadline:
                break
            beta = rng.choice([
                Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2),
                Fraction(2), Fraction(-2), Fraction(1, 3), Fraction(-1, 3),
                Fraction(1, 4), Fraction(-1, 4), Fraction(3), Fraction(-3),
                Fraction(1, 5), Fraction(-1, 5)
            ])
            tU = [row[:] for row in U]
            tV = [row[:] for row in V]
            tW = [row[:] for row in W]
            for i in range(n * m):
                tU[i][a] = tU[i][a] + beta * tU[i][b]
            for j in range(m * p):
                tV[j][a] = tV[j][a] + beta * tV[j][b]
            for k in range(p * n):
                tW[k][a] = tW[k][a] + beta * tW[k][b]
            # Check if column a is now zero in tensor sense
            zero = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = tU[i * m + j][a] * tV[j * p + k][a] * tW[k * n + i][a]
                        if val != Fraction(0):
                            zero = False
                            break
                    if not zero:
                        break
                if not zero:
                    break
            if zero:
                U2 = [row[:a] + row[a + 1:] for row in tU]
                V2 = [row[:a] + row[a + 1:] for row in tV]
                W2 = [row[:a] + row[a + 1:] for row in tW]
                if _check(U2, V2, W2, n, m, p):
                    return U2, V2, W2
    return None

def _structured_search(n, m, p, rng, deadline):
    """Search for low-rank decompositions using structured parameterization.
    Uses the fact that MM tensor has symmetries to guide the search."""
    # Try various rank values from low to high
    # Known bounds: rank(3,3,4) is between 14 and 21 (schoolbook is 36)
    # Let's try ranks from 12 upward
    
    for target_rank in range(12, 22):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each target rank
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random rational values
            U = [[Fraction(0)] * target_rank for _ in range(n * m)]
            V = [[Fraction(0)] * target_rank for _ in range(m * p)]
            W = [[Fraction(0)] * target_rank for _ in range(p * n)]
            
            # Seed with some structure: ensure each (i,j), (j,k), (k,i) 
            # has at least one nonzero entry per column group
            for c in range(target_rank):
                # Randomly assign which indices are active for this column
                active_i = set()
                active_j = set()
                active_k = set()
                
                for _ in range(3):  # Pick a few active indices per column
                    i = rng.randrange(n)
                    j = rng.randrange(m)
                    k = rng.randrange(p)
                    active_i.add(i)
                    active_j.add(j)
                    active_k.add(k)
                
                for i in active_i:
                    for j in active_j:
                        U[i * m + j][c] = Fraction(rng.choice([-1, 1, 2, -2, 3]))
                for j in active_j:
                    for k in active_k:
                        V[j * p + k][c] = Fraction(rng.choice([-1, 1, 2, -2, 3]))
                for k in active_k:
                    for i in active_i:
                        W[k * n + i][c] = Fraction(rng.choice([-1, 1, 2, -2, 3]))
            
            # Now try to refine: adjust entries to satisfy the tensor equation
            # Use iterative correction: for each (i,j,k), compute current sum,
            # and adjust a random column to fix the discrepancy
            for iteration in range(1000):
                if time.time() > deadline:
                    break
                
                # Check if we're already done
                if _check(U, V, W, n, m, p):
                    return U, V, W
                
                # Find a (i,j,k) where the sum is wrong
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(target_rank):
                                s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                            if s != Fraction(1):
                                errors.append((i, j, k, 1 - s))
                
                if not errors:
                    return U, V, W
                
                # Pick a random error to fix
                i, j, k, err = rng.choice(errors)
                
                # Try to fix by adjusting a random column
                c = rng.randrange(target_rank)
                
                # We need: U[i,j,c] * V[j,k,c] * W[k,i,c] to change by err
                # Try adjusting W[k,i,c] if U and V are nonzero
                u_val = U[i * m + j][c]
                v_val = V[j * p + k][c]
                w_val = W[k * n + i][c]
                
                if u_val != 0 and v_val != 0:
                    # Set W[k,i,c] such that the product changes by err
                    # Current contribution: u_val * v_val * w_val
                    # New contribution should be: u_val * v_val * w_val + err
                    # So new w_val = w_val + err / (u_val * v_val)
                    new_w = w_val + err / (u_val * v_val)
                    W[k * n + i][c] = new_w
                elif u_val != 0 and w_val != 0:
                    new_v = v_val + err / (u_val * w_val)
                    V[j * p + k][c] = new_v
                elif v_val != 0 and w_val != 0:
                    new_u = u_val + err / (v_val * w_val)
                    U[i * m + j][c] = new_u
                else:
                    # At least two are zero; try setting one to 1 and adjusting another
                    if u_val == 0 and v_val == 0:
                        U[i * m + j][c] = Fraction(1)
                        V[j * p + k][c] = Fraction(1)
                        W[k * n + i][c] = err
                    elif u_val == 0:
                        U[i * m + j][c] = Fraction(1)
                        V[j * p + k][c] = err / w_val
                    elif v_val == 0:
                        V[j * p + k][c] = Fraction(1)
                        U[i * m + j][c] = err / w_val
                    else:
                        # w_val == 0
                        W[k * n + i][c] = Fraction(1)
                        U[i * m + j][c] = err / v_val
            
            # Final check
            if _check(U, V, W, n, m, p):
                return U, V, W
    
    return None

def construct():
    start = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Start with schoolbook decomposition as fallback
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # First, try structured search for low-rank solutions
    result = _structured_search(n, m, p, rng, DEADLINE)
    if result is not None:
        best_U, best_V, best_W = result
        best_rank = len(best_U[0])
    
    # Then try to further reduce the rank via elimination
    if best_rank > 1:
        U_cur, V_cur, W_cur = best_U, best_V, best_W
        while len(U_cur[0]) > 1 and time.time() < DEADLINE:
            res = _try_rank_reduction(U_cur, V_cur, W_cur, n, m, p, rng, DEADLINE)
            if res is not None:
                U_cur, V_cur, W_cur = res
                if len(U_cur[0]) < best_rank:
                    best_rank = len(U_cur[0])
                    best_U, best_V, best_W = U_cur, V_cur, W_cur
            else:
                break
    
    # Convert to regular Python lists (integers or fractions)
    def _to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        new_row.append(val.numerator)
                    else:
                        new_row.append(val)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}

# EVOLVE-BLOCK-END
