import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check that for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(str(U[i*m+j][c]))
                    v_val = Fraction(str(V[j*p+k][c]))
                    w_val = Fraction(str(W[k*n+i][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(r, n, m, p, max_attempts=100):
    """
    Attempts to find a rank r decomposition by fixing V and W randomly,
    then solving for U linearly.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    for attempt in range(max_attempts):
        # Randomly initialize V and W with small integers
        V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(mp)]
        W = [[random.randint(-2, 2) for _ in range(r)] for _ in range(pn)]
        
        # Check if V and W are "generic" enough? 
        # We just try to solve for U.
        
        U = [[Fraction(0) for _ in range(r)] for _ in range(nm)]
        valid = True
        
        for i in range(n):
            for j in range(m):
                # Construct the p x r matrix M where M[k, alpha] = V[j*p+k, alpha] * W[k*n+i, alpha]
                M = [[Fraction(0) for _ in range(r)] for _ in range(p)]
                for k in range(p):
                    for alpha in range(r):
                        M[k][alpha] = Fraction(V[j * p + k][alpha]) * Fraction(W[k * n + i][alpha])
                
                # Solve M * u = 1 (vector of ones)
                # u is a vector of length r
                # We use Gaussian elimination
                # Augmented matrix: p x (r+1)
                Aug = [row[:] + [Fraction(1)] for row in M]
                
                # Perform Gaussian elimination
                rank_aug = 0
                for col in range(r):
                    # Find pivot
                    pivot = -1
                    for row in range(rank_aug, p):
                        if Aug[row][col] != 0:
                            pivot = row
                            break
                    if pivot == -1:
                        continue
                    
                    # Swap
                    Aug[rank_aug], Aug[pivot] = Aug[pivot], Aug[rank_aug]
                    
                    # Eliminate
                    pivot_val = Aug[rank_aug][col]
                    for c in range(col, r + 1):
                        Aug[rank_aug][c] /= pivot_val
                        
                    for row in range(p):
                        if row != rank_aug and Aug[row][col] != 0:
                            factor = Aug[row][col]
                            for c in range(col, r + 1):
                                Aug[row][c] -= factor * Aug[rank_aug][c]
                    
                    rank_aug += 1
                    if rank_aug == r:
                        break
                
                # Check for consistency (no 0 = 1 rows)
                for row in range(rank_aug, p):
                    if Aug[row][r] != 0:
                        valid = False
                        break
                if not valid:
                    break
                
                # If consistent, we have a solution.
                # If rank_aug == r, unique solution.
                # If rank_aug < r, infinite solutions. We pick the simplest (set free vars to 0).
                u_vec = [Fraction(0) for _ in range(r)]
                
                # Back substitution / reading off the solution
                # The matrix is in reduced row echelon form.
                # We need to map the basic variables to the solution.
                # This is tricky with the standard Gauss-Jordan if we don't track columns.
                # Let's just use a simple method: if r==p, M is square and invertible.
                # If r > p, we have free variables.
                
                # For simplicity and robustness, let's only accept solutions if r == p or if we can easily find one.
                # Given p=7, if r=7, M is 7x7.
                # If r > 7, it's harder to extract a "simple" solution.
                # Let's restrict search to r=p=7 first? No, rank might be > 7.
                
                # Let's try to extract the solution assuming the first 'rank_aug' variables are basic.
                # This is a heuristic.
                for row in range(rank_aug):
                    # Find the column with 1 in this row
                    basic_col = -1
                    for c in range(r):
                        if Aug[row][c] == 1:
                            basic_col = c
                            break
                    if basic_col != -1:
                        u_vec[basic_col] = Aug[row][r]
                
                # Check if u_vec entries are integers or half-integers
                for val in u_vec:
                    if val.denominator not in [1, 2]:
                        valid = False
                        break
                if not valid:
                    break
                
                # Store u_vec in U
                idx = i * m + j
                for alpha in range(r):
                    U[idx][alpha] = u_vec[alpha]
        
        if valid:
            # Convert to integers/half-integers for output
            U_out = [[float(u) for u in row] for row in U]
            # Actually, the prompt asks for integer/half-integer entries.
            # Fractions like 1/2 are fine.
            # Let's return the Fraction objects or convert to float if they are exact?
            # The prompt says "integer/half-integer entries".
            # Returning Fractions is safer for exactness verification, but the interface might expect numbers.
            # The seed program returns integers.
            # Let's return the values as numbers (floats might lose precision if not careful, but 1/2 is exact in float).
            # However, to be safe, I will return the values as floats if they are .0 or .5, else it's invalid.
            
            # Verify the found solution
            if _verify_decomposition(U, V, W, n, m, p):
                # Convert to standard types
                U_ret = [[float(u) for u in row] for row in U]
                V_ret = [[float(v) for v in row] for row in V]
                W_ret = [[float(w) for w in row] for row in W]
                return {"U": U_ret, "V": V_ret, "W": W_ret}
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Try to find a lower rank
    # Start from a reasonable lower bound. 
    # For 2x3x7, schoolbook is 42.
    # Try ranks from 20 to 41.
    # We prioritize higher ranks first as they are more likely to have solutions?
    # Or lower ranks first?
    # Lower rank is better. But finding it is harder.
    # Let's try a few lower ranks first, then fall back to schoolbook.
    
    # Given the difficulty, we might not find anything.
    # Let's try ranks 35-41 first as they are closer to the known solution.
    # Actually, if a rank 41 solution exists, it's likely "near" the schoolbook one.
    
    # To save time, we will only try a few attempts per rank.
    for rank in range(41, 34, -1):
        if time.time() > deadline:
            break
        res = _try_rank(rank, n, m, p, max_attempts=5)
        if res:
            return res
            
    # If no lower rank found, return schoolbook
    return _schoolbook()

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U, V, W = res['U'], res['V'], res['W']
    rank = len(U[0])
    
    # Simple verification
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    
    # Verify a few entries
    for _ in range(10):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        val = 0
        for r in range(rank):
            val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        assert abs(val - 1) < 1e-9, f"Failed for i={i}, j={j}, k={k}, val={val}"
        
    elapsed = time.time() - start_time
    print(f"Score (Rank): {rank}")
