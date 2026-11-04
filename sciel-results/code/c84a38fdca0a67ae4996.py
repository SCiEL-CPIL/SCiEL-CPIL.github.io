"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
- Use schoolbook (rank 42) as safe fallback.
- Exploit n=2 structure: W has only p*n=14 rows.
- Systematically search structured families of U, V, solve for W via linear algebra.
- Use exact rational arithmetic for verification.
"""

import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition_full(U, V, W, n, m, p):
    """
    Verify exact decomposition using rational arithmetic.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Tensor condition: sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i,c] = delta(i,i')*delta(j,j')*delta(k,k')
    """
    rank = len(U[0])
    
    # Pre-convert to Fraction for efficiency
    U_f = [[Fraction(str(U[i][c])) for c in range(rank)] for i in range(len(U))]
    V_f = [[Fraction(str(V[i][c])) for c in range(rank)] for i in range(len(V))]
    W_f = [[Fraction(str(W[i][c])) for c in range(rank)] for i in range(len(W))]
    
    # Check diagonal terms (should equal 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i*m+j][c] * V_f[j*p+k][c] * W_f[k*n+i][c]
                if total != 1:
                    return False
    
    # Check cross terms (should equal 0)
    # For efficiency, check all combinations where at least one index mismatches
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            if i == ip and j == jp and k == kp:
                                continue
                            total = Fraction(0)
                            for c in range(rank):
                                total += U_f[i*m+j][c] * V_f[jp*p+k][c] * W_f[kp*n+ip][c]
                            if total != 0:
                                return False
    
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast float-based verification for quick rejection."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_mat = U_arr.reshape(n, m, rank)
    V_mat = V_arr.reshape(m, p, rank)
    W_mat = W_arr.reshape(p, n, rank)
    
    # Check diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_mat[i, j, :] * V_mat[j, k, :] * W_mat[k, i, :]
                if abs(np.sum(prod) - 1.0) > 1e-8:
                    return False
    
    # Check random cross terms
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        jp = random.randint(0, m-1)
        k = random.randint(0, p-1)
        ip = random.randint(0, n-1)
        kp = random.randint(0, p-1)
        if i == ip and j == jp and k == kp:
            continue
        prod = U_mat[i, j, :] * V_mat[jp, k, :] * W_mat[kp, ip, :]
        if abs(np.sum(prod)) > 1e-8:
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

def _try_solve_for_W(U, V, n, m, p, rank):
    """
    Given U and V, try to solve for W such that the tensor decomposition holds.
    The condition is: for all (i,j,j',k,k',i'):
        sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i',c] = delta(i,i')*delta(j,j')*delta(k,k')
    
    This is a linear system in W. We set up the system and solve it.
    Returns W if a solution exists with rational entries, else None.
    """
    pn = p * n
    # Number of equations: n*m * m*p * p*n = (n*m)*(m*p)*(p*n)
    # But many are redundant. Let's set up the system properly.
    
    # For each combination of (i, j, jp, k, ip, kp), we have an equation.
    # The unknowns are W[k'*n+i', c] for all (k', i', c).
    # There are pn * rank unknowns.
    
    # Build the linear system A * w = b where w is the vectorized W.
    # For equation (i, j, jp, k, ip, kp):
    #   sum_c U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c] = rhs
    # where rhs = 1 if i==ip and j==jp and k==kp, else 0
    
    # Let's collect unique equations
    equations = []
    rhs = []
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            rhs_val = 1 if (i == ip and j == jp and k == kp) else 0
                            # Coefficients for each unknown W[kp*n+ip, c]
                            # The equation is: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c] = rhs_val
                            # So coefficient of W[kp*n+ip, c] is U[i*m+j,c] * V[jp*p+k,c]
                            row = [0.0] * (pn * rank)
                            for c in range(rank):
                                coeff = U[i*m+j][c] * V[jp*p+k][c]
                                if coeff != 0:
                                    row[(kp*n+ip) * rank + c] = coeff
                            equations.append(row)
                            rhs.append(rhs_val)
    
    A = np.array(equations, dtype=np.float64)
    b = np.array(rhs, dtype=np.float64)
    
    # Solve the linear system
    try:
        w_sol, residuals, rank_svd, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    # Check if the solution satisfies the system
    residual = np.linalg.norm(A @ w_sol - b)
    if residual > 1e-10:
        return None
    
    # Reshape w_sol to W
    W = [[0] * rank for _ in range(pn)]
    for idx in range(pn):
        for c in range(rank):
            W[idx][c] = w_sol[idx * rank + c]
    
    # Round to nearest rational (try integers and half-integers)
    W_rational = [[0] * rank for _ in range(pn)]
    for idx in range(pn):
        for c in range(rank):
            val = W[idx][c]
            # Try to find a simple rational
            best_frac = Fraction(0)
            best_err = float('inf')
            for den in [1, 2]:
                num = round(val * den)
                frac = Fraction(num, den)
                err = abs(float(frac) - val)
                if err < best_err:
                    best_err = err
                    best_frac = frac
            if best_err > 1e-8:
                return None
            W_rational[idx][c] = float(best_frac)
    
    return W_rational

def _try_structured_search(rank, n, m, p, time_limit):
    """
    Try to find a decomposition of given rank using structured families.
    """
    start = time.time()
    
    # Family 1: U and V with Kronecker-like structure
    # U[i*m+j, c] depends on i, j and some parameter
    # Try random but structured initializations
    
    attempts = 0
    while time.time() - start < time_limit and attempts < 200:
        attempts += 1
        
        # Generate structured U
        U = [[0] * rank for _ in range(n * m)]
        for i in range(n):
            for j in range(m):
                for c in range(rank):
                    # Use a pattern based on i, j, c
                    if (i + j + c) % 3 == 0:
                        U[i*m+j][c] = 1
                    elif (i + 2*j + c) % 3 == 0:
                        U[i*m+j][c] = -1
                    else:
                        U[i*m+j][c] = 0
        
        # Generate structured V
        V = [[0] * rank for _ in range(m * p)]
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    if (j + k + c) % 3 == 0:
                        V[j*p+k][c] = 1
                    elif (2*j + k + c) % 3 == 0:
                        V[j*p+k][c] = -1
                    else:
                        V[j*p+k][c] = 0
        
        W = _try_solve_for_W(U, V, n, m, p, rank)
        if W is not None:
            # Verify
            if _verify_decomposition_fast(U, V, W, n, m, p):
                if _verify_decomposition_full(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        # Try different random seed for next iteration
        random.seed(random.randint(0, 2**31))
    
    return None

def _try_random_search(rank, n, m, p, time_limit):
    """
    Try random search with small integer entries.
    """
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    while time.time() - start < time_limit:
        # Random U, V with small integers
        U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(nm)]
        V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(mp)]
        
        W = _try_solve_for_W(U, V, n, m, p, rank)
        if W is not None:
            if _verify_decomposition_fast(U, V, W, n, m, p):
                if _verify_decomposition_full(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        # Also try direct random W
        W_rand = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(pn)]
        if _verify_decomposition_fast(U, V, W_rand, n, m, p):
            if _verify_decomposition_full(U, V, W_rand, n, m, p):
                return {"U": U, "V": V, "W": W_rand}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from lower bound up to schoolbook rank
    # Lower bound: at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But realistically, for matrix multiplication, lower bounds are higher
    # Try from 25 to 41
    
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
        
        # Allocate time for this rank
        time_per_rank = min(10, (deadline - time.time()) / max(1, best_rank - rank))
        
        # Try structured search first
        result = _try_structured_search(rank, n, m, p, time_per_rank * 0.5)
        if result is not None:
            return result
        
        # Try random search
        if time.time() < deadline - 5:
            result = _try_random_search(rank, n, m, p, time_per_rank * 0.5)
            if result is not None:
                return result
    
    # Return schoolbook as fallback
    return best_result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    elapsed = time.time() - start
    rank = len(res['U'][0])
    print(f"Rank: {rank}, Time: {elapsed:.2f}s")
