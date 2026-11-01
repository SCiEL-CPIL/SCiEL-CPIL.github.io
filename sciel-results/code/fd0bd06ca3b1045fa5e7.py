import time
import random
import numpy as np
from itertools import product

def verify_decomposition_strong(U, V, W, n=2, m=2, p=7):
    """
    Strong verification:
    sum_r W[r, i, k] * U[r, a, b] * V[r, c, d] == delta(a,i) * delta(d,k) * delta(b,c)
    """
    n, m, p = 2, 2, 7
    rank = len(U[0])
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # Indices:
    # U[i*m+j, r]
    # V[j*p+k, r]
    # W[k*n+i, r] -> This is W[r, k, i] in the previous thought?
    # Let's stick to the seed's indexing: W[k*n+i, r]
    
    # The condition derived:
    # sum_r w_{r,i,k} u_{r,a,b} v_{r,c,d} = delta(a,i) delta(d,k) delta(b,c)
    # If W is stored as W[k*n+i, r], then W[k*n+i, r] corresponds to w_{r,k,i}.
    # So the condition is:
    # sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] == delta(a,i) * delta(d,k) * delta(b,c)
    
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and d == k and b == c) else 0.0
                            s = 0.0
                            for r in range(rank):
                                s += W[k*n+i][r] * U[a*m+b][r] * V[c*p+d][r]
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 2, 2, 7
    schoolbook_rank = n * m * p # 28
    
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find rank 25
    target_rank = 25
    
    # Strategy: Randomize U and V, solve for W
    # We need to solve:
    # sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = Target[a,b,c,d,i,k]
    # This is a linear system in W.
    # Number of unknowns: rank * p * n = 25 * 14 = 350.
    # Number of equations: n * p * n * m * m * p = 2 * 7 * 2 * 2 * 2 * 7 = 196.
    # Under-determined. We can find a minimum norm solution.
    # But we need integer/half-integer solutions.
    
    # We can construct the matrix A such that A * W_flat = b
    # W_flat is of size (rank * p * n)
    # A is of size (n * p * n * m * m * p, rank * p * n)
    
    # To save time, we will try a few random initializations of U and V.
    
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Random U, V with small integers
        U_rand = np.random.randint(-2, 3, size=(n*m, target_rank))
        V_rand = np.random.randint(-2, 3, size=(m*p, target_rank))
        
        # Build linear system for W
        # Equations indexed by (i, k, a, b, c, d)
        # Unknowns indexed by (r, k, i) -> W[k*n+i, r]
        
        num_eq = n * p * n * m * m * p
        num_unk = target_rank * p * n
        
        A_mat = np.zeros((num_eq, num_unk))
        b_vec = np.zeros(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                expected = 1.0 if (a == i and d == k and b == c) else 0.0
                                b_vec[eq_idx] = expected
                                
                                # Coefficients for W
                                # Term: W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r]
                                for r in range(target_rank):
                                    coeff = U_rand[a*m+b, r] * V_rand[c*p+d, r]
                                    # W index: k*n+i, r
                                    # Flat index: r * (p*n) + (k*n+i)
                                    unk_idx = r * (p*n) + (k*n+i)
                                    A_mat[eq_idx, unk_idx] += coeff
                                eq_idx += 1
                                
        # Solve A_mat * W_flat = b_vec
        # Use least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W_rand = W_flat.reshape(target_rank, p*n)
            W_rand = W_rand.T # W_rand[i, r] where i is k*n+i?
            # W_rand shape (p*n, target_rank)
            # W_rand[k*n+i, r]
            
            # Check if solution is close to integer/half-integer
            # And verify
            # Round to nearest half-integer
            W_rounded = np.round(W_rand * 2) / 2.0
            
            # Verify
            if verify_decomposition_strong(U_rand.tolist(), V_rand.tolist(), W_rounded.tolist()):
                best_U, best_V, best_W = U_rand.tolist(), V_rand.tolist(), W_rounded.tolist()
                best_rank = target_rank
                break # Found rank 25
                
        except Exception:
            continue
            
    return {"U": best_U, "V": best_V, "W": best_W}
