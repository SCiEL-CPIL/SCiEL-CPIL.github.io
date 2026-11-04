"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if i,j,k are "aligned" (the multiplication tensor)
    # Actually, the multiplication tensor has T[i,j,k] = 1 for the standard basis
    # The tensor for matrix mult: T_{i,j,k} where (i,j) indexes A, (j,k) indexes B, (i,k) indexes C
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k)
    
    # Actually, the multiplication tensor is:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is defined as T[i,j,k] = 1 for all i,j,k in range
    # Wait, no. The tensor T has entries T[i,j,k] where:
    # - i indexes rows of A (and C): 0..n-1
    # - j indexes cols of A / rows of B: 0..m-1  
    # - k indexes cols of B (and C): 0..p-1
    # T[i,j,k] = 1 for all valid indices (it's the structure tensor)
    
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using the W[k*n+i, r] convention
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _verify_exact_frac(U, V, W, n, m, p):
    """Verify with Fraction arithmetic for exactness."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m + j][c]) * Fraction(V[j*p + k][c]) * Fraction(W[k*n + i][c])
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using ALS
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with a reasonable initial rank and try to reduce
    # Known bounds: rank >= max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # Actually rank >= max(n, m, p) * something... 
    # For matrix multiplication, rank >= n*m + m*p + p*n - ... (various lower bounds)
    # Let's just search from low to high
    
    min_possible = max(nm, mp, pn)  # 30
    max_search = min(best_rank, 50)  # Don't go too high
    
    for target_rank in range(min_possible, max_search + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + trial)
            
            # Initialize U, V, W randomly
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # ALS iterations
            max_iter = 100
            tol = 1e-10
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] = (V[j*p+*, :] * W[*n+i, :])^T / T[i,j,:]
                # T[i,j,k] = 1 for all k, so we need sum_k V[j*p+k, c] * W[k*n+i, c] = 1 for all c? No.
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # For fixed i,j: the vector (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c])_k should be all 1s
                # This is: U[i*m+j, :] @ diag? No...
                # Let's think: for fixed i,j, define a_c = U[i*m+j, c]
                # Then we need: sum_c a_c * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is: (a * (V[j*p+*, :] * W[*n+i, :])) should give all 1s when summed over c
                # i.e., sum_c a_c * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # Let B_k = sum_c a_c * V[j*p+k,c] * W[k*n+i,c]
                # We want B_k = 1 for all k
                # This is a linear system in a: for each k, sum_c a_c * (V[j*p+k,c] * W[k*n+i,c]) = 1
                # So: M @ a = ones, where M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Build M: p x target_rank, M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                        M_mat = V[j*p:(j+1)*p, :] * W[i::n, :]  # W[k*n+i, c] for k=0..p-1 -> W[i, i+n, i+2n, ...]
                        # Wait, W has shape (p*n, rank). W[k*n + i, c]. For k=0..p-1: indices i, i+n, i+2n, ..., i+(p-1)*n
                        # So W[i::n, :] gives rows i, i+n, i+2n, ... which is p rows. Good.
                        # V[j*p:(j+1)*p, :] gives p rows. Good.
                        # M_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        try:
                            # Solve M_mat @ a = ones(p)
                            a, residuals, rank_r, sv = np.linalg.lstsq(M_mat, np.ones(p), rcond=None)
                            U_new[i*m + j, :] = a
                        except:
                            U_new[i*m + j, :] = U[i*m + j, :]
                U = U_new
                
                # Update V similarly
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # For fixed j,k: need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                        # Let b_c = V[j*p+k, c]
                        # sum_c b_c * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
                        # M[i,c] = U[i*m+j,c] * W[k*n+i,c], i=0..n-1
                        M_mat = U[j::m, :] * W[k*n:(k+1)*n, :]  # U[i*m+j, c] for i=0..n-1 -> U[j, j+m, j+2m, ...]
                        # U[i*m+j, c]: for i=0..n-1, index is i*m+j. So U[j::m, :] gives rows j, j+m, j+2m, ...
                        # W[k*n+i, c]: for i=0..n-1, index is k*n+i. So W[k*n:(k+1)*n, :]
                        try:
                            b, residuals, rank_r, sv = np.linalg.lstsq(M_mat, np.ones(n), rcond=None)
                            V_new[j*p + k, :] = b
                        except:
                            V_new[j*p + k, :] = V[j*p + k, :]
                V = V_new
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # For fixed k,i: need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                        # Let d_c = W[k*n+i, c]
                        # sum_c d_c * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
                        # M[j,c] = U[i*m+j,c] * V[j*p+k,c], j=0..m-1
                        M_mat = U[i*m:(i+1)*m, :] * V[k::p, :]  # U[i*m+j, c] for j=0..m-1, V[j*p+k, c] for j=0..m-1
                        # V[j*p+k, c]: for j=0..m-1, index is j*p+k. So V[k::p, :] gives rows k, k+p, k+2p, ...
                        try:
                            d, residuals, rank_r, sv = np.linalg.lstsq(M_mat, np.ones(m), rcond=None)
                            W_new[k*n + i, :] = d
                        except:
                            W_new[k*n + i, :] = W[k*n + i, :]
                W = W_new
                
                # Check convergence
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (val - 1.0)**2
                if err < tol:
                    found = True
                    break
        
        if found:
            # Convert to exact fractions
            U_frac = [[0]*target_rank for _ in range(nm)]
            V_frac = [[0]*target_rank for _ in range(mp)]
            W_frac = [[0]*target_rank for _ in range(pn)]
            
            for r in range(nm):
                for c in range(target_rank):
                    U_frac[r][c] = float(U[r, c])
            for r in range(mp):
                for c in range(target_rank):
                    V_frac[r][c] = float(V[r, c])
            for r in range(pn):
                for c in range(target_rank):
                    W_frac[r][c] = float(W[r, c])
            
            # Verify with fractions
            # Round to nearest rational with small denominator
            def to_frac(x):
                # Try to find a simple fraction
                if abs(x) < 1e-10:
                    return 0
                # Check if it's close to a simple fraction
                for den in range(1, 100):
                    num = round(x * den)
                    if abs(x - num/den) < 1e-9:
                        if den == 1:
                            return int(num)
                        return num/den
                return x
            
            U_final = [[0]*target_rank for _ in range(nm)]
            V_final = [[0]*target_rank for _ in range(mp)]
            W_final = [[0]*target_rank for _ in range(pn)]
            
            for r in range(nm):
                for c in range(target_rank):
                    U_final[r][c] = to_frac(U[r, c])
            for r in range(mp):
                for c in range(target_rank):
                    V_final[r][c] = to_frac(V[r, c])
            for r in range(pn):
                for c in range(target_rank):
                    W_final[r][c] = to_frac(W[r, c])
            
            # Verify exactness
            if _verify_exact_frac(U_final, V_final, W_final, n, m, p):
                best_result = {"U": U_final, "V": V_final, "W": W_final}
                best_rank = target_rank
                break
            else:
                # Try with higher precision
                # Use Fraction directly
                U_f = [[Fraction(0)]*target_rank for _ in range(nm)]
                V_f = [[Fraction(0)]*target_rank for _ in range(mp)]
                W_f = [[Fraction(0)]*target_rank for _ in range(pn)]
                
                for r in range(nm):
                    for c in range(target_rank):
                        U_f[r][c] = Fraction(U[r, c]).limit_denominator(1000)
                for r in range(mp):
                    for c in range(target_rank):
                        V_f[r][c] = Fraction(V[r, c]).limit_denominator(1000)
                for r in range(pn):
                    for c in range(target_rank):
                        W_f[r][c] = Fraction(W[r, c]).limit_denominator(1000)
                
                if _verify_exact_frac(U_f, V_f, W_f, n, m, p):
                    # Convert to int/float
                    U_out = [[int(U_f[r][c]) if U_f[r][c].denominator == 1 else float(U_f[r][c]) for c in range(target_rank)] for r in range(nm)]
                    V_out = [[int(V_f[r][c]) if V_f[r][c].denominator == 1 else float(V_f[r][c]) for c in range(target_rank)] for r in range(mp)]
                    W_out = [[int(W_f[r][c]) if W_f[r][c].denominator == 1 else float(W_f[r][c]) for c in range(target_rank)] for r in range(pn)]
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    best_rank = target_rank
                    break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
