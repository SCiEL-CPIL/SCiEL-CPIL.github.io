"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    Uses GF(2)/GF(3) linear algebra to search for low-rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    n_m = n * m  # 9
    m_p = m * p  # 12
    p_n = p * n  # 12
    
    # Schoolbook rank = n*m*p = 36
    schoolbook_rank = n * m * p
    
    def verify_decomposition_frac(U, V, W, rank):
        """Verify with exact rational arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    def verify_decomposition_float(U, V, W, rank, tol=1e-9):
        """Quick verification with floating point."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Start with schoolbook decomposition
    best_rank = schoolbook_rank
    best_U = [[0] * schoolbook_rank for _ in range(n_m)]
    best_V = [[0] * schoolbook_rank for _ in range(m_p)]
    best_W = [[0] * schoolbook_rank for _ in range(p_n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # GF(p) linear algebra approach
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    # We can fix W to be a matrix over GF(q) and solve for U, V
    
    def solve_over_gf(q, r, max_attempts=200):
        """
        Try to find a rank-r decomposition over GF(q).
        Fix W randomly, then solve linear system for U and V.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random W over GF(q)
            W = np.zeros((p_n, r), dtype=int)
            for idx in range(p_n * r):
                row = idx // r
                col = idx % r
                W[row, col] = random.randint(0, q - 1)
            
            # For each (i,k) pair, we need:
            # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # 
            # Fix U to be random, then solve for V (or vice versa)
            # Actually, let's fix both U and W, solve for V
            # Or better: fix W, and for each (i,k), the equation is linear in the products
            # U[i*m+j, r] * V[j*p+k, r]
            
            # Let's try: fix U randomly, fix W randomly, solve for V
            U = np.zeros((n_m, r), dtype=int)
            for idx in range(n_m * r):
                row = idx // r
                col = idx % r
                U[row, col] = random.randint(0, q - 1)
            
            # For each k (0..p-1), i (0..n-1):
            # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system in V[j*p+k, r] for fixed (j,k)
            
            # Actually, let's think differently.
            # For fixed (i,k): sum_{j,r} U[i*m+j,r] * W[k*n+i,r] * V[j*p+k,r] = 1
            # Let A_{i,k,j,r} = U[i*m+j,r] * W[k*n+i,r]
            # Then sum_{j,r} A_{i,k,j,r} * V[j*p+k,r] = 1
            
            # This is a system of n*p = 12 equations in m*p*r = 12*r unknowns (V entries)
            # But V[j*p+k,r] appears in multiple equations (for different i)
            
            # Let's set up the full linear system
            # Unknowns: V[j*p+k, r] for j in 0..m-1, k in 0..p-1, r in 0..r-1
            # That's m*p*r unknowns = 12*r unknowns
            # Equations: n*p = 12 equations
            
            # For equation (i,k): sum_{j=0}^{m-1} sum_{r=0}^{r-1} U[i*m+j,r] * W[k*n+i,r] * V[j*p+k,r] = 1
            
            # Build the system Ax = 1
            num_unknowns = m * p * r
            num_eqs = n * p
            
            # Map unknown index: (j, k, r) -> j * (p*r) + k * r + r
            def v_idx(j, k, rr):
                return j * (p * r) + k * r + rr
            
            # Build A matrix over GF(q)
            A = np.zeros((num_eqs, num_unknowns), dtype=int)
            
            for i in range(n):
                for k in range(p):
                    eq_idx = i * p + k
                    for j in range(m):
                        for rr in range(r):
                            coeff = (U[i*m+j, rr] * W[k*n+i, rr]) % q
                            if coeff != 0:
                                A[eq_idx, v_idx(j, k, rr)] = coeff
            
            # Solve Ax = 1 over GF(q)
            # Use Gaussian elimination
            b = np.ones(num_eqs, dtype=int)
            
            # Augmented matrix
            M = np.hstack([A, b.reshape(-1, 1)]) % q
            
            # Gaussian elimination
            rank_g = 0
            for col in range(num_unknowns):
                if time.time() > deadline:
                    return None
                # Find pivot
                pivot_row = -1
                for row in range(rank_g, num_eqs):
                    if M[row, col] % q != 0:
                        pivot_row = row
                        break
                if pivot_row == -1:
                    continue
                
                # Swap
                M[[rank_g, pivot_row]] = M[[pivot_row, rank_g]]
                
                # Normalize pivot row
                inv = pow(M[rank_g, col], q - 2, q)
                M[rank_g] = (M[rank_g] * inv) % q
                
                # Eliminate
                for row in range(num_eqs):
                    if row != rank_g and M[row, col] % q != 0:
                        factor = M[row, col]
                        M[row] = (M[row] - factor * M[rank_g]) % q
                
                rank_g += 1
            
            # Check consistency
            for row in range(rank_g, num_eqs):
                if M[row, num_unknowns] % q != 0:
                    break  # Inconsistent
            
            # Extract solution
            V = np.zeros(num_unknowns, dtype=int)
            for row in range(rank_g):
                col = -1
                for c in range(num_unknowns):
                    if M[row, c] % q != 0:
                        col = c
                        break
                if col >= 0:
                    V[col] = M[row, num_unknowns] % q
            
            # Reshape V
            V_mat = V.reshape(m, p, r).transpose(1, 0, 2).reshape(m*p, r)
            
            # Verify over integers (lift from GF(q))
            # For GF(2): entries are 0 or 1
            # For GF(3): entries are 0, 1, 2 -> map 2 to -1
            U_int = U.copy()
            V_int = V_mat.copy()
            W_int = W.copy()
            
            if q == 3:
                U_int[U_int == 2] = -1
                V_int[V_int == 2] = -1
                W_int[W_int == 2] = -1
            
            # Verify
            if verify_decomposition_float(U_int.tolist(), V_int.tolist(), W_int.tolist(), r):
                if verify_decomposition_frac(U_int.tolist(), V_int.tolist(), W_int.tolist(), r):
                    return (U_int.tolist(), V_int.tolist(), W_int.tolist())
        
        return None
    
    # Try GF(2) and GF(3) for various ranks
    random.seed(42)
    np.random.seed(42)
    
    for r in range(20, 35):
        if time.time() > deadline:
            break
        
        # Try GF(2)
        result = solve_over_gf(2, r, max_attempts=50)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
        
        if time.time() > deadline:
            break
        
        # Try GF(3)
        result = solve_over_gf(3, r, max_attempts=50)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
    
    # If GF approach didn't work, try direct random search over small integers
    if best_rank == schoolbook_rank:
        def try_direct_search(r, max_attempts=100):
            """Direct random search over {-1, 0, 1}."""
            if time.time() > deadline:
                return None
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    return None
                
                U = np.zeros((n_m, r), dtype=int)
                V = np.zeros((m_p, r), dtype=int)
                W = np.zeros((p_n, r), dtype=int)
                
                for idx in range(n_m * r):
                    row = idx // r
                    col = idx % r
                    U[row, col] = random.choice([-1, 0, 1])
                
                for idx in range(m_p * r):
                    row = idx // r
                    col = idx % r
                    V[row, col] = random.choice([-1, 0, 1])
                
                for idx in range(p_n * r):
                    row = idx // r
                    col = idx % r
                    W[row, col] = random.choice([-1, 0, 1])
                
                if verify_decomposition_float(U.tolist(), V.tolist(), W.tolist(), r):
                    if verify_decomposition_frac(U.tolist(), V.tolist(), W.tolist(), r):
                        return (U.tolist(), V.tolist(), W.tolist())
            
            return None
        
        for r in range(25, 35):
            if time.time() > deadline:
                break
            result = try_direct_search(r, max_attempts=30)
            if result is not None:
                best_rank = r
                best_U, best_V, best_W = result
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

