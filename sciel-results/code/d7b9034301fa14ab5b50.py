"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    The decomposition satisfies: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    schoolbook_rank = n * m * p  # 144
    
    # Start with schoolbook as fallback
    best_solution = _schoolbook_solution(n, m, p)
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # Known bounds for 4x4*4x9: likely around 36-50
    min_possible_rank = max(n * m, m * p, p * n)
    
    # Try to find lower rank solutions using iterative refinement with exact verification
    for target_rank in range(min_possible_rank, best_rank):
        if time.time() - start_time > deadline - 15:
            break
        
        # Multiple random restarts
        max_restarts = 8
        found = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline - 10:
                break
            
            # Random initialization with structured patterns
            seed_val = restart * 1000 + target_rank
            np.random.seed(seed_val)
            
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Iterative refinement: alternating least squares style
            converged = False
            for iteration in range(80):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Update U: fix V, W; for each (i,j), solve linear system
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            v_idx = j * p + k
                            w_idx = k * n + i
                            A[k, :] = V[v_idx, :] * W[w_idx, :]
                        b = np.ones(p)
                        try:
                            U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V: fix U, W
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            u_idx = i * m + j
                            w_idx = k * n + i
                            A[i, :] = U[u_idx, :] * W[w_idx, :]
                        b = np.ones(n)
                        try:
                            V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W: fix U, V
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            A[j, :] = U[u_idx, :] * V[v_idx, :]
                        b = np.ones(m)
                        try:
                            W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Try to convert to exact fractions
                solution = _try_convert_to_exact(U, V, W, target_rank, n, m, p)
                if solution is not None:
                    best_solution = solution
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Continue trying even lower ranks
            continue
    
    # Also try a direct structured approach for specific ranks
    if time.time() - start_time < deadline - 20:
        structured = _try_structured_search(n, m, p, deadline - time.time())
        if structured is not None and len(structured['U'][0]) < best_rank:
            best_solution = structured
            best_rank = len(structured['U'][0])
    
    return best_solution


def _schoolbook_solution(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the factor matrices exactly decompose the matrix multiplication tensor.
    Uses Fraction arithmetic for exactness.
    """
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                if total != 1:
                    return False
    return True


def _try_convert_to_exact(U_float, V_float, W_float, rank, n, m, p):
    """
    Try to convert floating-point solution to exact fractions and verify.
    """
    # Try with increasing precision
    for max_denom in [10, 100, 1000, 10000, 100000]:
        U_frac = []
        V_frac = []
        W_frac = []
        
        for i in range(n * m):
            row = []
            for c in range(rank):
                val = U_float[i, c]
                frac = Fraction(val).limit_denominator(max_denom)
                row.append(frac)
            U_frac.append(row)
        
        for i in range(m * p):
            row = []
            for c in range(rank):
                val = V_float[i, c]
                frac = Fraction(val).limit_denominator(max_denom)
                row.append(frac)
            V_frac.append(row)
        
        for i in range(p * n):
            row = []
            for c in range(rank):
                val = W_float[i, c]
                frac = Fraction(val).limit_denominator(max_denom)
                row.append(frac)
            W_frac.append(row)
        
        if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # If floating point solution is very close, try higher precision conversion
    # Check if the float solution is actually exact within tolerance
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_float[u_idx, :], V_float[v_idx, :] * W_float[w_idx, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-12:
        # Very close to exact; try much higher precision
        for max_denom in [1000000, 10000000]:
            U_frac = []
            V_frac = []
            W_frac = []
            
            for i in range(n * m):
                row = []
                for c in range(rank):
                    val = U_float[i, c]
                    frac = Fraction(val).limit_denominator(max_denom)
                    row.append(frac)
                U_frac.append(row)
            
            for i in range(m * p):
                row = []
                for c in range(rank):
                    val = V_float[i, c]
                    frac = Fraction(val).limit_denominator(max_denom)
                    row.append(frac)
                V_frac.append(row)
            
            for i in range(p * n):
                row = []
                for c in range(rank):
                    val = W_float[i, c]
                    frac = Fraction(val).limit_denominator(max_denom)
                    row.append(frac)
                W_frac.append(row)
            
            if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None


def _try_structured_search(n, m, p, time_budget):
    """
    Try structured patterns that might yield lower rank decompositions.
    Exploits the specific structure of 4x4 * 4x9.
    """
    start_time = time.time()
    
    # Try ranks that are multiples of small numbers
    candidate_ranks = [36, 40, 44, 48, 52, 56, 60, 64, 72, 80]
    
    for rank in candidate_ranks:
        if time.time() - start_time > time_budget - 10:
            break
        
        # Try Kronecker-like structure
        # For 4x4 * 4x9, we can think of 4x4 as 2x2 block of 2x2
        # and 4x9 as 2x2 block where second dim is 2x5
        
        # Pattern 1: Tensor product structure
        # U = U1 ⊗ U2, V = V1 ⊗ V2, W = W1 ⊗ W2
        # where U1 is 2x2, U2 is 2x2, etc.
        
        # Actually, let's try a more direct structured approach
        # Initialize with a pattern based on the index structure
        
        np.random.seed(rank)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Try to assign each (i,j,k) triple to a specific column
        # with some sharing
        
        # For rank 36 = m*p, try: column = j*p + k, and U depends on i
        if rank == m * p:
            for j in range(m):
                for k in range(p):
                    col = j * p + k
                    for i in range(n):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        # But this gives sum over i, not what we want
                        # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each (i,j,k)
                        # This approach won't work directly
        
        # Try random structured initialization
        for _ in range(3):
            if time.time() - start_time > time_budget - 5:
                break
            
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
            
            converged = False
            for iteration in range(60):
                if time.time() - start_time > time_budget - 3:
                    break
                
                # Alternating updates
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        A = np.zeros((p, rank))
                        for k in range(p):
                            v_idx = j * p + k
                            w_idx = k * n + i
                            A[k, :] = V[v_idx, :] * W[w_idx, :]
                        b = np.ones(p)
                        try:
                            U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A = np.zeros((n, rank))
                        for i in range(n):
                            u_idx = i * m + j
                            w_idx = k * n + i
                            A[i, :] = U[u_idx, :] * W[w_idx, :]
                        b = np.ones(n)
                        try:
                            V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A = np.zeros((m, rank))
                        for j in range(m):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            A[j, :] = U[u_idx, :] * V[v_idx, :]
                        b = np.ones(m)
                        try:
                            W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-10:
                    converged = True
                    break
            
            if converged:
                solution = _try_convert_to_exact(U, V, W, rank, n, m, p)
                if solution is not None:
                    return solution
    
    return None

# EVOLVE-BLOCK-END
