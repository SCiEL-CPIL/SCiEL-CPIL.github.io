"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Schoolbook rank as baseline
    schoolbook_rank = n * m * p
    
    # Helper to create schoolbook decomposition
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Helper to verify decomposition
    def verify(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        
        # Check all nonzero entries: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check zero entries
        for _ in range(200):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (j == j2 and i == i2 and k == k2):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        
        return True
    
    # Hierarchical decomposition using Kronecker structure
    # Strategy: decompose 2x5x6 into smaller components
    # 2 = 2, 5 = 2+3, 6 = 2*3
    # Use block structure: split j into j1 (size 2) and j2 (size 3)
    # Split k into k1 (size 2) and k2 (size 3)
    
    def make_hierarchical_decomp(n, m, p):
        """
        Decompose using the fact that m=5=2+3 and p=6=2*3
        This gives a structured decomposition with lower rank.
        """
        # Split m into m1=2, m2=3
        # Split p into p1=2, p2=3
        m1, m2 = 2, 3
        p1, p2 = 2, 3
        
        assert m1 + m2 == m, f"m split: {m1}+{m2} != {m}"
        assert p1 * p2 == p, f"p split: {p1}*{p2} != {p}"
        
        # The multiplication tensor T[i, j, k] where:
        # i in [0, n), j in [0, m), k in [0, p)
        # We split j = j1 + m1*j2 where j1 in [0,m1), j2 in [0,m2)
        # We split k = k1 + p1*k2 where k1 in [0,p1), k2 in [0,p2)
        
        # Rank components:
        # Block (j2=0, k2=0): n * m1 * p1 = 2 * 2 * 2 = 8
        # Block (j2=0, k2=1): n * m1 * p1 = 2 * 2 * 2 = 8
        # Block (j2=1, k2=0): n * m2 * p1 = 2 * 3 * 2 = 12
        # Block (j2=1, k2=1): n * m2 * p1 = 2 * 3 * 2 = 12
        # Total: 8 + 8 + 12 + 12 = 40
        
        total_rank = n * m1 * p1 + n * m1 * p1 + n * m2 * p1 + n * m2 * p1
        # = 2*2*2 + 2*2*2 + 2*3*2 + 2*3*2 = 8+8+12+12 = 40
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for j2 in range(m2):  # 0, 1
            for k2 in range(p2):  # 0, 1
                m_block = m1 if j2 == 0 else m2
                p_block = p1
                
                for i in range(n):
                    for j1 in range(m_block):
                        j = j1 + m1 * j2
                        for k1 in range(p_block):
                            k = k1 + p1 * k2
                            
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
        
        return U, V, W, total_rank
    
    # Initialize with schoolbook
    U, V, W, rank = make_schoolbook(n, m, p)
    best_result = (U.copy(), V.copy(), W.copy())
    best_rank = rank
    
    # Try hierarchical decomposition
    U_h, V_h, W_h, rank_h = make_hierarchical_decomp(n, m, p)
    
    if verify(U_h, V_h, W_h, n, m, p, tol=1e-6):
        if rank_h < best_rank:
            best_rank = rank_h
            best_result = (U_h.copy(), V_h.copy(), W_h.copy())
    
    # Try further optimization: random initialization with lower rank targets
    def random_decomp_attempt(target_rank, seed):
        """Try to find a decomposition of given rank via random init + local search."""
        np.random.seed(seed)
        
        U_r = np.random.randn(n * m, target_rank) * 0.5
        V_r = np.random.randn(m * p, target_rank) * 0.5
        W_r = np.random.randn(p * n, target_rank) * 0.5
        
        # Simple coordinate descent / local search
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Try to improve by adjusting one matrix at a time
            # For each element, try small perturbations
            improved = False
            
            for matrix, dim in [(U_r, n*m), (V_r, m*p), (W_r, p*n)]:
                if time.time() > deadline - 10:
                    break
                
                for _ in range(5):
                    row = np.random.randint(0, dim)
                    col = np.random.randint(0, target_rank)
                    old_val = matrix[row, col]
                    
                    # Try a few perturbations
                    for delta in [-0.1, -0.05, 0.05, 0.1]:
                        new_val = old_val + delta
                        matrix[row, col] = new_val
                        
                        # Quick check
                        if verify(U_r, V_r, W_r, n, m, p, tol=1e-4):
                            improved = True
                            break
                        
                        matrix[row, col] = old_val
                    else:
                        continue
                    break
            
            if not improved:
                break
        
        return U_r, V_r, W_r
    
    # Try various target ranks from hierarchical down to lower
    for target_rank in range(rank_h - 1, max(10, rank_h - 10), -1):
        if time.time() > deadline - 20:
            break
        
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            
            U_try, V_try, W_try = random_decomp_attempt(target_rank, seed * 1000 + target_rank)
            
            if verify(U_try, V_try, W_try, n, m, p, tol=1e-5):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = (U_try.copy(), V_try.copy(), W_try.copy())
                    break
    
    # Convert to list of lists with integer/half-integer entries
    def to_fraction_list(matrix):
        result = []
        for row in matrix:
            frac_row = []
            for val in row:
                if abs(val) < 1e-10:
                    frac_row.append(0)
                else:
                    # Try denominators 1, 2, 4, 8
                    found = False
                    for denom in [1, 2, 4, 8]:
                        numer = round(val * denom)
                        if abs(val - numer / denom) < 1e-6:
                            frac_row.append(numer / denom)
                            found = True
                            break
                    if not found:
                        frac_row.append(val)
            result.append(frac_row)
        return result
    
    U_final, V_final, W_final = best_result
    return {"U": to_fraction_list(U_final), "V": to_fraction_list(V_final), "W": to_fraction_list(W_final)}
# EVOLVE-BLOCK-END
