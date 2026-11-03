"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen patterns for 2x2x2
    strassen_patterns = [
        # M1: (A00+A11)(B00+B11)
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        # M2: (A10+A11)B00
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        # M3: A00(B01-B11)
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        # M4: A11(B10-B00)
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        # M5: (A00+A01)B11
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        # M6: (A10-A00)(B00+B01)
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        # M7: (A01-A11)(B10+B11)
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
    ]
    
    def block_rank(n_b, m_b, p_b):
        if n_b == 2 and m_b == 2 and p_b == 2:
            return 7
        return n_b * m_b * p_b
    
    def try_splits(n_splits, m_splits, p_splits):
        """Build decomposition with given splits, return (U, V, W, rank)."""
        n_off = [0]
        for s in n_splits[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in m_splits[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in p_splits[:-1]:
            p_off.append(p_off[-1] + s)
        
        total_rank = sum(block_rank(n_splits[i], m_splits[j], p_splits[k]) 
                        for i in range(len(n_splits)) for j in range(len(m_splits)) for k in range(len(p_splits)))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]
                    
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        for pattern in strassen_patterns:
                            for li, lj, val in pattern['U']:
                                U[(i_off + li) * m + (j_off + lj)][rank] = val
                            for lj, lk, val in pattern['V']:
                                V[(j_off + lj) * p + (k_off + lk)][rank] = val
                            for lk, li, val in pattern['W']:
                                W[(k_off + lk) * n + (i_off + li)][rank] = val
                            rank += 1
                    else:
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
                                    U[(i_off + li) * m + (j_off + lj)][rank] = 1
                                    V[(j_off + lj) * p + (k_off + lk)][rank] = 1
                                    W[(k_off + lk) * n + (i_off + li)][rank] = 1
                                    rank += 1
        
        return U, V, W, rank
    
    # Try various split configurations and pick the best
    best_U, best_V, best_W, best_rank = None, None, None, n*m*p
    
    # Generate candidate splits
    def gen_splits(total, max_parts=3):
        """Generate all compositions of 'total' into up to max_parts positive integers."""
        results = []
        def backtrack(remaining, parts, start):
            if not parts:
                if remaining == 0:
                    results.append([])
                return
            if len(parts) == 1:
                if remaining > 0:
                    results.append([remaining])
                return
            for v in range(1, remaining - (len(parts) - 1) + 1):
                backtrack(remaining - v, parts[1:], v)
        backtrack(total, list(range(max_parts)), 0)
        # Also add single-part split
        results.append([total])
        return results
    
    n_split_options = gen_splits(3)
    m_split_options = gen_splits(5)
    p_split_options = gen_splits(5)
    
    start_time = time.time()
    
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                if time.time() - start_time > 100:
                    break
                U_t, V_t, W_t, r = try_splits(ns, ms, ps)
                if r < best_rank:
                    best_U, best_V, best_W, best_rank = U_t, V_t, W_t, r
    
    # Verify the best decomposition
    def verify(U, V, W, n, m, p, tol=1e-6):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        U_r = U_arr.reshape(n, m, rank)
        V_r = V_arr.reshape(m, p, rank)
        W_r = W_arr.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i,j,:], V_r[j,k,:] * W_r[k,i,:])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if best_U is None or not verify(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
