import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W):
    """Verify exact tensor decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for M in (U, V, W):
        for row in M:
            if len(row) != rank:
                return False
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    one = Fraction(1, 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                urow = Uf[i * m + j]
                vrow = Vf[j * p + k]
                wrow = Wf[k * n + i]
                for r in range(rank):
                    s += urow[r] * vrow[r] * wrow[r]
                if s != one:
                    return False
    return True

def _build_rank88():
    """Build the known rank-88 decomposition."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: i=0, schoolbook 1x4x8
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2 via Strassen 2x2x2 blocks
    sU = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
    ]

    for jb in range(2):
        j0 = jb * 2
        for kb in range(4):
            k0 = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (j0 + lj)][col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(j0 + lj) * p + (k0 + lk)][col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(k0 + lk) * n + (1 + li)][col] += sW[t][l]
                col += 1

    return {"U": U, "V": V, "W": W}

def _try_improve_rank(U, V, W, target_rank, timeout):
    """Try to reduce rank by finding linear dependencies among the rank-1 terms."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Try to find if any two columns are linearly dependent in a way that allows merging
    # For each pair of columns (a, b), check if we can express one as a combination of the other
    # This is complex, so let's try a simpler approach:
    # Check if removing one column and adjusting others still works
    
    # Alternative: try to find a rank-(rank-1) decomposition by setting one variable to zero
    # and solving for the rest
    
    # For now, let's try a heuristic: look for sparse patterns that might allow reduction
    # Check if any U, V, or W row has a very sparse pattern that suggests redundancy
    
    # A more systematic approach: try to find a vector (u, v, w) in the null space
    # of the constraint matrix that has integer entries
    
    # The constraint matrix has 96 rows (one per (i,j,k)) and rank*3 columns
    # This is too large for direct LLL, so let's use a different strategy
    
    # Strategy: try to find if two terms can be combined
    # Term r contributes: U[:,r] * V[:,r] * W[:,r] to the tensor
    # If term a and term b have proportional U, V, or W vectors, they might be mergeable
    
    # For each pair (a, b), check if U[:,a] is proportional to U[:,b]
    # If so, we can potentially merge them
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for a in range(rank):
        if time.time() - start_time > timeout:
            break
        for b in range(a + 1, rank):
            # Check if U[:,a] and U[:,b] are proportional
            ua = U_arr[:, a]
            ub = U_arr[:, b]
            
            # Check sparsity - only consider if both are sparse
            if np.count_nonzero(ua) > 3 or np.count_nonzero(ub) > 3:
                continue
            
            # Check proportionality
            nonzero_a = ua != 0
            nonzero_b = ub != 0
            
            if not np.all(nonzero_a == nonzero_b):
                continue
            
            if not np.any(nonzero_a):
                continue
            
            # Check if ua/ub is constant on nonzero entries
            ratios = ua[nonzero_a] / ub[nonzero_b]
            if len(ratios) < 2:
                continue
            if np.allclose(ratios, ratios[0]):
                # Proportional! Try to merge
                ratio = ratios[0]
                # New term: U_new = U[:,a], V_new = V[:,a] + ratio*V[:,b], W_new = W[:,a]
                # Or: U_new = U[:,a] + ratio*U[:,b], V_new = V[:,a], W_new = W[:,a]
                # We need to check which merge preserves the tensor
                
                # Option 1: merge into a
                V_new = V_arr[:, a] + ratio * V_arr[:, b]
                W_new = W_arr[:, a]
                U_new = U_arr[:, a]
                
                # Check if this gives integer/half-integer entries
                if np.allclose(V_new * 2, np.round(V_new * 2)):
                    # Create new matrices with rank-1
                    U_new_full = U_arr.copy()
                    V_new_full = V_arr.copy()
                    W_new_full = W_arr.copy()
                    
                    # Remove column b, update column a
                    U_new_full[:, a] = U_new
                    V_new_full[:, a] = V_new
                    W_new_full[:, a] = W_new
                    
                    # Remove column b
                    cols_to_keep = [c for c in range(rank) if c != b]
                    U_final = U_new_full[:, cols_to_keep]
                    V_final = V_new_full[:, cols_to_keep]
                    W_final = W_new_full[:, cols_to_keep]
                    
                    # Convert to list of lists with Fraction
                    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else Fraction(x).limit_denominator(10000) for x in row] for row in U_final]
                    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else Fraction(x).limit_denominator(10000) for x in row] for row in V_final]
                    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else Fraction(x).limit_denominator(10000) for x in row] for row in W_final]
                    
                    if _verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    # Start with the known rank-88 solution
    best = _build_rank88()
    
    if not _verify(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        best = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to improve by finding linear dependencies
    if time.time() < deadline:
        improved = _try_improve_rank(best["U"], best["V"], best["W"], len(best["U"][0]) - 1, 50)
        if improved:
            best = improved
    
    return best
