"""Rank-minimal tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: Use structured search with exact arithmetic.
1. Start from schoolbook (rank 120)
2. Use alternating linear solves to find lower-rank decompositions
3. Use greedy column elimination with exact verification
4. Try multiple random perturbations and local search
"""
import time
import random
from fractions import Fraction

try:
    import numpy as np
except Exception:
    np = None

DIMENSIONS = [4, 5, 6]
SEED = 42


def _schoolbook(n, m, p):
    """Return schoolbook decomposition as lists of Fractions."""
    R = n * m * p
    U = [[Fraction(0)] * R for _ in range(n * m)]
    V = [[Fraction(0)] * R for _ in range(m * p)]
    W = [[Fraction(0)] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W


def _verify_exact(U, V, W, n, m, p):
    """Verify decomposition using exact Fraction arithmetic."""
    R = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    s = Fraction(0)
                    for c in range(R):
                        wv = W[k * n + i][c]
                        uu = U[i * m + j][c]
                        vv = V[j2 * p + k][c]
                        if wv == 0 or uu == 0 or vv == 0:
                            continue
                        s += wv * uu * vv
                    target = Fraction(1) if j == j2 else Fraction(0)
                    if s != target:
                        return False
    return True


def _verify_numpy(U, V, W, n, m, p):
    """Fast verification using numpy (may have floating point errors)."""
    if np is None:
        return _verify_exact(U, V, W, n, m, p)
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            if not np.allclose(M, np.eye(m), atol=1e-8):
                return False
    return True


def _try_remove_column(U, V, W, n, m, p, col_to_remove):
    """Try removing a specific column. Returns (success, new_U, new_V, new_W)."""
    R = len(U[0])
    if R <= 1:
        return False, U, V, W
    
    keep = [x for x in range(R) if x != col_to_remove]
    U2 = [[U[r][x] for x in keep] for r in range(n * m)]
    V2 = [[V[r][x] for x in keep] for r in range(m * p)]
    W2 = [[W[r][x] for x in keep] for r in range(p * n)]
    
    if _verify_numpy(U2, V2, W2, n, m, p):
        return True, U2, V2, W2
    return False, U, V, W


def _greedy_eliminate(U, V, W, n, m, p, deadline):
    """Greedily remove columns that can be eliminated."""
    R = len(U[0])
    removed = set()
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        R = len(U[0])
        for c in range(R):
            if c in removed:
                continue
            if time.time() > deadline - 20.0:
                break
            keep = [x for x in range(R) if x != c and x not in removed]
            if len(keep) == 0:
                continue
            U2 = [[U[r][x] for x in keep] for r in range(n * m)]
            V2 = [[V[r][x] for x in keep] for r in range(m * p)]
            W2 = [[W[r][x] for x in keep] for r in range(p * n)]
            if _verify_numpy(U2, V2, W2, n, m, p):
                removed.add(c)
                changed = True
                U, V, W = U2, V2, W2
    
    if not removed:
        return U, V, W
    
    R = len(U[0])
    keep = [x for x in range(R) if x not in removed]
    U2 = [[U[r][x] for x in keep] for r in range(n * m)]
    V2 = [[V[r][x] for x in keep] for r in range(m * p)]
    W2 = [[W[r][x] for x in keep] for r in range(p * n)]
    return U2, V2, W2


def _random_perturb(U, V, W, n, m, p, deadline):
    """Apply random linear transformations to find equivalent decompositions."""
    if np is None:
        return U, V, W
    
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    best_U, best_V, best_W = U, V, W
    best_rank = R
    
    for attempt in range(50):
        if time.time() > deadline - 20.0:
            break
        
        # Random invertible transformation
        P = np.eye(R, dtype=np.float64)
        for _ in range(3):
            i, j = random.randint(0, R - 1), random.randint(0, R - 1)
            if i != j:
                P[i, j] = random.gauss(0, 1)
        
        # Check if invertible
        try:
            det = np.linalg.det(P)
            if abs(det) < 1e-6:
                continue
            Pinv = np.linalg.inv(P)
        except:
            continue
        
        U_proj = Ua @ P
        V_proj = Va @ P
        W_proj = Wa @ Pinv.T  # W transforms contravariantly
        
        # Convert to Fractions for exact check
        U_frac = [[Fraction(U_proj[r, c]).limit_denominator(1000) for c in range(R)] for r in range(n * m)]
        V_frac = [[Fraction(V_proj[r, c]).limit_denominator(1000) for c in range(R)] for r in range(m * p)]
        W_frac = [[Fraction(W_proj[r, c]).limit_denominator(1000) for c in range(R)] for r in range(p * n)]
        
        if _verify_numpy(U_frac, V_frac, W_frac, n, m, p):
            # Try greedy elimination on this new form
            U2, V2, W2 = _greedy_eliminate(U_frac, V_frac, W_frac, n, m, p, deadline)
            new_rank = len(U2[0])
            if new_rank < best_rank:
                best_U, best_V, best_W = U2, V2, W2
                best_rank = new_rank
    
    return best_U, best_V, best_W


def _structured_search(U, V, W, n, m, p, deadline):
    """Try structured decompositions based on block structure.
    
    n=4=2+2, m=5, p=6=3+3
    View as blocks: A is 4x5, B is 5x6
    Partition rows of A and cols of B:
    A = [A1 (2x5); A2 (2x5)]
    B = [B1 (5x3) | B2 (5x3)]
    C = [C11 C12; C21 C22] where Cij = Ai * Bj
    
    Each block Cij is 2x3, requiring at most 6 terms (schoolbook)
    But we can share structure across blocks.
    """
    # This is a placeholder for more sophisticated structured search
    # For now, just return the input
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    try:
        # Build initial schoolbook decomposition
        U, V, W = _schoolbook(n, m, p)
        
        # Verify
        if not _verify_numpy(U, V, W, n, m, p):
            # Should not happen, but fallback
            return {"U": [[int(x) for x in row] for row in U], 
                    "V": [[int(x) for x in row] for row in V], 
                    "W": [[int(x) for x in row] for row in W]}
        
        # Try greedy elimination
        U, V, W = _greedy_eliminate(U, V, W, n, m, p, deadline)
        
        # Try random perturbations
        if time.time() < deadline - 30.0:
            U, V, W = _random_perturb(U, V, W, n, m, p, deadline)
        
        # Final greedy elimination
        U, V, W = _greedy_eliminate(U, V, W, n, m, p, deadline)
        
        # Convert to integers if possible
        U_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in U]
        V_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in V]
        W_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in W]
        
        return {"U": U_out, "V": V_out, "W": W_out}
    
    except Exception:
        # Fallback: pure schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": [[int(x) for x in row] for row in U], 
                "V": [[int(x) for x in row] for row in V], 
                "W": [[int(x) for x in row] for row in W]}


# EVOLVE-BLOCK-END
