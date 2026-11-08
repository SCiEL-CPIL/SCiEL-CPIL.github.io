"""Search for a rank-minimal decomposition of the 2x3x10 matrix-mult tensor."""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
TARGET_DEADLINE = 145.0  # safety margin under 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] == tensor target. Returns (ok, max_err)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Build full tensor: T[a,b,c] = sum_r Ua[a,r]*Va[b,r]*Wa[c,r]
    # a = i*m+j, b = j*p+k, c = k*n+i
    # For valid entries: i_a == i_c, j_a == j_b, k_b == k_c
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                e = abs(T[a,b,c] - 1.0)
                if e > max_err:
                    max_err = e
    return (max_err < tol), max_err

def _verify_zero_entries(U, V, W, n, m, p, tol=1e-9):
    """Check that zero entries of the tensor are actually zero."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    nm, mp, pn = n*m, m*p, p*n
    max_err = 0.0
    # Check all zero entries (skip valid ones)
    valid_set = set()
    for i in range(n):
        for k in range(p):
            for j in range(m):
                valid_set.add((i*m+j, j*p+k, k*n+i))
    # Sample zero entries
    rng = random.Random(42)
    checked = 0
    while checked < 2000:
        a = rng.randint(0, nm-1)
        b = rng.randint(0, mp-1)
        c = rng.randint(0, pn-1)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        i_c = c % n
        k_c = c // n
        if i_a == i_c and j_a == j_b and k_b == k_c:
            continue
        e = abs(T[a,b,c])
        if e > max_err:
            max_err = e
        checked += 1
    return (max_err < tol), max_err

def _current_rank55(n, m, p):
    """Build the known rank-55 solution."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # M1
        U[0][col] += 1; U[4][col] += 1
        V[k0][col] += 1; V[10+k1][col] += 1
        W[k0][col] += 1; W[k1+2][col] += 1
        col += 1
        # M2
        U[3][col] += 1; U[4][col] += 1
        V[k0][col] += 1
        W[k0+2][col] += 1; W[k1+2][col] -= 1
        col += 1
        # M3
        U[0][col] += 1
        V[k1][col] += 1; V[10+k1][col] -= 1
        W[k1][col] += 1; W[k1+2][col] += 1
        col += 1
        # M4
        U[4][col] += 1
        V[10+k0][col] += 1; V[k0][col] -= 1
        W[k0][col] += 1; W[k0+2][col] += 1
        col += 1
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[10+k1][col] += 1
        W[k0][col] -= 1; W[k1][col] += 1
        col += 1
        # M6
        U[3][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[k1][col] += 1
        W[k1+2][col] += 1
        col += 1
        # M7
        U[1][col] += 1; U[4][col] -= 1
        V[10+k0][col] += 1; V[10+k1][col] += 1
        W[k0][col] += 1
        col += 1
        # W00
        U[2][col] += 1; V[20+k0][col] += 1; W[k0][col] += 1
        col += 1
        # W01
        U[2][col] += 1; V[20+k1][col] += 1; W[k1][col] += 1
        col += 1
        # W10
        U[5][col] += 1; V[20+k0][col] += 1; W[k0+2][col] += 1
        col += 1
        # W11
        U[5][col] += 1; V[20+k1][col] += 1; W[k1+2][col] += 1
        col += 1
    return {"U": U, "V": V, "W": W}

def _solve_W(U, V, n, m, p, R):
    """Given U and V, solve for W to satisfy the tensor identity."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    W = np.zeros((p*n, R))
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, R))
            for j in range(m):
                M[j] = Ua[i*m+j] * Va[j*p+k]
            b = np.ones(m)
            try:
                w, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
                if residuals.size > 0 and residuals[0] > 1e-8:
                    return None
                W[k*n+i] = w
            except:
                return None
    return W.tolist()

def _try_rank_structured(R, n, m, p, rng, max_iters=5000):
    """Try to find a valid rank-R decomposition with structured U, V."""
    nm = n * m
    mp = m * p
    pn = p * n
    vals = [-1, -0.5, 0.5, 1]
    
    for it in range(max_iters):
        if time.time() > TARGET_DEADLINE - 5:
            break
        # Random sparse U, V with varying density
        density = 0.12 + 0.06 * (it % 3)
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        for r_ in range(nm):
            for c_ in range(R):
                if rng.random() < density:
                    U[r_, c_] = rng.choice(vals)
        for r_ in range(mp):
            for c_ in range(R):
                if rng.random() < density:
                    V[r_, c_] = rng.choice(vals)
        
        # Solve for W using the same approach but with numpy arrays
        Ua = U
        Va = V
        W = np.zeros((pn, R))
        valid = True
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j] = Ua[i*m+j] * Va[j*p+k]
                b = np.ones(m)
                try:
                    w, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
                    if residuals.size > 0 and residuals[0] > 1e-8:
                        valid = False
                        break
                    W[k*n+i] = w
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        ok, err = _verify(U_list, V_list, W_list, n, m, p)
        if ok:
            ok2, err2 = _verify_zero_entries(U_list, V_list, W_list, n, m, p)
            if ok2:
                return U_list, V_list, W_list
    return None

def _try_column_elimination(U, V, W, n, m, p, rng, max_attempts=500):
    """Try to eliminate one column from the decomposition by solving a linear system."""
    R = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    nm, mp, pn = n*m, m*p, p*n
    
    for attempt in range(max_attempts):
        if time.time() > TARGET_DEADLINE - 5:
            break
        col = rng.randint(0, R-1)
        # Skip if column is already zero
        if np.all(Ua[:,col] == 0) or np.all(Va[:,col] == 0) or np.all(Wa[:,col] == 0):
            continue
        
        # Try to zero out column col by adjusting other columns
        # For each valid (i,j,k) triple, we need:
        # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # After removing col: sum_{r!=col} ... = 1 - U[i*m+j,col]*V[j*p+k,col]*W[k*n+i,col]
        # We can adjust one other column to compensate.
        
        # Find a column r2 that shares structure
        found = False
        for r2 in range(R):
            if r2 == col:
                continue
            # Check if we can adjust column r2 to absorb column col
            # Need: for all (i,j,k) valid:
            #   delta * U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2] = -U[i*m+j,col]*V[j*p+k,col]*W[k*n+i,col]
            # This requires U[i*m+j,col]*V[j*p+k,col]*W[k*n+i,col] / (U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2]) to be constant
            # Let's check with a few samples
            ratios = []
            valid_count = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        a = i*m+j
                        b = j*p+k
                        c = k*n+i
                        num = Ua[a,col]*Va[b,col]*Wa[c,col]
                        den = Ua[a,r2]*Va[b,r2]*Wa[c,r2]
                        if abs(den) > 1e-12:
                            ratios.append(-num/den)
                            valid_count += 1
            if valid_count < 3:
                continue
            if max(ratios) - min(ratios) < 1e-6:
                # All ratios are the same! We can eliminate col by adjusting r2
                delta = ratios[0]
                Ua[:,r2] += delta * Ua[:,col]
                Va[:,r2] += delta * Va[:,col]
                Wa[:,r2] += delta * Wa[:,col]
                Ua[:,col] = 0
                Va[:,col] = 0
                Wa[:,col] = 0
                found = True
                break
        
        if found:
            # Verify
            U_new = Ua.tolist()
            V_new = Va.tolist()
            W_new = Wa.tolist()
            ok, err = _verify(U_new, V_new, W_new, n, m, p)
            ok2, err2 = _verify_zero_entries(U_new, V_new, W_new, n, m, p)
            if ok and ok2:
                return U_new, V_new, W_new
    return None

def _try_perturbation_search(U0, V0, W0, n, m, p, rng, max_iters=3000):
    """Start from the rank-55 solution and try to find a lower-rank one via perturbation."""
    R0 = len(U0[0])
    
    # First try column elimination on the rank-55 solution
    result = _try_column_elimination(U0, V0, W0, n, m, p, rng, max_attempts=200)
    if result is not None:
        U, V, W = result
        ok, err = _verify(U, V, W, n, m, p)
        ok2, err2 = _verify_zero_entries(U, V, W, n, m, p)
        if ok and ok2:
            return U, V, W
    
    # Try random structured search for rank 54
    if time.time() < TARGET_DEADLINE - 15:
        result = _try_rank_structured(54, n, m, p, rng, max_iters=3000)
        if result is not None:
            U, V, W = result
            ok, err = _verify(U, V, W, n, m, p)
            ok2, err2 = _verify_zero_entries(U, V, W, n, m, p)
            if ok and ok2:
                return U, V, W
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rng = random.Random(12345)
    
    # Start with the known rank-55 solution
    best = _current_rank55(n, m, p)
    best_rank = 55
    
    # Verify the seed solution
    ok, err = _verify(best["U"], best["V"], best["W"], n, m, p)
    if not ok:
        # Fallback: build a trivial rank-60 solution
        U = [[0]*60 for _ in range(n*m)]
        V = [[0]*60 for _ in range(m*p)]
        W = [[0]*60 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = 60
    
    # Try column elimination on rank-55 to get rank-54
    if time.time() < TARGET_DEADLINE - 20:
        elim_result = _try_column_elimination(best["U"], best["V"], best["W"], n, m, p, rng, max_attempts=300)
        if elim_result is not None:
            U, V, W = elim_result
            ok2, err2 = _verify(U, V, W, n, m, p)
            ok3, err3 = _verify_zero_entries(U, V, W, n, m, p)
            if ok2 and ok3:
                best = {"U": U, "V": V, "W": W}
                best_rank = 54
    
    # Try random structured search for lower ranks
    for target_rank in [54, 53, 52, 51, 50]:
        if best_rank <= target_rank:
            continue
        if time.time() > TARGET_DEADLINE - 15:
            break
        result = _try_rank_structured(target_rank, n, m, p, rng, max_iters=4000)
        if result is not None:
            U, V, W = result
            ok2, err2 = _verify(U, V, W, n, m, p)
            ok3, err3 = _verify_zero_entries(U, V, W, n, m, p)
            if ok2 and ok3:
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    # Deadline guard
    if time.time() > TARGET_DEADLINE - 3:
        return best
    
    return best

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    ok, err = _verify(res["U"], res["V"], res["W"], n, m, p)
    print(f"Rank: {len(res['U'][0])}, Valid: {ok}, MaxErr: {err:.2e}")
