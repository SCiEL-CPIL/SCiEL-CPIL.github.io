import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    def _compute_error(U, V, W):
        """Compute the error tensor E where E[i,a,k] = sum_mu U[i,a,mu]V[a,k,mu]W[k,i,mu] - 1."""
        rank = len(U[0])
        # Precompute U and V products for speed?
        # U: 6 x r, V: 24 x r, W: 16 x r
        # We need sum_mu U[ia, mu] V[ak, mu] W[ki, mu]
        # Let's just loop for now, it's small.
        error = np.zeros((n, m, p), dtype=int)
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_idx = i * m + a
                    v_idx = a * p + k
                    w_idx = k * n + i
                    for mu in range(rank):
                        if U[u_idx][mu] != 0 and V[v_idx][mu] != 0 and W[w_idx][mu] != 0:
                            acc += U[u_idx][mu] * V[v_idx][mu] * W[w_idx][mu]
                    error[i, a, k] = acc - 1
        return error

    def _search_for_rank(r, max_restarts=200, max_steps=5000):
        """
        Local search for a decomposition of rank r with entries in {-1, 0, 1}.
        """
        if not _elapsed_ok():
            return None
        
        U = np.zeros((nm, r), dtype=int)
        V = np.zeros((mp, r), dtype=int)
        W = np.zeros((pn, r), dtype=int)

        best_error = float('inf')
        
        for restart in range(max_restarts):
            if not _elapsed_ok():
                return None
            
            # Random initialization
            for idx in range(nm):
                for mu in range(r):
                    U[idx, mu] = random.choice([-1, 0, 1])
            for idx in range(mp):
                for mu in range(r):
                    V[idx, mu] = random.choice([-1, 0, 1])
            for idx in range(pn):
                for mu in range(r):
                    W[idx, mu] = random.choice([-1, 0, 1])
            
            error = _compute_error(U, V, W)
            curr_error = np.sum(error**2)
            
            if curr_error == 0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            no_improve_count = 0
            for step in range(max_steps):
                if not _elapsed_ok():
                    break
                
                # Pick a random variable to flip
                var_type = random.choice(['U', 'V', 'W'])
                if var_type == 'U':
                    idx = random.randint(0, nm - 1)
                    mu = random.randint(0, r - 1)
                    old_val = U[idx, mu]
                    new_val = 1 if old_val == -1 else -1 if old_val == 1 else 0
                    if new_val == old_val: continue
                    U[idx, mu] = new_val
                    new_error = _compute_error(U, V, W)
                    new_curr_error = np.sum(new_error**2)
                    if new_curr_error < curr_error:
                        curr_error = new_curr_error
                        error = new_error
                        no_improve_count = 0
                        if curr_error == 0:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    else:
                        U[idx, mu] = old_val
                        no_improve_count += 1
                elif var_type == 'V':
                    idx = random.randint(0, mp - 1)
                    mu = random.randint(0, r - 1)
                    old_val = V[idx, mu]
                    new_val = 1 if old_val == -1 else -1 if old_val == 1 else 0
                    if new_val == old_val: continue
                    V[idx, mu] = new_val
                    new_error = _compute_error(U, V, W)
                    new_curr_error = np.sum(new_error**2)
                    if new_curr_error < curr_error:
                        curr_error = new_curr_error
                        error = new_error
                        no_improve_count = 0
                        if curr_error == 0:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    else:
                        V[idx, mu] = old_val
                        no_improve_count += 1
                else:
                    idx = random.randint(0, pn - 1)
                    mu = random.randint(0, r - 1)
                    old_val = W[idx, mu]
                    new_val = 1 if old_val == -1 else -1 if old_val == 1 else 0
                    if new_val == old_val: continue
                    W[idx, mu] = new_val
                    new_error = _compute_error(U, V, W)
                    new_curr_error = np.sum(new_error**2)
                    if new_curr_error < curr_error:
                        curr_error = new_curr_error
                        error = new_error
                        no_improve_count = 0
                        if curr_error == 0:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    else:
                        W[idx, mu] = old_val
                        no_improve_count += 1
                
                if no_improve_count > 1000:
                    break

        return None

    # Try to find a decomposition with rank < 44
    # Current best is 40. Try 36, 38, 40.
    for r in [36, 38, 40]:
        if not _elapsed_ok():
            break
        result = _search_for_rank(r, max_restarts=50, max_steps=2000)
        if result:
            if _validate(result):
                return result

    # Fallback to schoolbook if search fails
    return _schoolbook()

# EVOLVE-BLOCK-END
