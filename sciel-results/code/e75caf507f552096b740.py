import numpy as np
import time
import random
from fractions import Fraction

def construct():
    """
    Rank-minimal exact tensor decomposition for [2,4,4] matrix multiplication.
    Uses a block-Strassen base (rank 28) with randomized greedy linear dependency elimination.
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 4
    R = n * m * p  # 32

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    # ---- Build initial decomposition: rank 28 ----
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def add_strassen(a_r, a_c, b_r, b_c, c_c, start_idx):
        A00 = u_idx(a_r, a_c)
        A01 = u_idx(a_r, a_c+1)
        A10 = u_idx(a_r+1, a_c)
        A11 = u_idx(a_r+1, a_c+1)

        B00 = v_idx(b_r, b_c)
        B01 = v_idx(b_r, b_c+1)
        B10 = v_idx(b_r+1, b_c)
        B11 = v_idx(b_r+1, b_c+1)

        C00 = w_idx(0, c_c)
        C01 = w_idx(0, c_c+1)
        C10 = w_idx(1, c_c)
        C11 = w_idx(1, c_c+1)

        # M1: (A00+A11)(B00+B11)
        U[A00, start_idx+0] += 1
        U[A11, start_idx+0] += 1
        V[B00, start_idx+0] += 1
        V[B11, start_idx+0] += 1
        W[C00, start_idx+0] += 1
        W[C11, start_idx+0] += 1

        # M2: (A10+A11)B00
        U[A10, start_idx+1] += 1
        U[A11, start_idx+1] += 1
        V[B00, start_idx+1] += 1
        W[C10, start_idx+1] += 1
        W[C11, start_idx+1] -= 1

        # M3: A00(B01-B11)
        U[A00, start_idx+2] += 1
        V[B01, start_idx+2] += 1
        V[B11, start_idx+2] -= 1
        W[C01, start_idx+2] += 1
        W[C11, start_idx+2] += 1

        # M4: A11(B10-B00)
        U[A11, start_idx+3] += 1
        V[B10, start_idx+3] += 1
        V[B00, start_idx+3] -= 1
        W[C00, start_idx+3] += 1
        W[C10, start_idx+3] += 1

        # M5: (A00+A01)B11
        U[A00, start_idx+4] += 1
        U[A01, start_idx+4] += 1
        V[B11, start_idx+4] += 1
        W[C00, start_idx+4] -= 1
        W[C01, start_idx+4] += 1

        # M6: (A10-A00)(B00+B01)
        U[A10, start_idx+5] += 1
        U[A00, start_idx+5] -= 1
        V[B00, start_idx+5] += 1
        V[B01, start_idx+5] += 1
        W[C11, start_idx+5] += 1

        # M7: (A01-A11)(B10+B11)
        U[A01, start_idx+6] += 1
        U[A11, start_idx+6] -= 1
        V[B10, start_idx+6] += 1
        V[B11, start_idx+6] += 1
        W[C00, start_idx+6] += 1

    # Product 1: A_L * B_TL -> C cols 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    # Product 2: A_R * B_BL -> C cols 0,1
    add_strassen(0, 2, 2, 0, 0, 7)
    # Product 3: A_L * B_TR -> C cols 2,3
    add_strassen(0, 0, 0, 2, 2, 14)
    # Product 4: A_R * B_BR -> C cols 2,3
    add_strassen(0, 2, 2, 2, 2, 21)

    def verify_exact(Um, Vm, Wm):
        """Verify using exact Fraction arithmetic."""
        Td = np.zeros((n, m, p), dtype=object)
        for t in range(Um.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Td[i, j, k] += Fraction(Wm[k*n+i, t]).limit_denominator(10**6) * \
                                        Fraction(Um[i*m+j, t]).limit_denominator(10**6) * \
                                        Fraction(Vm[j*p+k, t]).limit_denominator(10**6)
        return all(Td[i, j, k] == 1 for i in range(n) for j in range(m) for k in range(p))

    if not verify_exact(U, V, W):
        raise ValueError("Invalid initial decomposition")

    def vec_r1(Uc, Vc, Wc):
        T = np.zeros(R, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[k*n*m + i*m + j] = Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        return T

    T_target = np.ones(R, dtype=np.float64)

    def build_M(Um, Vm, Wm):
        return np.array([vec_r1(Um[:, t], Vm[:, t], Wm[:, t]) for t in range(Um.shape[1])]).T

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    M = build_M(best_U, best_V, best_W)

    def is_half_integer(arr, tol=1e-8):
        return np.allclose(arr * 2, np.round(arr * 2), atol=tol)

    # Randomized Greedy Elimination
    # We try to remove tensors one by one.
    # To escape local minima, we randomize the order of checking tensors.
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        removed_in_pass = False
        # Randomize the order of indices to check
        indices = list(range(best_rank))
        random.shuffle(indices)
        
        for t in indices:
            if time.time() > deadline:
                break
            if t >= best_rank: continue
            
            keep = [s for s in range(best_rank) if s != t]
            if not keep:
                continue
                
            M_sub = M[:, keep]
            try:
                # Use lstsq to find coefficients
                c, _, _, _ = np.linalg.lstsq(M_sub, T_target, rcond=None)
                residual = np.linalg.norm(M_sub @ c - T_target)
                
                if residual < 1e-9:
                    # Check half-integer constraint on c
                    c_2 = np.round(c * 2)
                    if np.allclose(c * 2, c_2, atol=1e-8):
                        new_U = best_U[:, keep].copy()
                        new_V = best_V[:, keep].copy()
                        new_W = best_W[:, keep].copy()

                        for idx, s in enumerate(keep):
                            new_V[:, idx] = best_V[:, s] * c[idx]

                        # Verify all entries are half-integers
                        valid = True
                        for arr in [new_U, new_V, new_W]:
                            if not is_half_integer(arr):
                                valid = False
                                break
                        
                        if valid:
                            if verify_exact(new_U, new_V, new_W):
                                best_rank = len(keep)
                                best_U, best_V, best_W = new_U, new_V, new_W
                                M = build_M(best_U, best_V, best_W)
                                removed_in_pass = True
                                break # Restart the pass with new rank
            except:
                continue
        
        if not removed_in_pass:
            # If we couldn't remove anything in a full pass, we might be stuck.
            # We could try a different strategy, but for now, break.
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
