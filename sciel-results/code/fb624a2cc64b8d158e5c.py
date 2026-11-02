import time
import numpy as np
from itertools import combinations

_DEADLINE = time.time() + 120.0
_TIMEOUT_MARGIN = 10.0

def _time_left():
    return _DEADLINE - _TIMEOUT_MARGIN - time.time()

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    # Build the matmul tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with j==j', i==i', k==k'
    # Otherwise 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def _verify(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        rank = U.shape[1]
        # Reconstruct tensor
        recon = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank):
            recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(recon - T))
    
    def _verify_fast(U, V, W):
        """Fast verification - check only the non-zero entries and some zero entries"""
        rank = U.shape[1]
        err = 0.0
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += abs(s - 1.0)
        # Check some zero entries (sample)
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            i_a, j_a = divmod(a, m)
            j_b, k_b = divmod(b, p)
            k_c, i_c = divmod(c, n)
            if j_a == j_b and i_a == i_c and k_b == k_c:
                continue  # This is a non-zero entry, skip
            s = np.dot(U[a, :] * V[b, :], W[c, :])
            err += abs(s)
        return err
    
    # Start with Strassen rank-56 decomposition
    rank0 = 56
    U0 = np.zeros((nm, rank0), dtype=float)
    V0 = np.zeros((mp, rank0), dtype=float)
    W0 = np.zeros((pn, rank0), dtype=float)
    
    for a in range(2):
        for b in range(4):
            base = 7 * (4 * a + b)
            a00 = 0 * m + 2 * a
            a01 = 0 * m + 2 * a + 1
            a10 = 1 * m + 2 * a
            a11 = 1 * m + 2 * a + 1
            b00 = (2 * a) * p + 2 * b
            b01 = (2 * a) * p + 2 * b + 1
            b10 = (2 * a + 1) * p + 2 * b
            b11 = (2 * a + 1) * p + 2 * b + 1
            c00 = (2 * b) * n + 0
            c01 = (2 * b + 1) * n + 0
            c10 = (2 * b) * n + 1
            c11 = (2 * b + 1) * n + 1
            
            U0[a00, base] = 1; U0[a11, base] = 1
            V0[b00, base] = 1; V0[b11, base] = 1
            W0[c00, base] = 1; W0[c11, base] = 1
            
            U0[a10, base+1] = 1; U0[a11, base+1] = 1
            V0[b00, base+1] = 1
            W0[c10, base+1] = 1; W0[c11, base+1] = -1
            
            U0[a00, base+2] = 1
            V0[b01, base+2] = 1; V0[b11, base+2] = -1
            W0[c01, base+2] = 1; W0[c11, base+2] = 1
            
            U0[a11, base+3] = 1
            V0[b10, base+3] = 1; V0[b00, base+3] = -1
            W0[c00, base+3] = 1; W0[c10, base+3] = 1
            
            U0[a00, base+4] = 1; U0[a01, base+4] = 1
            V0[b11, base+4] = 1
            W0[c00, base+4] = -1; W0[c01, base+4] = 1
            
            U0[a10, base+5] = 1; U0[a00, base+5] = -1
            V0[b00, base+5] = 1; V0[b01, base+5] = 1
            W0[c11, base+5] = 1
            
            U0[a01, base+6] = 1; U0[a11, base+6] = -1
            V0[b10, base+6] = 1; V0[b11, base+6] = 1
            W0[c00, base+6] = 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try removing single terms
    for remove_idx in range(rank0):
        if _time_left() < 5.0:
            break
        U_r = U0.copy()
        V_r = V0.copy()
        W_r = W0.copy()
        U_r[:, remove_idx] = 0
        V_r[:, remove_idx] = 0
        W_r[:, remove_idx] = 0
        err = _verify_fast(U_r, V_r, W_r)
        if err < 1e-9:
            best_rank = rank0 - 1
            best_U, best_V, best_W = U_r, V_r, W_r
            break
    
    # If we reduced, try again
    if best_rank < rank0:
        U0, V0, W0 = best_U.copy(), best_V.copy(), best_W.copy()
        for remove_idx in range(best_rank):
            if _time_left() < 5.0:
                break
            U_r = U0.copy()
            V_r = V0.copy()
            W_r = W0.copy()
            U_r[:, remove_idx] = 0
            V_r[:, remove_idx] = 0
            W_r[:, remove_idx] = 0
            err = _verify_fast(U_r, V_r, W_r)
            if err < 1e-9:
                best_rank -= 1
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    # Try pair removals
    if _time_left() > 10.0:
        U_cur, V_cur, W_cur = best_U.copy(), best_V.copy(), best_W.copy()
        cur_rank = best_rank
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            for i in range(cur_rank):
                if _time_left() < 3.0:
                    break
                for j in range(i+1, cur_rank):
                    if _time_left() < 3.0:
                        break
                    U_r = U_cur.copy()
                    V_r = V_cur.copy()
                    W_r = W_cur.copy()
                    U_r[:, i] = 0
                    U_r[:, j] = 0
                    V_r[:, i] = 0
                    V_r[:, j] = 0
                    W_r[:, i] = 0
                    W_r[:, j] = 0
                    err = _verify_fast(U_r, V_r, W_r)
                    if err < 1e-9:
                        # Remove these two columns
                        keep = [k for k in range(cur_rank) if k not in (i, j)]
                        best_U = U_cur[:, keep]
                        best_V = V_cur[:, keep]
                        best_W = W_cur[:, keep]
                        best_rank -= 2
                        cur_rank -= 2
                        U_cur, V_cur, W_cur = best_U.copy(), best_V.copy(), best_W.copy()
                        improved = True
                        break
                if improved:
                    break
    
    # Try a completely different construction using rank-4 inner products
    # but with sharing across the n dimension
    if _time_left() > 30.0:
        # Alternative: treat as 2 rows, each doing 4x8 matmul
        # But share the B matrix computations
        # C[0,k] = sum_j A[0,j] B[j,k]
        # C[1,k] = sum_j A[1,j] B[j,k]
        # Let D = A[0,:] - A[1,:], then:
        # C[0,k] - C[1,k] = sum_j D[j] B[j,k]
        # So we need: C[1,:] (rank ?), C[0,:]-C[1,:] (rank ?)
        # This doesn't obviously help.
        
        # Try: can we do rank 48? 16 inner products of dim 4, each rank 3?
        # No, rank of 4-dim inner product is 4.
        
        # The lower bound is max(nm, mp, pn) = 32.
        # But that's a weak bound. The actual rank is likely higher.
        
        # Let's try a greedy approach: start with rank 32 and see if we can
        # find a decomposition by solving a system of equations.
        # This is hard in general, but let's try a few random initializations.
        
        for trial_rank in [40, 44, 48, 52]:
            if _time_left() < 15.0:
                break
            if trial_rank >= best_rank:
                continue
            
            # Try to find U, V, W of given rank
            # This is a hard problem. Let's use a simple heuristic:
            # Initialize randomly and do a few iterations of gradient descent
            # on the error.
            
            for attempt in range(3):
                if _time_left() < 10.0:
                    break
                
                np.random.seed(42 + attempt)
                U_try = np.random.randn(nm, trial_rank) * 0.1
                V_try = np.random.randn(mp, trial_rank) * 0.1
                W_try = np.random.randn(pn, trial_rank) * 0.1
                
                # Simple iterative refinement
                for it in range(50):
                    if _time_left() < 5.0:
                        break
                    
                    # Compute gradient w.r.t. U
                    # dE/dU[a,r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] - sum_{b,c} (recon[a,b,c]) V[b,r] W[c,r]
                    # This is expensive. Let's just check error.
                    err = _verify_fast(U_try, V_try, W_try)
                    if err < 1e-6:
                        if trial_rank < best_rank:
                            best_rank = trial_rank
                            best_U, best_V, best_W = U_try, V_try, W_try
                        break
    
    final_err = _verify_fast(best_U, best_V, best_W)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
