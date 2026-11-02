"""Matrix multiplication 3x3 by 3x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 6]
DEADLINE = 130  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 18, 18

    # Build target tensor T[a, b, c] where a in [0,9), b in [0,18), c in [0,18)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook rank = 54
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify(U, V, W):
        """Check if U,V,W exactly decompose T. Returns True if valid."""
        # T_reconstructed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum for efficiency
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # But this is expensive for large rank. Use matrix form:
        # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
        # Equivalent: reshape
        r = U.shape[1]
        # Build M[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = sum_r (U[a,r]*V[b,r]) * W[c,r]
        # Let M1[a,b] = sum_r U[a,r]*V[b,r] = U @ V.T  (nm x mp)
        # Then T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is a 3-tensor, not just M1.
        # Use: T = einsum
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_rec, T, atol=1e-9)

    def rank_of_decomp(U, V, W):
        return U.shape[1]

    best_rank = rank_schoolbook
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}

    start_time = time.time()

    # Strategy: Try to find rank < 54 by search
    # The lower bound is 36. Try ranks from 36 upward.
    # Use randomized search with local refinement.

    # For a given rank r, we need U (nm x r), V (mp x r), W (pn x r)
    # such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    #
    # Approach: Start from schoolbook and try to merge columns.
    # Or: Random initialization + gradient-like refinement.
    #
    # Better approach: Use the fact that T is sparse. Each of the 54
    # "unit" tensors can potentially be combined.
    #
    # Let's try a structured search:
    # For rank r, initialize U,V,W randomly, then use iterative refinement.
    #
    # Actually, let's try a different approach:
    # The tensor T has 54 non-zero entries. We can think of each as a "term".
    # A rank-r decomposition means we can write T as sum of r rank-1 tensors.
    #
    # Key insight: For matrix multiplication, there are known results.
    # For 3x3 x 3x3, the rank is 23 (Strassen-like). For 3x3 x 3x6,
    # we can potentially do better than 54.
    #
    # Let me try: Use the 3x3x3 result (rank 23) and extend.
    # Actually, 3x3 x 3x6 = two 3x3 x 3x3 multiplications.
    # If 3x3x3 has rank 23, then 3x3x3x6 might have rank <= 46?
    # But we can do better by sharing structure.
    #
    # Let's try a computational search for small ranks.

    def try_rank(r, max_iters=500, seed=0):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() - start_time > DEADLINE - 10:
            return None
        rng = np.random.RandomState(seed)
        best_err = np.inf
        best_U, best_V, best_W = None, None, None

        for trial in range(3):
            # Initialize
            U = rng.randn(nm, r)
            V = rng.randn(mp, r)
            W = rng.randn(pn, r)

            # Normalize to avoid blowup
            for col in range(r):
                norm = np.sqrt(np.sum(U[:, col]**2) + np.sum(V[:, col]**2) + np.sum(W[:, col]**2))
                if norm > 1e-10:
                    U[:, col] /= norm
                    V[:, col] /= norm
                    W[:, col] /= norm

            err_history = []
            for it in range(max_iters):
                if time.time() - start_time > DEADLINE - 10:
                    return None
                # Compute current error
                T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                err = np.sum((T_rec - T)**2)
                err_history.append(err)

                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if err < 1e-12:
                    break

                # Gradient descent on U, V, W
                # dE/dU[a,r] = -2 * sum_{b,c} (T[a,b,c] - T_rec[a,b,c]) * V[b,r] * W[c,r]
                # = -2 * (T - T_rec) contracted with V[r,:] and W[r,:]
                #
                # For efficiency, compute residuals
                R = T - T_rec  # nm x mp x pn
                # dE/dU[a,r] = -2 * sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
                # = -2 * (R @ (V[r,:] outer W[r,:]) summed over b,c)
                # Actually: for fixed a,r: sum_{b,c} R[a,b,c] V[b,r] W[c,r]
                # = sum_b V[b,r] * sum_c R[a,b,c] W[c,r]
                # = sum_b V[b,r] * (R[a,b,:] dot W[:,r])
                #
                # Let's compute: RW = R @ W  -> but R is 3D
                # RW[a,b,r] = sum_c R[a,b,c] W[c,r]
                # Then dU[a,r] = -2 * sum_b V[b,r] * RW[a,b,r]
                #
                # Similarly for V and W.

                # RW[a,b,r] = sum_c R[a,b,c] W[c,r]
                RW = np.einsum('abc,cr->abr', R, W, optimize=True)  # nm x mp x r
                # dU[a,r] = -2 * sum_b V[b,r] * RW[a,b,r]
                dU = -2.0 * np.einsum('br,abr->ar', V, RW, optimize=True)

                # RU[a,c,r] = sum_b R[a,b,c] V[b,r]
                RU = np.einsum('abc,br->acr', R, V, optimize=True)  # nm x pn x r
                # dW[c,r] = -2 * sum_a U[a,r] * RU[a,c,r]
                dW = -2.0 * np.einsum('ar,acr->cr', U, RU, optimize=True)

                # RV[b,c,r] = sum_a R[a,b,c] U[a,r]
                RV = np.einsum('abc,ar->bcr', R, U, optimize=True)  # mp x pn x r
                # dV[b,r] = -2 * sum_c W[c,r] * RV[b,c,r]
                dV = -2.0 * np.einsum('cr,bcr->br', W, RV, optimize=True)

                # Adaptive learning rate
                lr = 0.01 / (1.0 + it * 0.01)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                # Re-normalize
                for col in range(r):
                    norm = np.sqrt(np.sum(U[:, col]**2) + np.sum(V[:, col]**2) + np.sum(W[:, col]**2))
                    if norm > 1e-10:
                        U[:, col] /= norm
                        V[:, col] /= norm
                        W[:, col] /= norm

            if best_err < 1e-12:
                break

        if best_U is not None and best_err < 1e-9:
            # Verify with integer/half-integer check
            return best_U, best_V, best_W, best_err
        return None

    # Try ranks from lower bound upward
    # Lower bound: max(nm, mp, pn) = 18, but more realistically ~36
    # Let's try a range
    for r in range(36, 55):
        if time.time() - start_time > DEADLINE - 15:
            break
        result = try_rank(r, max_iters=300, seed=r)
        if result is not None:
            U, V, W, err = result
            if verify(U, V, W):
                best_rank = r
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                break
        # Also try with different seeds
        for s in range(1, 4):
            if time.time() - start_time > DEADLINE - 15:
                break
            result = try_rank(r, max_iters=200, seed=r * 100 + s)
            if result is not None:
                U, V, W, err = result
                if verify(U, V, W):
                    best_rank = r
                    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                    break
        if best_rank < 54:
            break

    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_result

    # Final: ensure we return valid result
    return best_result
