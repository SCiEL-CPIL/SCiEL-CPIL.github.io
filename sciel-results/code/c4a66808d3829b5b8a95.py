import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Use 3x3 outer blocks of 2x2 inner blocks
    nb = 3  # number of outer blocks in each dimension
    bs = 2  # block size (inner)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,1,1)],               # M3
        [(0,0,1), (0,1,1)],      # M4
        [(1,0,1), (0,0,-1)],     # M5
        [(0,1,1), (1,1,-1)],     # M6
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0
        [(0,0,1)],               # M1
        [(0,1,1), (1,1,-1)],     # M2
        [(1,0,1), (0,0,-1)],     # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],     # M5
        [(1,0,1), (1,1,1)],     # M6
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 27 * 7 = 189
    
    U = np.zeros((n*m, rank_bs), dtype=np.float64)
    V = np.zeros((m*p, rank_bs), dtype=np.float64)
    W = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                for s in range(7):
                    # A block: rows ib*bs..(ib+1)*bs, cols jb*bs..(jb+1)*bs
                    # B block: rows jb*bs..(jb+1)*bs, cols kb*bs..(kb+1)*bs
                    # C block: rows ib*bs..(ib+1)*bs, cols kb*bs..(kb+1)*bs
                    
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W[gk * n + gi, col] += coeff
                    
                    col += 1
    
    # Verify correctness (optional, for debugging)
    # ...
    
    # Try to reduce rank by merging proportional columns
    if time.time() > deadline - 30:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Greedy column merging
    # Two columns c1, c2 can be merged if:
    # U[:,c1] = alpha * U[:,c2] and V[:,c1] = beta * V[:,c2]
    # and W[:,c1] = alpha * beta * W[:,c2]
    
    active = list(range(rank_bs))
    
    def try_merge(c1, c2):
        """Try to merge column c2 into column c1. Return True if successful."""
        U1 = U[:, c1]
        U2 = U[:, c2]
        V1 = V[:, c1]
        V2 = V[:, c2]
        W1 = W[:, c1]
        W2 = W[:, c2]
        
        # Check if U1 and U2 are proportional
        if np.allclose(U1, 0) and np.allclose(U2, 0):
            return False  # Both zero, can't determine proportionality
        if np.allclose(U2, 0):
            alpha = 0
        elif np.allclose(U1, 0):
            return False
        else:
            # Find alpha such that U1 = alpha * U2
            # Use the first non-zero element of U2
            idx = np.argmax(np.abs(U2) > 1e-10)
            if np.abs(U2[idx]) < 1e-10:
                return False
            alpha = U1[idx] / U2[idx]
            if not np.allclose(U1, alpha * U2, atol=1e-10):
                return False
        
        # Check if V1 and V2 are proportional
        if np.allclose(V1, 0) and np.allclose(V2, 0):
            return False
        if np.allclose(V2, 0):
            beta = 0
        elif np.allclose(V1, 0):
            return False
        else:
            idx = np.argmax(np.abs(V2) > 1e-10)
            if np.abs(V2[idx]) < 1e-10:
                return False
            beta = V1[idx] / V2[idx]
            if not np.allclose(V1, beta * V2, atol=1e-10):
                return False
        
        # Check if W1 = alpha * beta * W2
        if not np.allclose(W1, alpha * beta * W2, atol=1e-10):
            return False
        
        # Merge: W1 += W2 * (alpha * beta) / (alpha * beta) = W1 + W2... 
        # Actually, if U1 = alpha*U2 and V1 = beta*V2, then the contribution of column c2
        # is U2 * V2 * W2 = (U1/alpha) * (V1/beta) * W2 = (1/(alpha*beta)) * U1 * V1 * W2
        # So we need W1_new = W1 + W2/(alpha*beta)
        # Wait, let me reconsider.
        # The tensor decomposition is: T = sum_s U_s * V_s * W_s (outer product)
        # If U_c2 = (1/alpha) * U_c1 and V_c2 = (1/beta) * V_c1, then:
        # U_c2 * V_c2 * W_c2 = (1/(alpha*beta)) * U_c1 * V_c1 * W_c2
        # So we can absorb column c2 into column c1 by adding W_c2/(alpha*beta) to W_c1.
        
        W[:, c1] += W[:, c2] / (alpha * beta)
        # Mark c2 as removed
        return True
    
    merged = True
    while merged and time.time() < deadline - 10:
        merged = False
        for i in range(len(active)):
            if time.time() > deadline - 10:
                break
            for j in range(i+1, len(active)):
                c1 = active[i]
                c2 = active[j]
                if try_merge(c1, c2):
                    active.remove(c2)
                    merged = True
                    break
            if merged:
                break
    
    # Extract the reduced decomposition
    final_cols = active
    U_final = U[:, final_cols]
    V_final = V[:, final_cols]
    W_final = W[:, final_cols]
    
    final_rank = len(final_cols)
    
    if time.time() > deadline - 5:
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
