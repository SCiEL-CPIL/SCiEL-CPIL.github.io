import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor indices
    # T[a, b, c] where a in [0, nm), b in [0, mp), c in [0, pn)
    # Non-zero at (i*m+j, j*p+k, k*n+i)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
    
    # We want to find minimal rank R
    # Lower bound is max(nm, mp, pn) = 21
    # Upper bound is 63
    best_rank = 63
    best_factors = None
    
    # Pre-compute schoolbook factors as fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try ranks from 21 to 63
    for R in range(21, 64):
        if time.time() > deadline:
            break
        
        found = False
        # Number of restarts
        # More restarts for lower ranks
        n_restarts = 20 if R < 30 else 10
        
        for _ in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # ALS iterations
            for it in range(100):
                # Update U
                # T^(1) is 9 x 441
                # We only have non-zeros.
                # But T is zero elsewhere.
                # So T^(1) is sparse.
                # We can construct the full matrix or use sparse logic.
                # Given small size, full matrix is fine.
                
                # Construct T_unfolded_1
                T1 = np.zeros((n*m, m*p * p*n))
                for a, b, c in nz_indices:
                    T1[a, b * (p*n) + c] = 1.0
                
                # Khatri-Rao product of W and V
                # W is (p*n, R), V is (m*p, R)
                # We need column r to be W[:, r] \otimes V[:, r]
                # Size (p*n * m*p, R)
                # Note: order matters. T1 index is b * (p*n) + c.
                # So b is the "slow" index, c is the "fast" index.
                # Kronecker product W[:, r] \otimes V[:, r] has index (c, b) if standard?
                # np.kron(W_col, V_col) gives [W[0]V[0], W[0]V[1], ... W[1]V[0]...]
                # If we flatten V then W, index is b * (p*n) + c?
                # Let's check: np.kron(A, B) where A is (M,1), B is (N,1).
                # Result is (M*N, 1). Element (i,j) is A[i] B[j].
                # If we want index b * (p*n) + c, we need B to correspond to c and A to b?
                # No, if we use np.kron(V_col, W_col), the index is i * (p*n) + j where i is from V (size mp) and j is from W (size pn).
                # So index = b * (p*n) + c.
                # So we need np.kron(V[:, r], W[:, r]).
                
                # Construct KR matrix
                KR = np.zeros((m*p * p*n, R))
                for r in range(R):
                    KR[:, r] = np.kron(V[:, r], W[:, r])
                
                # Solve U = T1 @ KR^+
                U, _, _, _ = np.linalg.lstsq(KR.T, T1.T, rcond=None)
                
                # Update V
                # T^(2) is (m*p) x (n*m * p*n)
                T2 = np.zeros((m*p, n*m * p*n))
                for a, b, c in nz_indices:
                    T2[b, a * (p*n) + c] = 1.0
                
                KR_V = np.zeros((n*m * p*n, R))
                for r in range(R):
                    KR_V[:, r] = np.kron(W[:, r], U[:, r])
                    # Index: a * (p*n) + c.
                    # np.kron(W_col, U_col) gives index c * (n*m) + a?
                    # No, np.kron(A, B) with A size M, B size N gives index i*M + j.
                    # We want index a * (p*n) + c.
                    # So "slow" index is a, "fast" is c.
                    # So we need np.kron(U_col, W_col)?
                    # U is size nm, W is size pn.
                    # np.kron(U, W) index is i * (pn) + j. i from U, j from W.
                    # So i=a, j=c. Correct.
                    KR_V[:, r] = np.kron(U[:, r], W[:, r])
                
                V, _, _, _ = np.linalg.lstsq(KR_V.T, T2.T, rcond=None)
                
                # Update W
                # T^(3) is (p*n) x (n*m * m*p)
                T3 = np.zeros((p*n, n*m * m*p))
                for a, b, c in nz_indices:
                    T3[c, a * (m*p) + b] = 1.0
                
                KR_W = np.zeros((n*m * m*p, R))
                for r in range(R):
                    # Index a * (mp) + b.
                    # Slow: a, Fast: b.
                    # np.kron(U, V) index i * (mp) + j.
                    KR_W[:, r] = np.kron(U[:, r], V[:, r])
                
                W, _, _, _ = np.linalg.lstsq(KR_W.T, T3.T, rcond=None)
            
            # Check error
            # Reconstruct tensor
            T_rec = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Error on non-zeros
            err = 0.0
            for a, b, c in nz_indices:
                err += (T_rec[a, b, c] - 1.0)**2
            # Error on zeros
            # We can check a random subset of zeros to save time
            # Or just check if max abs value elsewhere is small
            # T_rec is small, so we can just check max
            # But we need to exclude the non-zeros
            mask = np.ones((n*m, m*p, p*n), dtype=bool)
            for a, b, c in nz_indices:
                mask[a, b, c] = False
            err_zero = np.sum(T_rec[mask]**2)
            
            total_err = err + err_zero
            
            if total_err < 1e-6:
                # Try rounding
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify rounded
                T_rec_r = np.zeros((n*m, m*p, p*n))
                for r in range(R):
                    T_rec_r += np.einsum('i,j,k->ijk', U_round[:, r], V_round[:, r], W_round[:, r])
                
                err_r = 0.0
                for a, b, c in nz_indices:
                    err_r += (T_rec_r[a, b, c] - 1.0)**2
                err_zero_r = np.sum(T_rec_r[mask]**2)
                
                if err_r + err_zero_r < 1e-9:
                    found = True
                    best_rank = R
                    best_factors = (U_round, V_round, W_round)
                    break
        
        if found:
            break
    
    if best_factors is None:
        # Return schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        return {"U": best_factors[0].tolist(), "V": best_factors[1].tolist(), "W": best_factors[2].tolist()}
