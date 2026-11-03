import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor vectors b_K
    # b_K is a vector of length nm * mp
    # b_K[I * mp + J] = 1 if I, J, K are consistent, else 0
    # I = i * m + j
    # J = j * p + k
    # K = k * n + i
    # Consistency: I determines (i, j), J determines (j', k), K determines (k', i')
    # We need i == i', j == j', k == k'
    
    b_vectors = []
    for K in range(pn):
        k = K // n
        i = K % n
        b = np.zeros(nm * mp)
        for j in range(m):
            I = i * m + j
            J = j * p + k
            b[I * mp + J] = 1.0
        b_vectors.append(b)
    b_vectors = np.array(b_vectors) # Shape (pn, nm * mp)
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try ranks from 40 down to 20
    for R in range(40, 19, -1):
        # We will try to find a decomposition of rank R
        # We iterate with random U, V
        for iter in range(2000):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Using small integers to help with snapping later
            U = np.random.randint(-1, 2, size=(nm, R))
            V = np.random.randint(-1, 2, size=(mp, R))
            
            # Construct A: shape (nm * mp, R)
            # A[I * mp + J, r] = U[I, r] * V[J, r]
            # We can use broadcasting
            # U[:, np.newaxis, :] * V[np.newaxis, :, :] -> (nm, mp, R)
            A_full = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            A = A_full.reshape(nm * mp, R)
            
            # Solve A W_K = b_K for each K
            # W is shape (pn, R)
            # We can solve all at once using least squares
            # A @ W.T = b_vectors.T
            # W.T = pinv(A) @ b_vectors.T
            try:
                # Using lstsq is more stable
                W_T, residuals, rank_A, s = np.linalg.lstsq(A, b_vectors.T, rcond=None)
                W = W_T.T
            except:
                continue
                
            # Check exactness
            # Reconstruct tensor
            # Rec[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
            # We only need to check the non-zero entries of the target tensor
            # And ensure the zero entries are zero?
            # The problem says "exactly decompose the matrix multiplication tensor".
            # This implies ALL entries must match.
            # The zero entries are crucial.
            
            # Let's check the full reconstruction for a few random entries or all?
            # Full check is O(nm * mp * pn * R) = 8 * 24 * 12 * 40 ~ 92k ops. Fast.
            
            # Rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            # Target T has 1s at (I, J, K) consistent, 0 elsewhere.
            
            # Let's check if the error is small
            # We can compute the error on the full tensor
            Rec = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            # Create target tensor
            T = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        I = i * m + j
                        J = j * p + k
                        K = k * n + i
                        T[I, J, K] = 1.0
                        
            err = np.linalg.norm(Rec - T)
            
            if err < 1e-9:
                # Found a real-valued solution
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                Rec_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                err_snap = np.linalg.norm(Rec_snap - T)
                
                if err_snap < 1e-9:
                    # Return the snapped solution
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                
                # If snapping failed, maybe the solution is not unique or ill-conditioned
                # Try to refine or accept the real solution if allowed?
                # Problem says "integer/half-integer entries".
                # So we must return snapped values.
                # If snapping fails, we continue searching.
                
    # Fallback to schoolbook
    rank = nm * mp
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
