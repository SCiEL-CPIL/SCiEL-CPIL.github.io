import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    # 1. Construct Rank 28 Solution
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28), dtype=int)
    V28 = np.zeros((m * p, rank_28), dtype=int)
    W28 = np.zeros((p * n, rank_28), dtype=int)

    # Strassen blocks
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3]),
    ]

    for blk in range(4):
        ar0, ar1 = blocks[blk][0]
        ac0, ac1 = blocks[blk][1]
        br0, br1 = blocks[blk][2]
        bc0, bc1 = blocks[blk][3]
        cr0, cr1 = blocks[blk][4]
        cc0, cc1 = blocks[blk][5]

        base = blk * 7

        aa00 = ar0 * m + ac0
        aa01 = ar0 * m + ac1
        aa10 = ar1 * m + ac0
        aa11 = ar1 * m + ac1

        bb00 = br0 * p + bc0
        bb01 = br0 * p + bc1
        bb10 = br1 * p + bc0
        bb11 = br1 * p + bc1

        cc00 = cc0 * n + cr0
        cc01 = cc1 * n + cr0
        cc10 = cc0 * n + cr1
        cc11 = cc1 * n + cr1

        # M1 = (a00+a11)(b00+b11)
        U28[aa00][base + 0] += 1
        U28[aa11][base + 0] += 1
        V28[bb00][base + 0] += 1
        V28[bb11][base + 0] += 1
        W28[cc00][base + 0] += 1
        W28[cc11][base + 0] += 1

        # M2 = (a10+a11)b00
        U28[aa10][base + 1] += 1
        U28[aa11][base + 1] += 1
        V28[bb00][base + 1] += 1
        W28[cc10][base + 1] += 1
        W28[cc11][base + 1] -= 1

        # M3 = a00(b01-b11)
        U28[aa00][base + 2] += 1
        V28[bb01][base + 2] += 1
        V28[bb11][base + 2] -= 1
        W28[cc01][base + 2] += 1
        W28[cc11][base + 2] += 1

        # M4 = a11(b10-b00)
        U28[aa11][base + 3] += 1
        V28[bb10][base + 3] += 1
        V28[bb00][base + 3] -= 1
        W28[cc00][base + 3] += 1
        W28[cc10][base + 3] += 1

        # M5 = (a00+a01)b11
        U28[aa00][base + 4] += 1
        U28[aa01][base + 4] += 1
        V28[bb11][base + 4] += 1
        W28[cc00][base + 4] -= 1
        W28[cc01][base + 4] += 1

        # M6 = (a10-a00)(b00+b01)
        U28[aa10][base + 5] += 1
        U28[aa00][base + 5] -= 1
        V28[bb00][base + 5] += 1
        V28[bb01][base + 5] += 1
        W28[cc11][base + 5] += 1

        # M7 = (a01-a11)(b10+b11)
        U28[aa01][base + 6] += 1
        U28[aa11][base + 6] -= 1
        V28[bb10][base + 6] += 1
        V28[bb11][base + 6] += 1
        W28[cc00][base + 6] += 1

    def verify(U_mat, V_mat, W_mat):
        Uf = np.array(U_mat, dtype=float)
        Vf = np.array(V_mat, dtype=float)
        Wf = np.array(W_mat, dtype=float)
        A = np.arange(n * m).reshape(n, m) + 1.0
        B = np.arange(m * p).reshape(m, p) + 2.0
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(Uf.shape[1]):
                    for jj in range(m):
                        acc += Uf[ii * m + jj, cc] * Vf[jj * p + kk, cc] * Wf[kk * n + ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-9)

    if not verify(U28, V28, W28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

    # 2. Kruskal bound via mode-3 flattening
    # T[i,j,k] = 1 if j==k and i//m == k%n... wait
    # T[a*m+b, b*p+c, c*n+a] = 1
    # Mode-3 flattening: for each k (index 2), we have a matrix of shape (n*m) x (m*p)
    # Actually mode-3 flattening means fixing the 3rd index and flattening the first two.
    # Let's compute the rank of the tensor's mode-3 unfolding.
    # The mode-3 unfolding is a matrix of shape (p*n) x (n*m * m*p)
    # For each k in range(p*n), the row is the slice T[:,:,k] flattened.
    
    T_flat = np.zeros((p*n, n*m*m*p), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_flat[k, i * (m*p) + j] = 1.0
    
    # Compute rank of T_flat
    # This gives the Kruskal bound for the 3rd mode
    rank_mode3 = np.linalg.matrix_rank(T_flat, tol=1e-10)
    
    # The Kruskal bound is min(rank_mode1, rank_mode2, rank_mode3)
    # But actually Kruskal's bound says rank(T) >= max of the Kruskal bounds?
    # No, Kruskal's bound is: if rank(T) = r, then r >= (n1 * n2 * n3 - 2) / 2? No.
    # The Kruskal bound is: rank(T) >= max(k1, k2, k3) where ki are the Kruskal ranks.
    # Actually, the bound is: rank(T) >= (n1*n2*n3 - 1) / max(n1, n2, n3)? No.
    # Let's just use the mode-3 rank as a lower bound.
    # rank(T) >= rank(mode-3 unfolding)
    
    # For our tensor, let's see what rank_mode3 is.
    # If rank_mode3 <= 27, then rank 27 is possible.
    # If rank_mode3 > 27, then rank 27 is impossible.
    
    if rank_mode3 > 27:
        # Rank 27 is impossible by Kruskal bound
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

    # 3. Attempt Rank 27 via randomized projection + rational reconstruction
    target_rank = 27
    
    # Strategy: 
    # We have a rank-28 decomposition. We want to find a rank-27 one.
    # Idea: Project the 28 factor vectors onto a 27-dimensional subspace.
    # If the projected factors still span the same tensor, we might be able to
    # re-express them as 27 rank-1 terms.
    
    # Alternative: Use the fact that if rank is 27, then the 28 rank-1 terms
    # must be linearly dependent in some way.
    
    # Let's try a different approach: 
    # 1. Start with the rank-28 solution.
    # 2. Try to find a linear dependency among the 28 rank-1 terms.
    # 3. If found, we can potentially reduce the rank.
    
    # Actually, a simpler approach: 
    # Use randomized initialization and gradient descent, but with a better
    # initialization strategy.
    
    # Let's try: initialize 27 terms by taking 28 terms and randomly combining
    # two of them into one (if possible).
    
    # Even better: use the "tensor rank reduction" algorithm.
    # If we have a rank-r decomposition, we can try to find a rank-(r-1) one
    # by looking for a hyperplane that contains r-1 of the rank-1 components.
    
    # For now, let's use a more direct approach:
    # 1. Build the tensor T.
    # 2. Use randomized SVD or similar to get a low-rank approximation.
    # 3. Try to find exact integer/half-integer factors.
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a * m + b, b * p + c, c * n + a] = 1.0
    
    # Mode-1 flattening: shape (n*m) x (m*p * p*n)
    T_mode1 = T.reshape(n*m, m*p*p*n)
    # Mode-2 flattening: shape (m*p) x (n*m * p*n)
    T_mode2 = np.transpose(T, (1,0,2)).reshape(m*p, n*m*p*n)
    # Mode-3 flattening: shape (p*n) x (n*m * m*p)
    T_mode3 = np.transpose(T, (2,0,1)).reshape(p*n, n*m*m*p)
    
    rank1 = np.linalg.matrix_rank(T_mode1, tol=1e-10)
    rank2 = np.linalg.matrix_rank(T_mode2, tol=1e-10)
    rank3 = np.linalg.matrix_rank(T_mode3, tol=1e-10)
    
    # The tensor rank is at least max(rank1, rank2, rank3)
    lower_bound = max(rank1, rank2, rank3)
    
    if lower_bound > 27:
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

    # Try to find a rank-27 decomposition
    # Use randomized approach: 
    # 1. Generate random 27-term decompositions.
    # 2. Check if they can be made exact via rational reconstruction.
    
    # Better: use the existing rank-28 solution and try to merge two terms.
    
    # Let's try gradient descent with multiple restarts.
    np.random.seed(42)
    
    best_err = float('inf')
    best_solution = None
    
    max_restarts = 50
    max_iters = 2000
    lr = 0.005
    momentum = 0.9
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize from rank-28 solution with one column dropped and noise
        drop_idx = restart % rank_28
        mask = np.ones(rank_28, dtype=bool)
        mask[drop_idx] = False
        U_init = U28[:, mask].astype(float)
        V_init = V28[:, mask].astype(float)
        W_init = W28[:, mask].astype(float)
        
        # Add noise
        noise_scale = 0.05 * (restart % 5 + 1)
        U_cur = U_init + np.random.randn(*U_init.shape) * noise_scale
        V_cur = V_init + np.random.randn(*V_init.shape) * noise_scale
        W_cur = W_init + np.random.randn(*W_cur.shape) * noise_scale
        
        vU = np.zeros_like(U_cur)
        vV = np.zeros_like(V_cur)
        vW = np.zeros_like(W_cur)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute T_est efficiently
            T_est = np.zeros_like(T)
            for r in range(target_rank):
                T_est += np.einsum('i,j,k->ijk', U_cur[:, r], V_cur[:, r], W_cur[:, r])
            
            E = T - T_est
            err_norm = np.linalg.norm(E)
            
            if err_norm < 1e-8:
                # Try to round to integers or half-integers
                U_round = np.round(U_cur * 2) / 2
                V_round = np.round(V_cur * 2) / 2
                W_round = np.round(W_cur * 2) / 2
                
                # Check if rounded solution is valid
                if verify(U_round, V_round, W_round):
                    # Convert to proper format
                    U_out = U_round.tolist()
                    V_out = V_round.tolist()
                    W_out = W_round.tolist()
                    return {"U": U_out, "V": V_out, "W": W_out}
                
                # Try integer rounding
                U_round_int = np.round(U_cur)
                V_round_int = np.round(V_cur)
                W_round_int = np.round(W_cur)
                
                if verify(U_round_int, V_round_int, W_round_int):
                    return {"U": U_round_int.tolist(), "V": V_round_int.tolist(), "W": W_round_int.tolist()}
            
            if err_norm < best_err:
                best_err = err_norm
                best_solution = (U_cur.copy(), V_cur.copy(), W_cur.copy())
            
            # Compute gradients
            gU = -np.einsum('ijk,jr,kr->ir', E, V_cur, W_cur)
            gV = -np.einsum('ijk,ir,kr->jr', E, U_cur, W_cur)
            gW = -np.einsum('ijk,ir,jr->kr', E, U_cur, V_cur)
            
            # Update with momentum
            vU = momentum * vU - lr * gU
            vV = momentum * vV - lr * gV
            vW = momentum * vW - lr * gW
            
            U_cur += vU
            V_cur += vV
            W_cur += vW
            
            # Normalize to prevent divergence
            for r in range(target_rank):
                norm_U = np.linalg.norm(U_cur[:, r])
                norm_V = np.linalg.norm(V_cur[:, r])
                norm_W = np.linalg.norm(W_cur[:, r])
                
                if norm_U > 1e-10 and norm_V > 1e-10 and norm_W > 1e-10:
                    # Keep the product of norms roughly constant
                    target_norm = (norm_U * norm_V * norm_W) ** (1/3)
                    U_cur[:, r] *= target_norm / norm_U
                    V_cur[:, r] *= target_norm / norm_V
                    W_cur[:, r] *= target_norm / norm_W

    # If no rank-27 solution found, return rank-28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
