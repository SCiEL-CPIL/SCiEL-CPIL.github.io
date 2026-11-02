import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def sanitize_matrix(M):
        M_2 = M * 2.0
        M_rounded = np.round(M_2)
        M_clean = M_rounded / 2.0
        return M_clean

    def verify(U, V, W, tol=1e-9):
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    u_idx = i*2 + j
                    v_idx = j*9 + k
                    w_idx = k*2 + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def build_rank32():
        terms = []
        def add_strassen_block(k0, k1):
            U1 = {0: 1, 3: 1}
            V1 = {k0: 1, 9+k1: 1}
            W1 = {2*k0: 1, 2*k1+1: 1}
            terms.append((U1, V1, W1))
            U2 = {2: 1, 3: 1}
            V2 = {k0: 1}
            W2 = {2*k0+1: 1, 2*k1+1: -1}
            terms.append((U2, V2, W2))
            U3 = {0: 1}
            V3 = {k1: 1, 9+k1: -1}
            W3 = {2*k1: 1, 2*k1+1: 1}
            terms.append((U3, V3, W3))
            U4 = {3: 1}
            V4 = {9+k0: 1, k0: -1}
            W4 = {2*k0: 1, 2*k0+1: 1}
            terms.append((U4, V4, W4))
            U5 = {0: 1, 1: 1}
            V5 = {9+k1: 1}
            W5 = {2*k0: -1, 2*k1: 1}
            terms.append((U5, V5, W5))
            U6 = {2: 1, 0: -1}
            V6 = {k0: 1, k1: 1}
            W6 = {2*k1+1: 1}
            terms.append((U6, V6, W6))
            U7 = {1: 1, 3: -1}
            V7 = {9+k0: 1, 9+k1: 1}
            W7 = {2*k0: 1}
            terms.append((U7, V7, W7))

        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            add_strassen_block(k0, k1)
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))

        rank = len(terms)
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W

    # Try to find a lower-rank decomposition using alternating minimization
    # on the 2x2x9 tensor.
    def try_lower_rank(target_ranks=[30, 31], max_iter=500, tol=1e-8):
        # Build the target tensor T[u,v,w]
        T = np.zeros((4, 18, 18))
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    T[i*2+j, j*9+k, k*2+i] = 1.0
        
        # Precompute for faster ALS
        # T_reshaped for solving U: (4, 18*18)
        T_u = T.reshape(4, 324)
        T_v = T.transpose(1,0,2).reshape(18, 72)
        T_w = T.transpose(2,0,1).reshape(18, 72)
        
        best_result = None
        for r in target_ranks:
            if time.time() > deadline - 20:
                break
            np.random.seed(42)
            U = np.random.randn(4, r)
            V = np.random.randn(18, r)
            W = np.random.randn(18, r)
            # Normalize
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 20:
                    break
                # Solve for U
                for t in range(r):
                    vw = np.kron(V[:,t], W[:,t])
                    U[:,t] = T_u @ vw / np.dot(vw, vw)
                # Solve for V
                for t in range(r):
                    uw = np.kron(U[:,t], W[:,t])
                    V[:,t] = T_v @ uw / np.dot(uw, uw)
                # Solve for W
                for t in range(r):
                    uv = np.kron(U[:,t], V[:,t])
                    W[:,t] = T_w @ uv / np.dot(uv, uv)
                
                # Check error
                T_recon = np.einsum('ut,vt,wt->uvw', U, V, W)
                err = np.linalg.norm(T_recon - T)
                if err < tol:
                    U_s = sanitize_matrix(U)
                    V_s = sanitize_matrix(V)
                    W_s = sanitize_matrix(W)
                    if verify(U_s, V_s, W_s):
                        return (U_s, V_s, W_s, r)
                    if verify(U, V, W):
                        U_s = sanitize_matrix(U)
                        V_s = sanitize_matrix(V)
                        W_s = sanitize_matrix(W)
                        if verify(U_s, V_s, W_s):
                            return (U_s, V_s, W_s, r)
                if err > prev_err * 1.01:
                    break
                prev_err = err
        return None

    # First, get the known rank-32 solution as fallback
    U32, V32, W32 = build_rank32()
    U32 = sanitize_matrix(U32)
    V32 = sanitize_matrix(V32)
    W32 = sanitize_matrix(W32)
    
    # Try to find lower rank
    result = try_lower_rank()
    
    if result is not None:
        U, V, W, r = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 32
    if not verify(U32, V32, W32):
        raise ValueError("Internal verification failed for rank 32 solution")
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
# EVOLVE-BLOCK-END
