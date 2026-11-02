"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    schoolbook_rank = n * m * p  # 32
    deadline = time.time() + 140  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # C[i,j] -> index j*2 + i (trace convention k*n+i with n=2, k=j, i=i)
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi,(ga,gb,gc) in enumerate(blocks):
            for t in range(7):
                col = bi*7+t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li,lj), col] = U_s[li*2+lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj,lk), col] = V_s[lj*2+lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk,li), col] = W_s[lk*2+li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()
    if not verify(U28, V28, W28, tol=1e-8):
        U = np.zeros((n*m, schoolbook_rank)); V = np.zeros((m*p, schoolbook_rank)); W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
        return to_lists(U, V, W, schoolbook_rank)

    # Try to find a rank-27 decomposition by optimizing all 27 columns
    def try_rank27(seed, max_iters=3000):
        rng = np.random.RandomState(seed)
        r = 27
        # Start from Strassen-28, drop one column and randomize it
        U = U28.copy(); V = V28.copy(); W = W28.copy()
        U = U[:, :r].copy(); V = V[:, :r].copy(); W = W[:, :r].copy()
        # Add a small random perturbation to all entries
        U += rng.randn(n*m, r) * 0.1
        V += rng.randn(m*p, r) * 0.1
        W += rng.randn(p*n, r) * 0.1

        def loss_and_grads(U, V, W):
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            F = recon - T
            loss = float(np.sum(F * F))
            gU = np.einsum('abc,bt,ct->at', F, V, W)
            gV = np.einsum('abc,at,ct->bt', F, U, W)
            gW = np.einsum('abc,at,bt->ct', F, U, V)
            return loss, gU, gV, gW

        best_loss = np.inf
        bestU, bestV, bestW = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            loss, gU, gV, gW = loss_and_grads(U, V, W)
            if loss < best_loss:
                best_loss = loss
                bestU, bestV, bestW = U.copy(), V.copy(), W.copy()
                if loss < 1e-14:
                    break
            
            # Adaptive learning rate
            err = float(np.sum(F * F)) if 'F' in locals() else loss
            lr = 0.01 / (1.0 + loss)
            gnorm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2)) + 1e-12
            lr = min(lr, 0.1 / gnorm)
            
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW

        U, V, W = bestU, bestV, bestW
        # Snap to half-integers
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if verify(U2, V2, W2, tol=1e-8):
            return U2, V2, W2

        # Try integer snap
        U3 = np.round(U)
        V3 = np.round(V)
        W3 = np.round(W)
        if verify(U3, V3, W3, tol=1e-8):
            return U3, V3, W3

        return None

    # Try multiple seeds for rank-27
    seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        res = try_rank27(seed)
        if res is not None:
            Uf, Vf, Wf = res
            return to_lists(Uf, Vf, Wf, 27)

    # Fallback: verified rank-28 Strassen
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
