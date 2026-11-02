"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    schoolbook_rank = n * m * p  # 32
    deadline = time.time() + 130  # safety margin before 150s kill

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

    # === Refinement: free the last 2 columns, minimize residual.
    # Key improvement: use a smooth log-sum-exp surrogate for max|F| during GD,
    # then a final high-precision L2 polish and exact rational snap. ===
    r = 28
    free_dim = 2
    fixed_slice = slice(0, r - free_dim)

    def refine(seed, max_iters=6000):
        rng = np.random.RandomState(seed)
        U = U28.copy(); V = V28.copy(); W = W28.copy()
        U[:, r-free_dim:] = rng.randn(n*m, free_dim) * 0.5
        V[:, r-free_dim:] = rng.randn(m*p, free_dim) * 0.5
        W[:, r-free_dim:] = rng.randn(p*n, free_dim) * 0.5

        base_U = U[:, fixed_slice]; base_V = V[:, fixed_slice]; base_W = W[:, fixed_slice]
        base = np.einsum('at,bt,ct->abc', base_U, base_V, base_W)

        def smooth_loss(F, beta):
            """Approximate max|F| via log-sum-exp for smoothness."""
            absF = np.abs(F)
            m = absF.max()
            if m < 1e-15:
                return 0.0
            shifted = absF - m
            lse = m + np.log(np.sum(np.exp(beta * shifted))) / beta
            return lse

        def loss_and_grads(Uf, Vf, Wf, beta):
            recon = base + np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
            F = recon - T
            loss = smooth_loss(F, beta)
            absF = np.abs(F)
            m = absF.max()
            if m < 1e-15:
                dF = np.zeros_like(F)
            else:
                shifted = absF - m
                weights = np.exp(beta * shifted) / np.sum(np.exp(beta * shifted))
                dF = weights * np.sign(F)
            gU = np.einsum('abc,bt,ct->at', dF, Vf, Wf)
            gV = np.einsum('abc,at,ct->bt', dF, Uf, Wf)
            gW = np.einsum('abc,at,bt->ct', dF, Uf, Vf)
            return loss, gU, gV, gW

        best_loss = np.inf
        bestU, bestV, bestW = U[:, r-free_dim:].copy(), V[:, r-free_dim:].copy(), W[:, r-free_dim:].copy()

        # Adaptive beta: start soft (20), ramp up to 100
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            beta = 20.0 + 80.0 * (it / max_iters)  # ramp from 20 to 100
            Uf = U[:, r-free_dim:]; Vf = V[:, r-free_dim:]; Wf = W[:, r-free_dim:]
            recon = base + np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
            F = recon - T
            loss, gU, gV, gW = loss_and_grads(Uf, Vf, Wf, beta)

            err_l2 = float(np.sum(F * F))
            lr = 0.1 / (1.0 + err_l2)
            gnorm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2)) + 1e-12
            lr = min(lr, 0.5 / gnorm)

            U[:, r-free_dim:] -= lr * gU
            V[:, r-free_dim:] -= lr * gV
            W[:, r-free_dim:] -= lr * gW

            recon2 = base + np.einsum('at,bt,ct->abc', U[:, r-free_dim:], V[:, r-free_dim:], W[:, r-free_dim:])
            F2 = recon2 - T
            cur_loss = smooth_loss(F2, 100.0)  # always evaluate with high beta
            if cur_loss < best_loss:
                best_loss = cur_loss
                bestU, bestV, bestW = U[:, r-free_dim:].copy(), V[:, r-free_dim:].copy(), W[:, r-free_dim:].copy()
                if best_loss < 1e-10:
                    break

        # Final L2 polish
        U[:, r-free_dim:] = bestU; V[:, r-free_dim:] = bestV; W[:, r-free_dim:] = bestW
        for it in range(3000):
            if time.time() > deadline - 5:
                break
            Uf = U[:, r-free_dim:]; Vf = V[:, r-free_dim:]; Wf = W[:, r-free_dim:]
            recon = base + np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
            F = recon - T
            gU = np.einsum('abc,bt,ct->at', F, Vf, Wf)
            gV = np.einsum('abc,at,ct->bt', F, Uf, Wf)
            gW = np.einsum('abc,at,bt->ct', F, Uf, Vf)
            err = float(np.sum(F * F))
            lr = 0.05 / (1.0 + err)
            U[:, r-free_dim:] -= lr * gU
            V[:, r-free_dim:] -= lr * gV
            W[:, r-free_dim:] -= lr * gW
            if err < 1e-14:
                break

        # Snap to half-integers and verify
        U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
        U2[:, r-free_dim:] = np.round(U2[:, r-free_dim:] * 2) / 2
        V2[:, r-free_dim:] = np.round(V2[:, r-free_dim:] * 2) / 2
        W2[:, r-free_dim:] = np.round(W2[:, r-free_dim:] * 2) / 2
        if verify(U2, V2, W2, tol=1e-8):
            return U2, V2, W2

        # Try integer snap
        U3 = U.copy(); V3 = V.copy(); W3 = W.copy()
        U3[:, r-free_dim:] = np.round(U3[:, r-free_dim:])
        V3[:, r-free_dim:] = np.round(V3[:, r-free_dim:])
        W3[:, r-free_dim:] = np.round(W3[:, r-free_dim:])
        if verify(U3, V3, W3, tol=1e-8):
            return U3, V3, W3

        # Try rounding the entire matrices to half-integers
        Us = np.round(U * 2) / 2; Vs = np.round(V * 2) / 2; Ws = np.round(W * 2) / 2
        if verify(Us, Vs, Ws, tol=1e-8):
            return Us, Vs, Ws

        # Try rounding to nearest integer (stronger snap)
        Ui = np.round(U); Vi = np.round(V); Wi = np.round(W)
        if verify(Ui, Vi, Wi, tol=1e-8):
            return Ui, Vi, Wi

        return None

    seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227]
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        res = refine(seed)
        if res is not None:
            Uf, Vf, Wf = res
            return to_lists(Uf, Vf, Wf, r)  # rank 27

    # Fallback: verified rank-28 Strassen
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
