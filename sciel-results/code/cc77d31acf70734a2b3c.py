# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 130

    def build_tensor_224():
        # M_{2,2,4}
        n_sub, m_sub, p_sub = 2, 2, 4
        T = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    T[a, b, c] = 1.0
        return T

    T_224 = build_tensor_224()

    def verify(U, V, W, T, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(U.shape[0])]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(V.shape[1])]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(W.shape[2])]
        return {"U": U_list, "V": V_list, "W": W_list}

    def build_strassen28():
        # ... (keep existing strassen28 code as fallback) ...
        n, m, p = DIMENSIONS
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

    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    U28, V28, W28 = build_strassen28()
    
    # Search for rank 12 decomposition of M_{2,2,4}
    def find_rank12_224():
        rng = np.random.RandomState(42)
        r = 12
        U = rng.randn(4, r)
        V = rng.randn(8, r)
        W = rng.randn(8, r)
        
        for _ in range(1000):
            if time.time() > deadline:
                return None
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            F = recon - T_224
            gU = np.einsum('abc,bt,ct->at', F, V, W)
            gV = np.einsum('abc,at,ct->bt', F, U, W)
            gW = np.einsum('abc,at,bt->ct', F, U, V)
            
            err = float(np.sum(F*F))
            lr = 0.1 / (1.0 + err)
            
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            
            if err < 1e-6:
                # Polish
                for _ in range(100):
                    recon = np.einsum('at,bt,ct->abc', U, V, W)
                    F = recon - T_224
                    gU = np.einsum('abc,bt,ct->at', F, V, W)
                    gV = np.einsum('abc,at,ct->bt', F, U, W)
                    gW = np.einsum('abc,at,bt->ct', F, U, V)
                    err = float(np.sum(F*F))
                    lr = 0.1 / (1.0 + err)
                    U -= lr * gU
                    V -= lr * gV
                    W -= lr * gW
                    if err < 1e-10:
                        break
                
                if verify(U, V, W, T_224, tol=1e-8):
                    return U, V, W
        return None

    res12 = find_rank12_224()
    if res12 is not None:
        U12, V12, W12 = res12
        # Construct M_{2,4,4} from M_{2,2,4}
        U_full = np.zeros((8, 24))
        V_full = np.zeros((16, 24))
        W_full = np.zeros((8, 24))
        
        # A_1 (rows 0-3 of U_full) uses U12
        U_full[0:4, 0:12] = U12
        # A_2 (rows 4-7 of U_full) uses U12 (same structure)
        U_full[4:8, 12:24] = U12
        
        # B_top (rows 0-7 of V_full) uses V12
        V_full[0:8, 0:12] = V12
        # B_bot (rows 8-15 of V_full) uses V12
        V_full[8:16, 12:24] = V12
        
        # W is shared output, so we sum contributions
        W_full[0:8, 0:12] = W12
        W_full[0:8, 12:24] = W12
        
        # Verify full tensor
        T_full = np.zeros((8, 16, 8))
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = k * 2 + i
                    T_full[a, b, c] = 1.0
        
        if verify(U_full, V_full, W_full, T_full, tol=1e-8):
            return to_lists(U_full, V_full, W_full, 24)

    # Fallback to rank 28
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
