import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s

    n, m, p = 2, 2, 11

    # ---- Build the 39-term Strassen construction ----
    def build_strassen():
        rank = 39
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        r = 0
        for block in range(5):
            k0 = block*2; k1 = block*2+1
            ve = k0; vf = k1; vg = 11+k0; vh = 11+k1
            wc00 = 2*k0; wc10 = 2*k0+1; wc01 = 2*k1; wc11 = 2*k1+1
            # m1=(a+d)(e+h)
            U[0,r]+=1; U[3,r]+=1; V[ve,r]+=1; V[vh,r]+=1; W[wc00,r]+=1; W[wc11,r]+=1; r+=1
            # m2=(c+d)e
            U[2,r]+=1; U[3,r]+=1; V[ve,r]+=1; W[wc10,r]+=1; W[wc11,r]+=-1; r+=1
            # m3=a(f-h)
            U[0,r]+=1; V[vf,r]+=1; V[vh,r]+=-1; W[wc01,r]+=1; W[wc11,r]+=1; r+=1
            # m4=d(g-e)
            U[3,r]+=1; V[vg,r]+=1; V[ve,r]+=-1; W[wc00,r]+=1; W[wc10,r]+=1; r+=1
            # m5=(a+b)h
            U[0,r]+=1; U[1,r]+=1; V[vh,r]+=1; W[wc00,r]+=-1; W[wc01,r]+=1; r+=1
            # m6=(c-a)(e+f)
            U[2,r]+=1; U[0,r]+=-1; V[ve,r]+=1; V[vf,r]+=1; W[wc11,r]+=1; r+=1
            # m7=(b-d)(g+h)
            U[1,r]+=1; U[3,r]+=-1; V[vg,r]+=1; V[vh,r]+=1; W[wc00,r]+=1; r+=1
        # last column schoolbook
        k2=10; ve2=k2; vg2=11+k2; wc02=2*k2; wc12=2*k2+1
        U[0,r]+=1; V[ve2,r]+=1; W[wc02,r]+=1; r+=1
        U[1,r]+=1; V[vg2,r]+=1; W[wc02,r]+=1; r+=1
        U[2,r]+=1; V[ve2,r]+=1; W[wc12,r]+=1; r+=1
        U[3,r]+=1; V[vg2,r]+=1; W[wc12,r]+=1; r+=1
        return U, V, W, rank

    def build_target():
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T

    def verify(U, V, W, T):
        cur = np.zeros_like(T)
        for r in range(U.shape[1]):
            cur += np.einsum('a,b,c->abc', U[:,r].astype(float), V[:,r].astype(float), W[:,r].astype(float))
        return np.max(np.abs(cur - T)) < 1e-9

    U, V, W, rank = build_strassen()
    T = build_target()
    assert verify(U, V, W, T), "baseline construction failed"

    # ---- Linear-solve rank reduction: fix U,W pattern, solve for V ----
    # Equations: for each (alpha,beta,gamma) with T=1: sum_r U[a,r]*V[b,r]*W[g,r] = 1
    # and for T=0 entries: sum = 0. Linear in V.
    # We'll try dropping columns of the Strassen pattern and re-solving V (and allow U,W to stay).
    def try_reduce(U0, V0, W0, T, max_iter=2000):
        best = (U0.copy(), V0.copy(), W0.copy(), U0.shape[1])
        # list of nonzero equation indices
        nz = np.argwhere(T != 0)  # (a,b,g)
        zero = np.argwhere(T == 0)
        # We'll work in float
        Uf = U0.astype(float); Vf = V0.astype(float); Wf = W0.astype(float)
        cur_rank = U0.shape[1]
        # Greedy: try removing each column, then re-solve V with U,W fixed (dropped col removed)
        for attempt in range(30):
            if time.time() > deadline:
                break
            improved = False
            for r_remove in range(cur_rank):
                if time.time() > deadline:
                    break
                # build U,W without col r_remove
                keep = [i for i in range(cur_rank) if i != r_remove]
                Ur = Uf[:, keep]; Wr = Wf[:, keep]
                R = len(keep)
                # unknowns: V[:, keep] shape (m*p, R)
                # equations: for each nz (a,b,g): sum_{r in keep} U[a,r]*V[b,r]*W[g,r] = 1
                # This is linear in V. Build A_eq * V_flat = rhs
                # V_flat index: b*R + r
                A_rows = []
                rhs = []
                for (a,b,g) in nz:
                    row = np.zeros(R * (m*p))
                    for ri, r in enumerate(keep):
                        coeff = Ur[a,ri] * Wr[g,ri]
                        if coeff != 0:
                            row[b*R + ri] = coeff
                    if np.any(row != 0):
                        A_rows.append(row)
                        rhs.append(1.0)
                if not A_rows:
                    continue
                A_eq = np.array(A_rows)
                rhs = np.array(rhs)
                # Also add zero-equations to constrain (helps uniqueness but not needed for existence)
                # Solve least squares
                try:
                    sol, res, rank_, sv = np.linalg.lstsq(A_eq, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                Vsol = sol.reshape(m*p, R)
                # check exact fit on nz
                pred = np.array([np.dot(A_eq[i], sol) for i in range(len(A_rows))])
                if np.max(np.abs(pred - rhs)) > 1e-8:
                    continue
                # build full U,V,W and verify against whole tensor
                Unew = Ur.astype(np.int64); Vnew = Vsol.astype(np.int64); Wnew = Wr.astype(np.int64)
                # check Vnew is integer (within tol)
                if np.max(np.abs(Vsol - np.round(Vsol))) > 1e-6:
                    continue
                Vnew = np.round(Vsol).astype(np.int64)
                if verify(Unew, Vnew, Wnew, T):
                    cur_rank = R
                    Uf = Unew.astype(float); Vf = Vnew.astype(float); Wf = Wnew.astype(float)
                    improved = True
                    break
            if not improved:
                break
        return Uf.astype(np.int64), Vf.astype(np.int64), Wf.astype(np.int64), cur_rank

    U, V, W, rank = try_reduce(U, V, W, T)

    # ---- Fallback: ensure valid ----
    if not verify(U, V, W, T):
        U, V, W, rank = build_strassen()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
