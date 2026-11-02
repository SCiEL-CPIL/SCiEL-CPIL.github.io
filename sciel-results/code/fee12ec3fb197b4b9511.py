"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 135  # safety margin before 150s kill
    schoolbook_rank = n * m * p  # 32

    # Build the multiplication tensor T[a,b,c] (official trace convention W[k*n+i, r])
    # a = i*m + j ; b = j*p + k ; c = k*n + i
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

    def recon(U, V, W):
        return np.einsum('at,bt,ct->abc', U, V, W)

    def max_err(U, V, W):
        return np.max(np.abs(recon(U, V, W) - T))

    def verify(U, V, W, tol=1e-8):
        return max_err(U, V, W) < tol

    def to_lists(U, V, W):
        r = U.shape[1]
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # ------------------------------------------------------------------
    # STRASSEN 2x2x2x2 base decomposition (rank 7)
    # local A idx = i*2+j, B idx = j*2+k, C idx (trace) = k*2+i
    # ------------------------------------------------------------------
    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # M1=(A00+A11)(B00+B11); M2=(A10+A11)B00; M3=A00(B01-B11); M4=A11(B10-B00)
        # M5=(A00+A01)B11; M6=(A10-A00)(B00+B01); M7=(A01-A11)(B10+B11)
        # C00=M1+M4-M5+M7 ; C01=M3+M5 ; C10=M2+M4 ; C11=M1-M2+M3+M6
        # C idx (trace k*2+i): C00=0, C10=1, C01=2, C11=3
        U[0,0]=1; U[3,0]=1;  V[0,0]=1; V[3,0]=1;  W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1;  V[0,1]=1;             W[1,1]=1; W[3,1]=-1
        U[0,2]=1;             V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
        U[3,3]=1;             V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
        U[0,4]=1; U[1,4]=1;  V[3,4]=1;             W[0,4]=-1; W[2,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1;   W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1;   W[0,6]=1
        return U, V, W

    U_s, V_s, W_s = strassen_2x2_decomp()

    # ------------------------------------------------------------------
    # Build full rank-28 block decomposition from 4 Strassen products.
    # A = [A_left | A_right] (2x4) ; B split into 4 2x2 blocks.
    # P1=A_left*B_TL, P2=A_right*B_BL, P3=A_left*B_TR, P4=A_right*B_BR
    # ------------------------------------------------------------------
    def build_strassen_full():
        full_rank = 28
        Uf = np.zeros((n*m, full_rank))   # 8 x 28
        Vf = np.zeros((m*p, full_rank))   # 16 x 28
        Wf = np.zeros((p*n, full_rank))   # 8 x 28
        # block (a_offset_row, a_offset_col, b_offset_row, b_offset_col, c_offset_col)
        # a global = i*m + j ; b global = j*p + k ; c global = k*n + i
        blocks = [
            # (jA0, jA1, jB0, jB1, kC0)  meaning A cols jA0..jA0+1, B rows jB0..jB0+1, C cols kC0..kC0+1
            (0, 0, 0),   # P1: A cols 0-1, B rows 0-1, C cols 0-1
            (2, 0, 0),   # P2: A cols 2-3, B rows 2-3, C cols 0-1
            (0, 2, 0),   # P3: A cols 0-1, B rows 0-1, C cols 2-3
            (2, 2, 2),   # P4: A cols 2-3, B rows 2-3, C cols 2-3
        ]
        for bi, (jA0, jB0, kC0) in enumerate(blocks):
            for t in range(7):
                col = bi*7 + t
                # U: A rows i in 0..1, local j = 0,1 -> global j = jA0 + j
                for li in range(2):
                    for lj in range(2):
                        local_a = li*2 + lj
                        global_a = li*m + (jA0 + lj)
                        Uf[global_a, col] = U_s[local_a, t]
                # V: B rows j in jB0..jB0+1, local j = 0,1 -> global row jB0+j, col k = kC0+lk
                for lj in range(2):
                    for lk in range(2):
                        local_b = lj*2 + lk
                        global_b = (jB0 + lj)*p + (kC0 + lk)
                        Vf[global_b, col] = V_s[local_b, t]
                # W: C rows i in 0..1, cols k in kC0..kC0+1 -> global c = k*n + i
                for lk in range(2):
                    for li in range(2):
                        local_c = lk*2 + li
                        global_c = (kC0 + lk)*n + li
                        Wf[global_c, col] = W_s[local_c, t]
        return Uf, Vf, Wf

    U28, V28, W28 = build_strassen_full()
    assert verify(U28, V28, W28), "Strassen rank-28 construction must be exact"
    best = (28, U28.copy(), V28.copy(), W28.copy())

    # ------------------------------------------------------------------
    # Border-method rank bound: rank(T) >= max over slices of rank of slice.
    # Also a simple "flattening" lower bound. This tells us if <28 is possible.
    # ------------------------------------------------------------------
    def rank_bound():
        # slice along each mode
        lb = 0
        for a in range(n*m):
            sl = T[a, :, :].reshape(n*m, m*p*p*n)  # not a valid mat; use 2D slices
        # Use 2D matrix flattenings: fix one index -> matrix
        for a in range(n*m):
            M = T[a, :, :].reshape(n*m, m*p, p*n)[0]  # shape (m*p, p*n)
            r = np.linalg.matrix_rank(M, tol=1e-9)
            lb = max(lb, r)
        for b in range(m*p):
            M = T[:, b, :].reshape(n*m, m*p, p*n)[:, b, :]
            r = np.linalg.matrix_rank(M, tol=1e-9)
            lb = max(lb, r)
        for c in range(p*n):
            M = T[:, :, c].reshape(n*m, m*p, p*n)[:, :, c]
            r = np.linalg.matrix_rank(M, tol=1e-9)
            lb = max(lb, r)
        return lb

    lb = rank_bound()

    # ------------------------------------------------------------------
    # Greedy rank-1 peeling with snapping: treat tensor as one entity.
    # Start from a rank-r candidate (Strassen), try to merge/zero terms to
    # reduce effective rank, snapping to half-integers at the end.
    # ------------------------------------------------------------------
    def snap_half(X):
        return np.round(X * 2.0) / 2.0

    def greedy_reduce(U, V, W, target_rank, max_iters=2000):
        """Try to reduce the decomposition to target_rank terms via greedy
        rank-1 peeling + local merge. Returns (U,V,W) of width target_rank if
        exact, else None."""
        r = U.shape[1]
        if r <= target_rank:
            return U, V, W
        # Work on copies
        Uc, Vc, Wc = U.copy(), V.copy(), W.copy()
        rng = np.random.RandomState(0)
        # Strategy: iteratively pick a term, try to absorb it into others by
        # adjusting. This is heuristic; we instead try a direct randomized
        # re-parameterization to target_rank and Newton-refine.
        return newton_to_rank(Uc, Vc, Wc, target_rank, max_iters)

    def newton_to_rank(U, V, W, target_rank, max_iters=3000, seed=1):
        """Random re-init to target_rank and Newton-refine to exactness."""
        rng = np.random.RandomState(seed)
        r = target_rank
        Ur = rng.randn(n*m, r) * 0.5
        Vr = rng.randn(m*p, r) * 0.5
        Wr = rng.randn(p*n, r) * 0.5
        best_err = np.inf
        best = (Ur.copy(), Vr.copy(), Wr.copy())
        for it in range(max_iters):
            if time.time() > deadline:
                break
            F = recon(Ur, Vr, Wr) - T
            err = np.sum(F*F)
            if err < best_err:
                best_err = err
                best = (Ur.copy(), Vr.copy(), Wr.copy())
            grad_U = np.einsum('abc,bt,ct->at', F, Vr, Wr)
            grad_V = np.einsum('abc,at,ct->bt', F, Ur, Wr)
            grad_W = np.einsum('abc,at,bt->ct', F, Ur, Vr)
            lr = 0.02 / (1.0 + err)
            Ur -= lr * grad_U
            Vr -= lr * grad_V
            Wr -= lr * grad_W
            if it % 200 == 199:
                # snap and check
                Us = snap_half(Ur); Vs = snap_half(Vr); Ws = snap_half(Wr)
                se = max_err(Us, Vs, Ws)
                if se < 1e-8:
                    return Us, Vs, Ws
        return None

    # ------------------------------------------------------------------
    # Try to beat rank 28: search target ranks from lb up to 27.
    # ------------------------------------------------------------------
    for target_rank in range(max(lb, 20), min(28, best[0])):
        if time.time() > deadline:
            break
        found = False
        for seed in range(1, 12):
            if time.time() > deadline:
                break
            res = newton_to_rank(U28, V28, W28, target_rank,
                                 max_iters=2500, seed=seed)
            if res is not None:
                Ux, Vx, Wx = res
                if verify(Ux, Vx, Wx):
                    best = (target_rank, Ux, Vx, Wx)
                    found = True
                    break
        if found:
            break

    # ------------------------------------------------------------------
    # Fallback: if no lower rank found, return the rank-28 Strassen solution.
    # ------------------------------------------------------------------
    rank, Ub, Vb, Wb = best
    return to_lists(Ub, Vb, Wb)
# EVOLVE-BLOCK-END
