"""Program for matrix multiplication tensor decomposition 4x4 x 4x6."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def _tensor_vec(Ucol, Vcol, Wcol):
        """Flatten the rank-1 tensor T[i,j,k]=U[i,j]*V[j,k]*W[k,i] into a length-96 vector."""
        Uv = np.array(Ucol, dtype=np.float64).reshape(n, m)
        Vv = np.array(Vcol, dtype=np.float64).reshape(m, p)
        Wv = np.array(Wcol, dtype=np.float64).reshape(p, n)
        T = np.einsum('ij,jk,ki->ijk', Uv, Vv, Wv)
        return T.reshape(-1)

    def _validate(U, V, W, rank):
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-8)

    def _verify_random(U, V, W, rank, n, m, p):
        rng = np.random.default_rng(12345)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        Cexp = A @ B
        Ccomp = np.zeros((n, p))
        for r in range(rank):
            alpha = np.sum(A * np.array(U[:, r], dtype=np.float64).reshape(n, m))
            beta = np.sum(B * np.array(V[:, r], dtype=np.float64).reshape(m, p))
            Ccomp += alpha * beta * np.array(W[:, r], dtype=np.float64).reshape(p, n).T
        return np.allclose(Ccomp, Cexp, atol=1e-8)

    # ---- Strassen 2x2x2 (rank 7) ----
    def strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    # ---- 2x2x3 with rank 11 (Strassen for cols 0-1 + schoolbook col 2) ----
    def strassen_2x2x3():
        U1, V1, W1 = strassen_2x2()
        U = np.zeros((4, 11), dtype=int)
        V = np.zeros((6, 11), dtype=int)
        W = np.zeros((6, 11), dtype=int)
        for c in range(7):
            U[:, c] = U1[:, c]
            V[0, c] = V1[0, c]   # j0k0
            V[1, c] = V1[1, c]   # j0k1
            V[3, c] = V1[2, c]   # j1k0
            V[4, c] = V1[3, c]   # j1k1
            W[0, c] = W1[0, c]
            W[1, c] = W1[1, c]
            W[2, c] = W1[2, c]
            W[3, c] = W1[3, c]
        # schoolbook col 2 (k=2): 4 terms
        # C[0,2]=A00*B02+A01*B12 ; C[1,2]=A10*B02+A11*B12
        U[0,7]=1; V[2,7]=1; W[4,7]=1   # A00*B02 -> C02
        U[1,8]=1; V[5,8]=1; W[4,8]=1   # A01*B12 -> C02
        U[2,9]=1; V[2,9]=1; W[5,9]=1   # A10*B02 -> C12
        U[3,10]=1; V[5,10]=1; W[5,10]=1 # A11*B12 -> C12
        return U, V, W

    # ---- Kronecker product of two decompositions ----
    def kron_decomposition(U1, V1, W1, U2, V2, W2):
        r1 = U1.shape[1]; r2 = U2.shape[1]
        rank = r1 * r2
        U = np.zeros((U1.shape[0]*U2.shape[0], rank), dtype=int)
        V = np.zeros((V1.shape[0]*V2.shape[0], rank), dtype=int)
        W = np.zeros((W1.shape[0]*W2.shape[0], rank), dtype=int)
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1*r2 + c2
                U[:, c] = np.kron(U1[:, c1], U2[:, c2])
                V[:, c] = np.kron(V1[:, c1], V2[:, c2])
                W[:, c] = np.kron(W1[:, c1], W2[:, c2])
        return U, V, W

    # ---- Block diagonal: A1(A2B2) + A2(B1B2) ----
    def block_diagonal(U1, V1, W1, U2, V2, W2):
        """Two independent 2x2x3 blocks on disjoint A-rows.
        Block1 uses A1(2x2) acting on B1(2x3) -> C1(2x3) [A rows 0-1]
        Block2 uses A2(2x2) acting on B2(2x3) -> C2(2x3) [A rows 2-3]
        """
        r1 = U1.shape[1]; r2 = U2.shape[1]
        rank = r1 + r2
        U = np.zeros((4, rank), dtype=int)
        V = np.zeros((6, rank), dtype=int)
        W = np.zeros((6, rank), dtype=int)
        # Block 1: A rows 0,1 ; B rows 0,1 ; C rows 0,1
        U[0:2, 0:r1] = U1[0:2, :]
        U[2:4, 0:r1] = 0
        V[0:2, 0:r1] = V1[:, :]
        V[2:6, 0:r1] = 0
        W[0:2, 0:r1] = W1[:, :]
        W[2:6, 0:r1] = 0
        # Block 2: A rows 2,3 ; B rows 2,3 ; C rows 2,3
        U[2:4, r1:rank] = U2[0:2, :]
        U[0:2, r1:rank] = 0
        V[2:4, r1:rank] = V2[:, :]
        V[0:2, r1:rank] = 0
        W[2:4, r1:rank] = W2[:, :]
        W[0:2, r1:rank] = 0
        return U, V, W

    # ---- Linear dependency elimination ----
    def _eliminate(U, V, W, rank, time_budget=25.0):
        """Find integer null-space relations among the rank-1 term vectors and remove
        as many terms as possible while keeping integer entries."""
        if time.time() > deadline - time_budget:
            return None
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        M = np.zeros((n*m*p, rank), dtype=np.float64)
        for c in range(rank):
            M[:, c] = _tensor_vec(Ua[:, c], Va[:, c], Wa[:, c])
        # null space of M^T (relations sum_c z_c * v_c = 0)
        try:
            _, s, Vt = np.linalg.svd(M.T, full_matrices=True)
        except np.linalg.LinAlgError:
            return None
        tol = 1e-9
        n_null = int(np.sum(s < tol))
        if n_null == 0:
            return None
        # null space basis rows of Vt (last n_null rows)
        N = Vt[rank - n_null: rank, :]  # shape (n_null, rank)
        # Greedy: find integer vectors in null space with one large coefficient,
        # and remove the term with that coefficient.
        removed = set()
        # Build a candidate integer relation from each null-space basis vector by
        # rounding and checking, plus small integer combinations.
        def _is_valid_relation(z):
            return np.allclose(M @ z, 0.0, atol=1e-7)

        candidates = []
        # each basis vector rounded
        for row in N:
            zr = np.round(row)
            if np.allclose(zr, 0.0):
                continue
            if _is_valid_relation(zr):
                candidates.append(zr)
        # small integer combinations of basis vectors
        if n_null <= 3:
            for combo in itertools.product([-2,-1,0,1,2], repeat=n_null):
                if combo == (0,)*n_null:
                    continue
                z = np.array(combo, dtype=np.float64) @ N
                zr = np.round(z)
                if np.allclose(zr, 0.0):
                    continue
                if _is_valid_relation(zr):
                    candidates.append(zr)
        # For each candidate, pick the term with max |coefficient| to remove
        for z in candidates:
            if time.time() > deadline - 8:
                break
            idx = np.argmax(np.abs(z))
            if idx in removed:
                continue
            # removing term idx means new decomposition = sum_{c!=idx} e_c v_c
            # = (sum_c e_c v_c) - e_idx v_idx = 0 - e_idx v_idx  ... wait.
            # The relation is sum_c z_c v_c = 0. If we remove term idx, we need the
            # remaining terms to still sum to the identity tensor. We can't just drop
            # one term unless z has support only on idx (impossible for a relation).
            # Instead: use the relation to REPLACE term idx. But removal requires that
            # term idx be expressible as a combo of the others, i.e., there exists z
            # with z_idx = -1 and the rest integer, so v_idx = sum_{c!=idx} z_c v_c.
            # Then dropping idx and scaling others by z_c gives the same tensor.
            # So we need a relation with z_idx = +/-1.
            if abs(z[idx]) >= 1.0 - 1e-6:
                # normalize so z_idx = 1
                factor = z[idx]
                znorm = z / factor
                if _is_valid_relation(znorm):
                    # v_idx = sum_{c!=idx} znorm_c v_c  (since znorm_idx=1, sum znorm v=0 => v_idx = -sum_{c!=idx} znorm_c v_c)
                    # Actually sum_c znorm_c v_c = 0 => v_idx = - sum_{c!=idx} znorm_c v_c
                    # New decomposition: for c != idx, multiply term c by -znorm_c, drop idx.
                    # This keeps the sum equal to identity.
                    new_U = Ua.copy(); new_V = Va.copy(); new_W = Wa.copy()
                    for c in range(rank):
                        if c == idx:
                            continue
                        scale = -znorm[c]
                        if abs(scale) < 1e-12:
                            continue
                        new_U[:, c] *= scale
                        new_V[:, c] *= scale
                        new_W[:, c] *= scale
                    # drop column idx
                    mask = [c for c in range(rank) if c != idx]
                    new_U = new_U[:, mask]
                    new_V = new_V[:, mask]
                    new_W = new_W[:, mask]
                    # check integer-ish
                    if (np.allclose(new_U, np.round(new_U), atol=1e-6) and
                        np.allclose(new_V, np.round(new_V), atol=1e-6) and
                        np.allclose(new_W, np.round(new_W), atol=1e-6)):
                        removed.add(idx)
                        # apply
                        Ua = np.round(new_U).astype(int)
                        Va = np.round(new_V).astype(int)
                        Wa = np.round(new_W).astype(int)
                        rank = rank - 1
                        # rebuild M
                        M = np.zeros((n*m*p, rank), dtype=np.float64)
                        for c in range(rank):
                            M[:, c] = _tensor_vec(Ua[:, c], Va[:, c], Wa[:, c])
                        if not _validate(Ua, Va, Wa, rank):
                            return None
                        # recompute null space
                        try:
                            _, s, Vt = np.linalg.svd(M.T, full_matrices=True)
                        except np.linalg.LinAlgError:
                            break
                        n_null = int(np.sum(s < tol))
                        if n_null == 0:
                            break
                        N = Vt[rank - n_null: rank, :]
                        candidates = []
                        for row in N:
                            zr = np.round(row)
                            if np.allclose(zr, 0.0):
                                continue
                            if _is_valid_relation(zr):
                                candidates.append(zr)
                        if n_null <= 3:
                            for combo in itertools.product([-2,-1,0,1,2], repeat=n_null):
                                if combo == (0,)*n_null:
                                    continue
                                z = np.array(combo, dtype=np.float64) @ N
                                zr = np.round(z)
                                if np.allclose(zr, 0.0):
                                    continue
                                if _is_valid_relation(zr):
                                    candidates.append(zr)
                        if not candidates:
                            break
                        # continue loop with new candidates
                        continue
        if not removed:
            return None
        return Ua.tolist(), Va.tolist(), Wa.tolist(), rank

    # ---- Build candidates ----
    best_rank = n*m*p
    best_result = None

    # 1. Schoolbook
    def build_schoolbook():
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    U_s, V_s, W_s, r_s = build_schoolbook()
    if _validate(U_s, V_s, W_s, r_s):
        best_rank = r_s
        best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # 2. Kronecker: M_{2,2,3}(11) ⊗ M_{2,2,2}(7) -> rank 77
    if time.time() < deadline - 30:
        U23, V23, W23 = strassen_2x2x3()
        U22, V22, W22 = strassen_2x2()
        U_k, V_k, W_k = kron_decomposition(U23, V23, W23, U22, V22, W22)
        rk = U_k.shape[1]
        if _validate(U_k, V_k, W_k, rk) and _verify_random(U_k, V_k, W_k, rk, n, m, p):
            if rk < best_rank:
                best_rank = rk
                best_result = {"U": U_k.tolist(), "V": V_k.tolist(), "W": W_k.tolist()}
            # try elimination
            elim = _eliminate(U_k, V_k, W_k, rk, time_budget=20.0)
            if elim is not None:
                Ue, Ve, We, re = elim
                if re < best_rank and _validate(Ue, Ve, We, re):
                    best_rank = re
                    best_result = {"U": Ue, "V": Ve, "W": We}

    # 3. Block diagonal: two 2x2x3 blocks (rank 22) -- but this is WRONG for full 4x4x6
    # Actually block diagonal only covers 2x2x3 + 2x2x3 = 4x2x6, not 4x4x6.
    # Skip block diagonal since it doesn't produce a valid 4x4x6 decomposition.

    if best_result is None:
        best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return best_result
# EVOLVE-BLOCK-END
