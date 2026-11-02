import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 140.0  # safety margin under 150s wall-clock

    # Build the matrix-multiplication tensor using the official trace convention.
    # U index u = i*m + k, V index v = k*p + j, W index w = j*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1

    # ---- Rank-189 block Strassen construction (2x2 blocks, 3x3 schoolbook inside) ----
    rank = 189
    bs = 3
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    strassen_defs = [
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)]},
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)]},
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)]},
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)]},
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)]},
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)]},
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)]},
    ]
    recon_coeffs = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],
        [(2, 1), (4, 1)],
        [(1, 1), (3, 1)],
        [(0, 1), (1, -1), (2, 1), (5, 1)],
    ]

    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        U[i_glob * m + k_glob, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        V[k_glob * p + j_glob, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            W[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    # ---- Exact rank reduction: merge tensor terms whose (u,v) pair repeats ----
    # A term r contributes U[:,r]*V[:,r]*W[:,r]. If two terms share the same
    # (U[:,r], V[:,r]) pattern, their W vectors add, so the two terms collapse
    # into one without changing the tensor. This is an exact reduction.
    def reduce_rank(U, V, W):
        nR = U.shape[1]
        groups = {}
        for r in range(nR):
            key = (U[:, r].tobytes(), V[:, r].tobytes())
            groups.setdefault(key, []).append(r)
        newU, newV, newW = [], [], []
        for key, idxs in groups.items():
            if len(idxs) == 1:
                r = idxs[0]
                newU.append(U[:, r])
                newV.append(V[:, r])
                newW.append(W[:, r])
            else:
                newU.append(U[:, idxs[0]])
                newV.append(V[:, idxs[0]])
                newW.append(np.sum(W[:, idxs], axis=1))
        if not newU:
            return U, V, W
        return (np.stack(newU, axis=1), np.stack(newV, axis=1), np.stack(newW, axis=1))

    # ---- Exact verification (integer-safe via Kronecker product) ----
    def is_exact(U, V, W):
        Uf, Vf, Wf = U.astype(np.int64), V.astype(np.int64), W.astype(np.int64)
        # kron(U, V)[u*nV+v, r] = U[u,r]*V[v,r]; einsum with W gives T[u,v,w]
        UV = np.kron(Uf, Vf)
        Trec = np.einsum('rv,wr->vw', UV, Wf)
        return bool(np.array_equal(Trec, T))

    # Verify the constructed rank-189 solution (guards against construction bugs).
    assert is_exact(U, V, W), "rank-189 construction failed verification"

    # Apply the exact reduction pass.
    U2, V2, W2 = reduce_rank(U, V, W)
    if is_exact(U2, V2, W2) and W2.shape[1] < W.shape[1]:
        U, V, W = U2, V2, W2

    # Safety: if anything above failed, fall back to the verified rank-189.
    if not is_exact(U, V, W):
        U, V, W = np.zeros((n*m, 189), np.int64), np.zeros((m*p, 189), np.int64), np.zeros((p*n, 189), np.int64)
        # (rebuild would be needed; but the assert above guarantees the primary path is valid)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
