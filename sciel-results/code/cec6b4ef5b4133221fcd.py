import time
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 130.0  # safety margin before 150s kill

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Build the rank-71 baseline: 4 disjoint 2x2x2 Strassen blocks + trivial terms
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]

    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True

    trivial_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    trivial_terms.append((i, j, k))

    def build_strassen_block(i0, j0, k0, col, U, V, W):
        U[i0*m + j0][col+0] = 1
        U[(i0+1)*m + (j0+1)][col+0] = 1
        V[j0*p + k0][col+0] = 1
        V[(j0+1)*p + (k0+1)][col+0] = 1
        W[k0*n + i0][col+0] = 1
        W[(k0+1)*n + (i0+1)][col+0] = 1

        U[(i0+1)*m + j0][col+1] = -1
        U[(i0+1)*m + (j0+1)][col+1] = 1
        V[j0*p + k0][col+1] = 1
        W[k0*n + (i0+1)][col+1] = 1
        W[(k0+1)*n + (i0+1)][col+1] = -1

        U[i0*m + j0][col+2] = 1
        V[j0*p + (k0+1)][col+2] = -1
        V[(j0+1)*p + (k0+1)][col+2] = 1
        W[(k0+1)*n + i0][col+2] = 1
        W[(k0+1)*n + (i0+1)][col+2] = 1

        U[(i0+1)*m + (j0+1)][col+3] = 1
        V[j0*p + k0][col+3] = -1
        V[(j0+1)*p + k0][col+3] = 1
        W[k0*n + i0][col+3] = 1
        W[k0*n + (i0+1)][col+3] = 1

        U[i0*m + j0][col+4] = -1
        U[i0*m + (j0+1)][col+4] = 1
        V[(j0+1)*p + (k0+1)][col+4] = 1
        W[k0*n + i0][col+4] = 1
        W[(k0+1)*n + i0][col+4] = -1

        U[i0*m + j0][col+5] = -1
        U[(i0+1)*m + j0][col+5] = 1
        V[j0*p + k0][col+5] = -1
        V[j0*p + (k0+1)][col+5] = 1
        W[(k0+1)*n + (i0+1)][col+5] = 1

        U[i0*m + (j0+1)][col+6] = 1
        V[(j0+1)*p + k0][col+6] = -1
        V[(j0+1)*p + (k0+1)][col+6] = 1
        W[k0*n + i0][col+6] = 1

    total_rank = len(blocks) * 7 + len(trivial_terms)
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)

    col = 0
    for i0, j0, k0 in blocks:
        build_strassen_block(i0, j0, k0, col, U, V, W)
        col += 7
    for (i, j, k) in trivial_terms:
        U[i*m + j, col] = 1
        V[j*p + k, col] = 1
        W[k*n + i, col] = 1
        col += 1

    # Verify baseline
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    T_decomp = np.einsum('ar,br,cr->abc', U.astype(np.float64), V.astype(np.float64), W.astype(np.float64))
    if not np.allclose(T_decomp, target, atol=1e-10):
        rank = n * m * p
        U_f = np.zeros((n*m, rank), dtype=np.int64)
        V_f = np.zeros((m*p, rank), dtype=np.int64)
        W_f = np.zeros((p*n, rank), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j, c] = 1
                    V_f[j*p+k, c] = 1
                    W_f[k*n+i, c] = 1
                    c += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    best_rank = total_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to reduce rank by finding pairs of columns whose combined contribution
    # is rank-1, allowing us to merge them into a single column.
    # For each pair (r1, r2), check if T = col_r1 + col_r2 has rank 1.
    # If so, we can replace both with a single rank-1 term.

    # Precompute column tensors as sparse representations
    def get_col_tensor(r):
        """Return dict mapping (a,b,c) -> value for column r."""
        t = {}
        for a in range(n*m):
            ua = U[a, r]
            if ua == 0:
                continue
            for b in range(m*p):
                vb = V[b, r]
                if vb == 0:
                    continue
                for c in range(p*n):
                    vc = W[c, r]
                    if vc == 0:
                        continue
                    t[(a, b, c)] = ua * vb * vc
        return t

    col_tensors = [get_col_tensor(r) for r in range(total_rank)]

    # Try to find mergeable pairs
    merged = [False] * total_rank
    new_U = U.copy()
    new_V = V.copy()
    new_W = W.copy()
    new_rank = total_rank

    for r1 in range(total_rank):
        if merged[r1]:
            continue
        if time.time() - start > DEADLINE - 5:
            break
        for r2 in range(r1+1, total_rank):
            if merged[r2]:
                continue
            # Combine tensors
            combined = {}
            for key, val in col_tensors[r1].items():
                combined[key] = val
            for key, val in col_tensors[r2].items():
                combined[key] = combined.get(key, 0) + val

            # Remove zeros
            combined = {k: v for k, v in combined.items() if v != 0}

            if not combined:
                # Both columns cancel each other exactly
                merged[r1] = True
                merged[r2] = True
                new_rank -= 2
                continue

            # Check if combined tensor is rank-1
            # A tensor is rank-1 if T[a,b,c] = u[a]*v[b]*w[c] for some u,v,w
            # Check by looking at the structure: all entries must factorize
            keys = list(combined.keys())
            if len(keys) <= 1:
                # Trivially rank-1
                # Find u, v, w
                if len(keys) == 1:
                    (a, b, c), val = keys[0], combined[keys[0]]
                    # We need u[a]*v[b]*w[c] = val
                    # Set u[a]=val, v[b]=1, w[c]=1
                    new_U[a, new_rank-1] = val
                    new_V[b, new_rank-1] = 1
                    new_W[c, new_rank-1] = 1
                    merged[r1] = True
                    merged[r2] = True
                    new_rank -= 1
                    continue

            # For larger tensors, check rank-1 by flattening
            # Flatten as matrix of shape (n*m) x (m*p*p*n)
            # But this is expensive. Use a heuristic: check if the tensor
            # can be written as outer product.

            # Pick a reference entry
            ref_key = keys[0]
            ref_a, ref_b, ref_c = ref_key
            ref_val = combined[ref_key]

            # Try to factor: T[a,b,c] = T[ref_a,ref_b,ref_c] * (T[a,ref_b,ref_c]/T[ref_a,ref_b,ref_c]) * ...
            # This is complex. Skip for now.

    # If we found any merges, update
    if new_rank < best_rank:
        # Rebuild from new_U, new_V, new_W up to new_rank
        # But we need to verify
        T_test = np.einsum('ar,br,cr->abc', new_U[:n*m, :new_rank].astype(np.float64),
                          new_V[:m*p, :new_rank].astype(np.float64),
                          new_W[:p*n, :new_rank].astype(np.float64))
        if np.allclose(T_test, target, atol=1e-10):
            best_rank = new_rank
            best_U = new_U[:, :new_rank].copy()
            best_V = new_V[:, :new_rank].copy()
            best_W = new_W[:, :new_rank].copy()

    if time.time() - start > DEADLINE:
        rank = n * m * p
        U_f = np.zeros((n*m, rank), dtype=np.int64)
        V_f = np.zeros((m*p, rank), dtype=np.int64)
        W_f = np.zeros((p*n, rank), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j, c] = 1
                    V_f[j*p+k, c] = 1
                    W_f[k*n+i, c] = 1
                    c += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
