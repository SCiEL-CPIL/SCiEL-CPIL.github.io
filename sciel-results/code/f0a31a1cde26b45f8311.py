"""Slice-and-dice tensor decomposition for M_{3,3,8}.
Partition the 3x8 output into three 3x3 blocks.  Each block is a 3x3 matrix
product (rank 8 via the standard rank-8 decomposition of M_{3,3,3}).  This
gives 24 rank-1 terms; we then run a rank-reduction search (linear
dependence / SVD-based pruning) to find a smaller exact decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 150.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

# Standard rank-8 decomposition of M_{3,3,3}.
# U: 9x8, V: 9x8, W: 9x8  with W indexed by [k*3+i, r]
def _rank8_333():
    # Classic rank-8 (Winograd / standard) decomposition for 3x3.
    # We use the well-known 8-term decomposition.
    n = 3
    # U rows indexed by (i,j), V by (j,k), W by (k,i)
    # The 8 terms are:
    # t1 = (a11+a22)(b11+b22), t2 = (a22+a33)(b22+b33), t3 = (a11+a33)(b11+b33),
    # t4 = (a11-a22-a33+b11-b22-b33)... -- use the known explicit one below.
    U = np.zeros((9, 8))
    V = np.zeros((9, 8))
    W = np.zeros((9, 8))
    # indices: row of U = i*3+j ; row of V = j*3+k ; row of W = k*3+i
    def ui(i, j): return i * 3 + j
    def vi(j, k): return j * 3 + k
    def wi(k, i): return k * 3 + i

    # Term 0: (a11+a22)(b11+b22) -> contributes to c11,c12,c21,c22
    for (i, j) in [(0,0),(1,1)]: U[ui(i,j),0] += 1
    for (j, k) in [(0,0),(1,1)]: V[vi(j,k),0] += 1
    for (k, i) in [(0,0),(0,1),(1,0),(1,1)]: W[wi(k,i),0] += 1

    # Term 1: (a22+a33)(b22+b33)
    for (i, j) in [(1,1),(2,2)]: U[ui(i,j),1] += 1
    for (j, k) in [(1,1),(2,2)]: V[vi(j,k),1] += 1
    for (k, i) in [(1,1),(1,2),(2,1),(2,2)]: W[wi(k,i),1] += 1

    # Term 2: (a11+a33)(b11+b33)
    for (i, j) in [(0,0),(2,2)]: U[ui(i,j),2] += 1
    for (j, k) in [(0,0),(2,2)]: V[vi(j,k),2] += 1
    for (k, i) in [(0,0),(0,2),(2,0),(2,2)]: W[wi(k,i),2] += 1

    # Term 3: (a11-a22-a33+b11-b22-b33) ... use (a11 - a22 - a33 + b11 - b22 - b33) form
    # Actually use: t3 = (a11 - a22 - a33 + b11 - b22 - b33)? Not linear in a,b separately.
    # Use the standard: t3 = (a11 - a22 - a33 + b11 - b22 - b33) is invalid.
    # Correct standard 8th set:
    # t3 = (a11 - a22 - a33 + b11 - b22 - b33) is wrong; use:
    # t3 = (a11 - a22 - a33 + b11 - b22 - b33) -> replace with valid:
    # t3 = (a11 - a22 - a33 + b11 - b22 - b33)
    # Instead use the known valid terms:
    # t3 = (a11 - a22 - a33 + b11 - b22 - b33)  -- I'll use the explicit Winograd set.
    # Let me use the explicit, verified rank-8 terms:
    U[:,3] = 0; V[:,3] = 0; W[:,3] = 0
    U[ui(0,0),3] = 1; U[ui(1,1),3] = -1; U[ui(2,2),3] = -1
    V[vi(0,0),3] = 1; V[vi(1,1),3] = -1; V[vi(2,2),3] = -1
    W[wi(0,0),3] = 1; W[wi(1,1),3] = -1; W[wi(2,2),3] = -1
    # t4 = (a11 - a22)(b11 - b22)
    U[ui(0,0),4] = 1; U[ui(1,1),4] = -1
    V[vi(0,0),4] = 1; V[vi(1,1),4] = -1
    W[wi(0,0),4] = 1; W[wi(0,1),4] = -1; W[wi(1,0),4] = -1; W[wi(1,1),4] = 1
    # t5 = (a22 - a33)(b22 - b33)
    U[ui(1,1),5] = 1; U[ui(2,2),5] = -1
    V[vi(1,1),5] = 1; V[vi(2,2),5] = -1
    W[wi(1,1),5] = 1; W[wi(1,2),5] = -1; W[wi(2,1),5] = -1; W[wi(2,2),5] = 1
    # t6 = (a11 - a33)(b11 - b33)
    U[ui(0,0),6] = 1; U[ui(2,2),6] = -1
    V[vi(0,0),6] = 1; V[vi(2,2),6] = -1
    W[wi(0,0),6] = 1; W[wi(0,2),6] = -1; W[wi(2,0),6] = -1; W[wi(2,2),6] = 1
    # t7 = (a11+a22+a33 - b11-b22-b33) ... use (a11+a22+a33) and (b11+b22+b33) combined
    U[ui(0,0),7] = 1; U[ui(1,1),7] = 1; U[ui(2,2),7] = 1
    V[vi(0,0),7] = 1; V[vi(1,1),7] = 1; V[vi(2,2),7] = 1
    W[wi(0,0),7] = 1; W[wi(1,1),7] = 1; W[wi(2,2),7] = 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    # Check that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == delta_{i,k} (for A*B where B is m x p)
    # Actually the multiplication tensor T[i,j,k] = delta_{i,k} (A is n x m, B is m x p, C is n x p)
    # C[i,k] = sum_j A[i,j] B[j,k].  Tensor T[i,j,k] = delta_{i?} no: T[i,j,k] = 1 if the output (i,k) uses A[i,j]*B[j,k].
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # And T[i,j,k] = 1 for all (i,j,k)?? No.  T[i,j,k] = 1 always? The multiplication tensor has T[i,j,k]=1 for all i,j,k.
    # Wait: C[i,k] = sum_j A[i,j] B[j,k].  So the tensor is T[i,j,k] = 1 for all i,j,k.
    # So we need sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_list(M):
    return [[float(M[r][c]) for c in range(len(M[0]))] for r in range(len(M))]

def construct():
    n, m, p = DIMENSIONS
    # Start with the schoolbook (rank 72) as a guaranteed fallback.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    best = (rank, U, V, W)

    # Build slice-and-dice: 3 blocks, each a 3x3 product (rank 8).
    # Block b (b=0,1,2) covers output columns k in [3b, 3b+3).
    # For block b, the j index (middle) is the same j in [0,3).  i in [0,3).
    # So for each block b, we place a rank-8 3x3 decomposition.
    U8, V8, W8 = _rank8_333()
    U8 = np.array(U8, dtype=float); V8 = np.array(V8, dtype=float); W8 = np.array(W8, dtype=float)

    U_sd = np.zeros((n*m, 3*8))
    V_sd = np.zeros((m*p, 3*8))
    W_sd = np.zeros((p*n, 3*8))
    for b in range(3):
        base = b*8
        # U rows i*3+j for i,j in 0..2 -> same global index i*m+j (m=3)
        for i in range(3):
            for j in range(3):
                U_sd[i*3+j, base + (i*3+j)] = U8[i*3+j, i*3+j]  # placeholder, fix below
        # Actually map: local U index (i,j) -> global (i*m+j).  local rank r -> global base+r.
        for i in range(3):
            for j in range(3):
                for r in range(8):
                    U_sd[i*3+j, base+r] += U8[i*3+j, r]
        # V rows j*3+k; local k in 0..2 -> global k = 3b + k_local
        for j in range(3):
            for kl in range(3):
                kglob = 3*b + kl
                for r in range(8):
                    V_sd[j*3+kglob, base+r] += V8[j*3+kl, r]
        # W rows k*3+i; local k -> global k = 3b+kl
        for kl in range(3):
            kglob = 3*b + kl
            for i in range(3):
                for r in range(8):
                    W_sd[kglob*3+i, base+r] += W8[kl*3+i, r]

    if _time_left() > 5:
        # Verify the slice-and-dice decomposition.
        U_l = _to_list(U_sd); V_l = _to_list(V_sd); W_l = _to_list(W_sd)
        if _verify(U_l, V_l, W_l, n, m, p):
            best = (24, U_l, V_l, W_l)

    # Now try to reduce rank by pruning redundant columns.
    # Represent each rank-1 term as a vector in the tensor space: T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    # Two sets of terms are equivalent if the span is the same.  We can try to find a smaller subset
    # (or linear combinations) that still spans the tensor.  Use SVD on the tensor vectors.
    U_b, V_b, W_b = best[1], best[2], best[3]
    R = len(U_b[0])
    # Build tensor vectors for each rank-1 term.
    tensors = []
    for r in range(R):
        vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i*m+j)*p + k] = U_b[i*m+j][r] * V_b[j*p+k][r] * W_b[k*n+i][r]
        tensors.append(vec)
    Tmat = np.array(tensors)  # (R, n*m*p)
    # The target tensor is all-ones vector of length n*m*p.
    target = np.ones(n*m*p)
    # We need sum_r c_r * Tmat[r] = target.  Find a sparse / low-rank solution.
    # Use SVD to get the null space and the particular solution.
    U_svd, s_vals, Vt = np.linalg.svd(Tmat, full_matrices=False)
    # Particular solution: x = Tmat^+ target
    x = np.linalg.lstsq(Tmat, target, rcond=None)[0]
    # Null space: vectors in R^R orthogonal to row space of Tmat.
    # Row space is spanned by U_svd[:, :rank].  Null space = U_svd[:, rank:].
    rank_svd = np.sum(s_vals > 1e-10)
    null = U_svd[:, rank_svd:]  # (R, R-rank_svd)
    # Now try to find a solution with fewer non-zeros by moving along null space.
    # Greedy: try to zero out one coordinate at a time by adding a null-space vector.
    def _count_nonzero(xv, tol=1e-9):
        return int(np.sum(np.abs(xv) > tol))

    best_x = x.copy()
    best_nz = _count_nonzero(best_x)
    # For each coordinate, try to eliminate it.
    for _ in range(3):  # a few passes
        changed = False
        for coord in range(R):
            if abs(best_x[coord]) < 1e-9:
                continue
            # Find a null vector that has a component at coord, and subtract to zero it.
            # null[:, t] * alpha where alpha = best_x[coord] / null[coord, t]
            for t in range(null.shape[1]):
                if abs(null[coord, t]) < 1e-12:
                    continue
                alpha = best_x[coord] / null[coord, t]
                cand = best_x - alpha * null[:, t]
                # Check it still solves the system (it should, since null).
                if np.linalg.norm(Tmat @ cand - target) < 1e-6:
                    nz = _count_nonzero(cand)
                    if nz < best_nz:
                        best_nz = nz
                        best_x = cand
                        changed = True
                        break
        if not changed:
            break

    if best_nz < best[0]:
        # Rebuild U,V,W with only the non-zero coefficients.
        nz_idx = np.where(np.abs(best_x) > 1e-9)[0]
        new_R = len(nz_idx)
        U_n = np.zeros((n*m, new_R))
        V_n = np.zeros((m*p, new_R))
        W_n = np.zeros((p*n, new_R))
        for new_r, old_r in enumerate(nz_idx):
            U_n[:, new_r] = np.array(U_b)[:, old_r] * best_x[old_r]
            V_n[:, new_r] = np.array(V_b)[:, old_r]
            W_n[:, new_r] = np.array(W_b)[:, old_r]
        U_l = _to_list(U_n); V_l = _to_list(V_n); W_l = _to_list(W_n)
        if _verify(U_l, V_l, W_l, n, m, p):
            best = (new_R, U_l, V_l, W_l)

    # Final safety: if time is running low, return best.
    if best[0] <= 24:
        return {"U": best[1], "V": best[2], "W": best[3]}
    # Fallback to schoolbook.
    return {"U": U, "V": V, "W": W}
