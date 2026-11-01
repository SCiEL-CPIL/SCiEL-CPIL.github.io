"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition.

Design:
  - The schoolbook rank is n*m*p = 56.
  - View the 2x14 matrix B as 7 blocks of 2x2. Each block is multiplied by the
    same 2x2 matrix A. A 2x2*2x2 product has bilinear complexity 7 (Strassen),
    so 7 blocks give a rank-49 decomposition.
  - We attempt exact linear-algebraic rank reduction by finding integer linear
    dependencies among the rank-1 tensors.
"""
import numpy as np
import time
from fractions import Fraction
from math import gcd

try:
    from scipy.linalg import null_space as _scipy_null_space
except Exception:
    _scipy_null_space = None

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # hard safety margin inside the 150s wall-clock

    size_a = n * m      # 4
    size_b = m * p      # 28
    size_c = p * n      # 28

    # True multiplication tensor: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i.
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0

    # ------------------------------------------------------------------ #
    #  Verification helpers
    # ------------------------------------------------------------------ #
    def verify(U, V, W, atol=1e-9):
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        T = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
        return np.allclose(T, T_true, atol=atol)

    def schoolbook():
        rank = n * m * p  # 56
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # ------------------------------------------------------------------ #
    #  Rank-49 Strassen block decomposition
    # ------------------------------------------------------------------ #
    def strassen49():
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)

        for l in range(7):
            base = 7 * l
            k0 = 2 * l
            k1 = 2 * l + 1

            # A row/col flat indices: a00=0,a01=1,a10=2,a11=3
            # B flat: B0k0=k0, B0k1=k1, B1k0=p+k0, B1k1=p+k1
            # W flat: C0k0=k0*n, C1k0=k0*n+1, C0k1=k1*n, C1k1=k1*n+1

            c0 = base + 0
            U[0, c0] += 1; U[3, c0] += 1
            V[k0, c0] += 1; V[p + k1, c0] += 1
            W[k0 * n, c0] += 1; W[k1 * n + 1, c0] += 1

            c1 = base + 1
            U[2, c1] += 1; U[3, c1] += 1
            V[k0, c1] += 1
            W[k0 * n + 1, c1] += 1; W[k1 * n + 1, c1] -= 1

            c2 = base + 2
            U[0, c2] += 1
            V[k1, c2] += 1; V[p + k1, c2] -= 1
            W[k1 * n, c2] += 1; W[k1 * n + 1, c2] += 1

            c3 = base + 3
            U[3, c3] += 1
            V[p + k0, c3] += 1; V[k0, c3] -= 1
            W[k0 * n, c3] += 1; W[k0 * n + 1, c3] += 1

            c4 = base + 4
            U[0, c4] += 1; U[1, c4] += 1
            V[p + k1, c4] += 1
            W[k0 * n, c4] -= 1; W[k1 * n, c4] += 1

            c5 = base + 5
            U[2, c5] += 1; U[0, c5] -= 1
            V[k0, c5] += 1; V[k1, c5] += 1
            W[k1 * n + 1, c5] += 1

            c6 = base + 6
            U[1, c6] += 1; U[3, c6] -= 1
            V[p + k0, c6] += 1; V[p + k1, c6] += 1
            W[k0 * n, c6] += 1

        return U, V, W

    # ------------------------------------------------------------------ #
    #  Exact linear-algebraic rank reduction
    # ------------------------------------------------------------------ #
    def _rref_basis(M):
        """Return an integer row-echelon basis (list of Fraction rows) spanning
        the row space of the rational matrix M (m x k)."""
        m, k = M.shape
        rows = []
        for r in range(m):
            row = [Fraction(M[r, c]) for c in range(k)]
            if any(x != 0 for x in row):
                rows.append(row)
        # Gaussian elimination over the rationals.
        pivot_cols = []
        basis = []
        for i, row in enumerate(rows):
            # eliminate earlier pivots
            for j, pc in enumerate(pivot_cols):
                if row[pc] != 0:
                    factor = row[pc] / rows[j][pc]
                    row = [row[c] - factor * rows[j][c] for c in range(k)]
            # find next pivot
            pc = None
            for c in range(k):
                if row[c] != 0:
                    pc = c
                    break
            if pc is None:
                continue
            # normalize so pivot = 1
            pv = row[pc]
            row = [row[c] / pv for c in range(k)]
            rows[i] = row
            basis.append(row)
            pivot_cols.append(pc)
        # clean: zero out pivot entries in other basis rows
        for idx, pc in enumerate(pivot_cols):
            for jdx in range(len(basis)):
                if jdx != idx and basis[jdx][pc] != 0:
                    f = basis[jdx][pc]
                    basis[jdx] = [basis[jdx][c] - f * basis[idx][c] for c in range(k)]
        return basis

    def _nullspace_int(C, tol=1e-7):
        """C: (rank, d). Return list of integer null vectors (length rank)."""
        try:
            s, vh = np.linalg.svd(C, full_matrices=True)
        except np.linalg.LinAlgError:
            return []
        t = 1e-9 * (s[0] if s.size else 1.0)
        nsv = int(np.sum(s > t))
        if nsv >= C.shape[0]:
            return []
        null = vh[nsv:]  # (rank-nsv, rank)
        out = []
        for row in null:
            xi = np.rint(row)
            if np.all(np.abs(xi - row) < 1e-5):
                # normalize to smallest integer form
                g = 1
                for v in xi:
                    if v != 0:
                        g = gcd(g, int(abs(v)))
                xi = xi // g if g else xi
                # verify it's actually in the null space
                if np.linalg.norm(C @ xi) < tol * max(1.0, np.linalg.norm(C, axis=1).max()):
                    out.append(xi)
        return out

    def reduce_rank(U, V, W, max_iter=5):
        """Try to find a strictly smaller exact decomposition by exploiting
        linear dependencies among the rank-1 tensors."""
        rank = W.shape[1]
        flat_dim = size_a * size_b * size_c
        Tmat = np.array([np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).reshape(-1) 
                        for r in range(rank)], dtype=np.float64)

        # Numeric rank of the span of the rank-1 tensors.
        try:
            s = np.linalg.svd(Tmat, compute_uv=False)
        except np.linalg.LinAlgError:
            return None
        tol = 1e-9 * s[0]
        num_rank = int(np.sum(s > tol))
        if num_rank >= rank:
            return None  # all independent: no linear reduction possible

        # Build the rational basis of the span.
        try:
            M = Tmat.T  # (flat_dim, rank)
            M_round = np.rint(M)
            if not np.allclose(M_round, M, atol=1e-6):
                return None
            basis = _rref_basis(M_round.astype(object))
        except Exception:
            return None

        if not basis:
            return None
        d = len(basis)
        if d >= rank:
            return None

        # Find pivot columns for the basis.
        pivot_cols = []
        for row in basis:
            pc = None
            for c in range(len(row)):
                if row[c] != 0:
                    pc = c
                    break
            pivot_cols.append(pc)

        # Coordinates of T_r in basis: (rank, d)
        coords = np.zeros((rank, d), dtype=np.float64)
        for r in range(rank):
            for j, pc in enumerate(pivot_cols):
                coords[r, j] = float(M_round[pc, r])

        # Find integer relations and greedily eliminate tensors.
        Ucur, Vcur, Wcur = U.copy(), V.copy(), W.copy()
        kept = list(range(rank))
        progress = True
        best_rank = rank
        while progress and time.time() < deadline - 5:
            progress = False
            cur_rank = len(kept)
            if cur_rank <= d:
                break
            Ck = coords[kept, :]
            nulls_k = _nullspace_int(Ck)
            for y in nulls_k:
                # Try all possible elimination positions (unit entries)
                candidates = [idx for idx, val in enumerate(y) if val == 1 or val == -1]
                for elim in candidates:
                    oldU = Ucur[:, kept[elim]]
                    oldV = Vcur[:, kept[elim]]
                    oldW = Wcur[:, kept[elim]]
                    newU = []
                    newV = []
                    newW = []
                    newkept = []
                    for i, keep_idx in enumerate(kept):
                        if i == elim:
                            continue
                        coeff = -int(y[i])
                        if coeff == 0:
                            newU.append(Ucur[:, keep_idx])
                            newV.append(Vcur[:, keep_idx])
                            newW.append(Wcur[:, keep_idx])
                        else:
                            newU.append(Ucur[:, keep_idx] + coeff * oldU)
                            newV.append(Vcur[:, keep_idx] + coeff * oldV)
                            newW.append(Wcur[:, keep_idx] + coeff * oldW)
                        newkept.append(keep_idx)
                    newU = np.column_stack(newU)
                    newV = np.column_stack(newV)
                    newW = np.column_stack(newW)
                    if verify(newU, newV, newW):
                        Ucur, Vcur, Wcur = newU, newV, newW
                        kept = newkept
                        progress = True
                        best_rank = min(best_rank, len(kept))
                        break
                if progress:
                    break
        if len(kept) < rank:
            Uf = Ucur[:, kept]
            Vf = Vcur[:, kept]
            Wf = Wcur[:, kept]
            if verify(Uf, Vf, Wf):
                return Uf, Vf, Wf
        return None

    def reduce_rank_randomized(U, V, W, num_trials=20):
        """Randomized search: try different elimination orderings to find
        a smaller decomposition than the greedy approach."""
        rank = W.shape[1]
        flat_dim = size_a * size_b * size_c
        Tmat = np.array([np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).reshape(-1) 
                        for r in range(rank)], dtype=np.float64)

        try:
            s = np.linalg.svd(Tmat, compute_uv=False)
        except np.linalg.LinAlgError:
            return None
        tol = 1e-9 * s[0]
        num_rank = int(np.sum(s > tol))
        if num_rank >= rank:
            return None

        try:
            M = Tmat.T
            M_round = np.rint(M)
            if not np.allclose(M_round, M, atol=1e-6):
                return None
            basis = _rref_basis(M_round.astype(object))
        except Exception:
            return None

        if not basis:
            return None
        d = len(basis)
        if d >= rank:
            return None

        pivot_cols = []
        for row in basis:
            pc = None
            for c in range(len(row)):
                if row[c] != 0:
                    pc = c
                    break
            pivot_cols.append(pc)

        coords = np.zeros((rank, d), dtype=np.float64)
        for r in range(rank):
            for j, pc in enumerate(pivot_cols):
                coords[r, j] = float(M_round[pc, r])

        rng = np.random.RandomState(42)
        best_result = None
        best_rank = rank

        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            Ucur, Vcur, Wcur = U.copy(), V.copy(), W.copy()
            kept = list(range(rank))
            progress = True
            while progress and time.time() < deadline - 5:
                progress = False
                cur_rank = len(kept)
                if cur_rank <= d:
                    break
                Ck = coords[kept, :]
                nulls_k = _nullspace_int(Ck)
                if not nulls_k:
                    break
                # Randomly pick a relation and elimination position
                y = nulls_k[rng.randint(len(nulls_k))]
                candidates = [idx for idx, val in enumerate(y) if val == 1 or val == -1]
                if not candidates:
                    continue
                elim = candidates[rng.randint(len(candidates))]
                oldU = Ucur[:, kept[elim]]
                oldV = Vcur[:, kept[elim]]
                oldW = Wcur[:, kept[elim]]
                newU = []
                newV = []
                newW = []
                newkept = []
                for i, keep_idx in enumerate(kept):
                    if i == elim:
                        continue
                    coeff = -int(y[i])
                    if coeff == 0:
                        newU.append(Ucur[:, keep_idx])
                        newV.append(Vcur[:, keep_idx])
                        newW.append(Wcur[:, keep_idx])
                    else:
                        newU.append(Ucur[:, keep_idx] + coeff * oldU)
                        newV.append(Vcur[:, keep_idx] + coeff * oldV)
                        newW.append(Wcur[:, keep_idx] + coeff * oldW)
                    newkept.append(keep_idx)
                newU = np.column_stack(newU)
                newV = np.column_stack(newV)
                newW = np.column_stack(newW)
                if verify(newU, newV, newW):
                    Ucur, Vcur, Wcur = newU, newV, newW
                    kept = newkept
                    progress = True
                    if len(kept) < best_rank:
                        best_rank = len(kept)
                        Uf = Ucur[:, kept]
                        Vf = Vcur[:, kept]
                        Wf = Wcur[:, kept]
                        if verify(Uf, Vf, Wf):
                            best_result = (Uf, Vf, Wf)
            if len(kept) < best_rank:
                Uf = Ucur[:, kept]
                Vf = Vcur[:, kept]
                Wf = Wcur[:, kept]
                if verify(Uf, Vf, Wf):
                    best_rank = len(kept)
                    best_result = (Uf, Vf, Wf)

        return best_result

    # ------------------------------------------------------------------ #
    #  Main
    # ------------------------------------------------------------------ #
    U, V, W = strassen49()
    if verify(U, V, W):
        # First try greedy reduction
        reduced = reduce_rank(U, V, W)
        best_rank = U.shape[1]
        best_result = (U, V, W)
        if reduced is not None:
            Ur, Vr, Wr = reduced
            if verify(Ur, Vr, Wr) and Ur.shape[1] < best_rank:
                best_rank = Ur.shape[1]
                best_result = (Ur, Vr, Wr)
        
        # Try randomized search with remaining time
        if time.time() < deadline - 15 and best_rank > 42:
            rand_result = reduce_rank_randomized(U, V, W, num_trials=10)
            if rand_result is not None:
                Ur, Vr, Wr = rand_result
                if verify(Ur, Vr, Wr) and Ur.shape[1] < best_rank:
                    best_rank = Ur.shape[1]
                    best_result = (Ur, Vr, Wr)
        
        Ur, Vr, Wr = best_result
        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}

    # Fallback: guaranteed schoolbook rank 56.
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
