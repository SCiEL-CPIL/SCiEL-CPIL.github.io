import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    # ---------- verification ----------
    def verify(U, V, W, rank):
        U_arr = np.asarray(U, dtype=float)
        V_arr = np.asarray(V, dtype=float)
        W_arr = np.asarray(W, dtype=float)
        # vectorized: for each (i,j,k), dot(U[i*m+j], V[j*p+k]*W[k*n+i]) == 1
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    s = np.dot(u_row, V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def verify_fast(U_arr, V_arr, W_arr):
        """Vectorized verification using einsum."""
        # contribs[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        U3 = U_arr.reshape(n, m, -1)
        V3 = V_arr.reshape(m, p, -1)
        W3 = W_arr.reshape(p, n, -1)
        contribs = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3)
        return bool(np.all(np.abs(contribs - 1.0) < 1e-8))

    def build_contrib_matrix(U_arr, V_arr, W_arr):
        """contribs[i,j,k,r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        U3 = U_arr.reshape(n, m, -1)
        V3 = V_arr.reshape(m, p, -1)
        W3 = W_arr.reshape(p, n, -1)
        # contribs shape (n*m*p, rank)
        rank = U_arr.shape[1]
        contribs = np.zeros((n*m*p, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contribs[idx, :] = U3[i, j, :] * V3[j, k, :] * W3[k, i, :]
                    idx += 1
        return contribs

    # ---------- Strassen construction ----------
    def build_strassen_decomp(j_pairs, k_pairs):
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0); used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]

        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))

        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[j0, col] += 1; U[m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0, col] += 1; W[n+k1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[m+j0, col] += 1; U[m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[n+k0, col] += 1; W[n+k1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[n+k1, col] += 1; W[2*n+k1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[n+k0, col] += 1; W[2*n+k0, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[j0, col] += 1; U[j1, col] += 1
                V[j1*p+k1, col] += 1
                W[n+k0, col] -= 1; W[2*n+k1, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[m+j0, col] += 1; U[j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[2*n+k1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[j1, col] += 1; U[m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[n+k0, col] += 1
                col += 1

        for j in remaining_j:
            for k in range(p):
                U[j, col] += 1
                V[j*p+k, col] += 1
                W[k, col] += 1
                col += 1
                U[m+j, col] += 1
                V[j*p+k, col] += 1
                W[n+k, col] += 1
                col += 1

        return U, V, W, rank

    # ---------- term elimination (SVD-based) ----------
    def try_eliminate_terms(U, V, W, rank):
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        cur_rank = rank
        max_iterations = 60
        for _ in range(max_iterations):
            if time_left() < 15:
                break
            U_arr = np.array(U_list, dtype=float)
            V_arr = np.array(V_list, dtype=float)
            W_arr = np.array(W_list, dtype=float)
            contribs = build_contrib_matrix(U_arr, V_arr, W_arr)
            try:
                svd_rank = np.linalg.matrix_rank(contribs, tol=1e-9)
            except np.linalg.LinAlgError:
                svd_rank = cur_rank
            if svd_rank >= cur_rank:
                break
            removed = False
            for term_idx in range(cur_rank - 1, -1, -1):
                if time_left() < 15:
                    break
                other = [i for i in range(cur_rank) if i != term_idx]
                other_c = contribs[:, other]
                try:
                    sol, residuals, _, _ = np.linalg.lstsq(other_c, contribs[:, term_idx], rcond=1e-9)
                    if len(residuals) > 0 and residuals[0] < 1e-8:
                        U_t = U_arr.copy(); V_t = V_arr.copy(); W_t = W_arr.copy()
                        U_t[:, term_idx] = 0; V_t[:, term_idx] = 0; W_t[:, term_idx] = 0
                        if verify_fast(U_t, V_t, W_t):
                            U_list = [row[:term_idx] + row[term_idx+1:] for row in U_list]
                            V_list = [row[:term_idx] + row[term_idx+1:] for row in V_list]
                            W_list = [row[:term_idx] + row[term_idx+1:] for row in W_list]
                            cur_rank -= 1
                            removed = True
                            break
                except np.linalg.LinAlgError:
                    continue
            if not removed:
                break
        return U_list, V_list, W_list, cur_rank

    # ---------- structured ansatz: fix U,V, solve for W ----------
    def solve_for_W(U, V, target_rank):
        """Given U (n*m x R) and V (m*p x R), solve for W (p*n x R) such that
        sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all (i,j,k).
        Returns W if a solution exists, else None.
        """
        R = target_rank
        # Unknowns: W has p*n*R = 12*R unknowns
        # Equations: n*m*p = 60 equations
        # Build linear system A x = b
        # For each (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
        # Coefficient of W[k*n+i, r] in equation (i,j,k) is U[i*m+j,r]*V[j*p+k,r]
        A = np.zeros((n*m*p, p*n*R))
        b = np.ones(n*m*p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        A[eq, (k*n+i)*R + r] = U[i*m+j, r] * V[j*p+k, r]
                    eq += 1
        try:
            sol, residuals, rankA, sv = np.linalg.lstsq(A, b, rcond=1e-9)
        except np.linalg.LinAlgError:
            return None
        if len(residuals) > 0 and residuals[0] > 1e-7:
            return None
        W = sol.reshape(p*n, R)
        # Verify
        if not verify_fast(U, V, W):
            return None
        return W

    def try_structured_ansatz(max_rank, U_pattern_fn, V_pattern_fn):
        """Try ranks from some low value up to max_rank with given U,V patterns."""
        best = None
        for R in range(max_rank, 0, -1):
            if time_left() < 20:
                break
            U = U_pattern_fn(R)
            V = V_pattern_fn(R)
            if U is None or V is None:
                continue
            W = solve_for_W(U, V, R)
            if W is not None:
                # try to eliminate
                U_l, V_l, W_l, new_rank = try_eliminate_terms(U, V, W, R)
                if best is None or new_rank < best[0]:
                    best = (new_rank, U_l, V_l, W_l)
                # since we go downward, first success is best rank for this pattern
                break
        return best

    # ---------- partitions ----------
    def generate_partitions(indices, pair_size=2):
        if not indices:
            yield [], []
            return
        if len(indices) < pair_size:
            yield [], list(indices)
            return
        first = indices[0]
        rest = indices[1:]
        for pairs, singles in generate_partitions(rest, pair_size):
            yield pairs, [first] + singles
        for i in range(len(rest)):
            partner = rest[i]
            remaining = rest[:i] + rest[i+1:]
            for pairs, singles in generate_partitions(remaining, pair_size):
                yield [(first, partner)] + pairs, singles

    # ---------- main search ----------
    j_indices = list(range(m))
    k_indices = list(range(p))

    j_partitions = set()
    for pairs, singles in generate_partitions(j_indices):
        j_partitions.add((tuple(sorted(pairs)), tuple(sorted(singles))))

    k_partitions = set()
    for pairs, singles in generate_partitions(k_indices):
        k_partitions.add((tuple(sorted(pairs)), tuple(sorted(singles))))

    configurations = []
    for j_pairs_tuple, j_singles in j_partitions:
        for k_pairs_tuple, k_singles in k_partitions:
            if not j_pairs_tuple or not k_pairs_tuple:
                continue
            num_j_blocks = len(j_pairs_tuple)
            num_k_blocks = len(k_pairs_tuple)
            num_blocks = num_j_blocks * num_k_blocks
            rank = 7 * num_blocks + 2 * len(j_singles) * p
            configurations.append((list(j_pairs_tuple), list(k_pairs_tuple), rank))

    configurations.sort(key=lambda x: x[2])

    best_result = None
    best_rank = float('inf')

    # Phase 1: Strassen + elimination over configurations
    for j_pairs, k_pairs, est_rank in configurations:
        if time_left() < 25:
            break
        if est_rank >= best_rank:
            continue
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        if not verify_fast(U, V, W):
            continue
        U_l, V_l, W_l, new_rank = try_eliminate_terms(U, V, W, rank)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_l, "V": V_l, "W": W_l}

    # Phase 2: structured ansatz - sparse symmetric U,V, solve for W
    # Try to beat best_rank. Use a structured U and V that are sparse.
    if time_left() > 30:
        # Pattern A: U[i*m+j, r] = 1 if r corresponds to (i,j) pair, V similar
        # Use a "diagonal + off-diagonal" structured pattern.
        def make_U_pattern_A(R):
            # Assign each (i,j) a base set of columns
            U = np.zeros((n*m, R))
            # simple: U[i*m+j, r] = 1 for r in a block
            # Use R columns, each (i,j) gets ~R/(n*m) ones
            for idx in range(n*m):
                start = (idx * R) // (n*m)
                end = ((idx + 1) * R) // (n*m)
                if end > start:
                    U[idx, start:end] = 1.0
            return U

        def make_V_pattern_A(R):
            V = np.zeros((m*p, R))
            for idx in range(m*p):
                start = (idx * R) // (m*p)
                end = ((idx + 1) * R) // (m*p)
                if end > start:
                    V[idx, start:end] = 1.0
            return V

        res = try_structured_ansatz(min(best_rank - 1, 40), make_U_pattern_A, make_V_pattern_A)
        if res is not None and res[0] < best_rank:
            best_rank = res[0]
            best_result = {"U": res[1], "V": res[2], "W": res[3]}

    # Phase 3: try a few more structured patterns if time remains
    if time_left() > 30 and best_rank > 30:
        def make_U_pattern_B(R):
            # checkerboard: U[i*m+j, r] = 1 if (i*m+j + r) % 3 == 0
            U = np.zeros((n*m, R))
            for idx in range(n*m):
                for r in range(R):
                    if (idx + r) % 3 == 0:
                        U[idx, r] = 1.0
            return U

        def make_V_pattern_B(R):
            V = np.zeros((m*p, R))
            for idx in range(m*p):
                for r in range(R):
                    if (idx + r) % 3 == 0:
                        V[idx, r] = 1.0
            return V

        res = try_structured_ansatz(min(best_rank - 1, 35), make_U_pattern_B, make_V_pattern_B)
        if res is not None and res[0] < best_rank:
            best_rank = res[0]
            best_result = {"U": res[1], "V": res[2], "W": res[3]}

    # Fallback: schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
