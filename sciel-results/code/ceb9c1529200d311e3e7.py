import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_terms():
    # 7 bilinear forms for 2x2 product P = X Y
    # X indices 0:00 1:01 2:10 3:11 ; same for Y and P
    terms = []
    # M1 = (X00+X11)(Y00+Y11); P00 += M1, P11 += M1
    terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
    # M2 = (X10+X11)Y00; P10 += M2, P11 -= M2
    terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
    # M3 = X00(Y01-Y11); P01 += M3, P11 += M3
    terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
    # M4 = X11(Y10-Y00); P00 += M4, P10 += M4
    terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
    # M5 = (X00+X01)Y11; P00 -= M5, P01 += M5
    terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
    # M6 = (X10-X00)(Y00+Y01); P11 += M6
    terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
    # M7 = (X01-X11)(Y10+Y11); P00 += M7
    terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
    return terms


def _map_u(loc, a_off):
    out = {}
    for li, c in loc.items():
        r, cc = li // 2, li % 2
        g = r * 4 + (a_off * 2 + cc)
        out[g] = out.get(g, 0) + c
    return out


def _map_v(loc, b_row, b_col):
    out = {}
    for li, c in loc.items():
        r, cc = li // 2, li % 2
        g = (b_row * 2 + r) * 4 + (b_col * 2 + cc)
        out[g] = out.get(g, 0) + c
    return out


def _map_w(loc, c_off):
    # W[k*n + i], k=col, i=row
    out = {}
    for li, c in loc.items():
        r, cc = li // 2, li % 2
        gcol = c_off * 2 + cc
        g = gcol * 2 + r
        out[g] = out.get(g, 0) + c
    return out


def _build_blocks():
    base = _strassen_terms()
    # blocks: (a_off, b_row, b_col, c_off)
    blocks = [(0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1), (1, 1, 1, 1)]
    block_terms = []
    for (ao, br, bc, co) in blocks:
        bt = []
        for (u_loc, v_loc, w_loc) in base:
            bt.append((_map_u(u_loc, ao), _map_v(v_loc, br, bc), _map_w(w_loc, co)))
        block_terms.append(bt)
    return block_terms


def _assemble(block_terms, target_block, coeff):
    # coeff: list of 7 floats, block 3's forms expressed as sum coeff[t]*form_t(block3)
    # returns U,V,W of shape (8,28),(16,28),(8,28)
    n, m, p = 2, 4, 4
    U = np.zeros((n * m, 28), dtype=float)
    V = np.zeros((m * p, 28), dtype=float)
    W = np.zeros((p * n, 28), dtype=float)
    for b in range(4):
        if b == target_block:
            continue
        for t in range(7):
            col = b * 7 + t
            u_loc, v_loc, w_loc = block_terms[b][t]
            for idx, c in u_loc.items():
                U[idx, col] += c
            for idx, c in v_loc.items():
                V[idx, col] += c
            for idx, c in w_loc.items():
                W[idx, col] += c
    # block 3 = target: express as linear combo of block3's own 7 forms
    for t in range(7):
        col = target_block * 7 + t
        if coeff[t] == 0:
            continue
        u_loc, v_loc, w_loc = block_terms[target_block][t]
        for idx, c in u_loc.items():
            U[idx, col] += coeff[t] * c
        for idx, c in v_loc.items():
            V[idx, col] += coeff[t] * c
        for idx, c in w_loc.items():
            W[idx, col] += coeff[t] * c
    return U, V, W


def _matmul_check(U, V, W, A, B, C, tol=1e-6):
    n, m, p = 2, 4, 4
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    return np.allclose(C, Cf, atol=tol)


def _is_half_integer(a, tol=1e-9):
    # check a is in (1/2) Z
    b = 2.0 * a
    return np.allclose(b, np.round(b), atol=tol)


def _try_rank27(deadline):
    # Build full 28-form tensor T (8,16,8). Form k contributes T[:, :, k] = u_k v_k^T w_k^T
    n, m, p = 2, 4, 4
    block_terms = _build_blocks()
    T = np.zeros((n * m, m * p, p * n))
    for b in range(4):
        for t in range(7):
            k = b * 7 + t
            u_loc, v_loc, w_loc = block_terms[b][t]
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            for idx, c in u_loc.items():
                u[idx] = c
            for idx, c in v_loc.items():
                v[idx] = c
            for idx, c in w_loc.items():
                w[idx] = c
            T[:, :, k] = np.einsum('i,j,k->ijk', u, v, w)

    target = 3  # block index to try to eliminate
    target_forms = list(range(target * 7, target * 7 + 7))
    other_forms = [k for k in range(28) if k not in target_forms]

    # For each target form k, we need T[:, :, k] = sum_{j in other} c_j T[:, :, j]
    # => reshape to (8*16, 28): rows = (i,j) pairs, cols = form index
    A_mat = np.zeros((n * m * m * p, 28))
    for k in range(28):
        A_mat[:, k] = T[:, :, k].reshape(-1)
    A_sub = A_mat[:, other_forms]  # (128, 21)

    found = None
    for k in target_forms:
        if time.time() > deadline:
            break
        b_vec = A_mat[:, k]
        # solve least squares
        try:
            sol, res, rank, sv = np.linalg.lstsq(A_sub, b_vec, rcond=None)
        except Exception:
            continue
        # check residual
        r = np.linalg.norm(A_sub @ sol - b_vec)
        if r > 1e-8:
            continue
        # check half-integer
        if not _is_half_integer(sol):
            continue
        found = (k, sol)
        break
    if found is None:
        return None
    k, sol = found
    # Build coeff vector for block 3's 7 forms (in order t=0..6)
    # sol is indexed by other_forms; we need to map back to the 7 block3 forms.
    # Actually coeff[t] multiplies form (target*7 + t). We need coeff for each of the 7.
    # But we solved T[:, :, k] = sum_{j in other} sol[j] T[:, :, j].
    # We want to REPLACE form k (one of block3's 7) with this combo.
    # So in the assembled matrix, column k becomes sum_{j in other} sol[j] * (column j's contribution).
    # That means: keep all 28 columns, but column k = sum sol[j]*col_j.
    # Equivalent: set column k to that combo and zero out... no, we want rank 27.
    # The decomposition: for all columns except k, use original. Column k = combo.
    # This gives 28 columns but column k is dependent => rank <= 27.
    # But the interface expects rank = number of columns. We must DROP column k and keep 27.
    # To drop column k: the product sum over columns must be unchanged.
    # Original: sum_{all 28} col_c. New: sum_{c != k} col_c + col_k_combo.
    # col_k_combo = sum_{j in other} sol[j] col_j.
    # So new sum = sum_{c != k} col_c + sum_{j in other} sol[j] col_j.
    # Since k not in other, and other = all except target_forms... wait other includes the other 6 block3 forms.
    # Let me just build: 27 columns = all except k. For each kept column c, its coefficient is 1,
    #   PLUS if c in other, add sol[j] where j=c.
    # So kept column c has multiplier (1 + sol[c] if c in other else 1).
    # Let's verify: sum_{c != k} (1 + sol_c_if_in_other) * col_c
    #   = sum_{c != k} col_c + sum_{c in other} sol_c col_c
    #   = (sum_{all} col_c - col_k) + sum_{j in other} sol_j col_j
    #   = sum_{all} col_c - col_k + col_k   (since col_k = sum sol_j col_j)
    #   = sum_{all} col_c.  Correct!
    # So we keep 27 columns, each with a multiplier.
    U, V, W = _assemble(block_terms, target, [0.0] * 7)  # placeholder, we'll do it manually
    # Manual assembly:
    U = np.zeros((n * m, 27), dtype=float)
    V = np.zeros((m * p, 27), dtype=float)
    W = np.zeros((p * n, 27), dtype=float)
    col_map = {}  # new_col -> (old_col, multiplier)
    new_idx = 0
    for c in range(28):
        if c == k:
            continue
        mult = 1.0
        if c in other_forms:
            j_pos = other_forms.index(c)
            mult += sol[j_pos]
        col_map[new_idx] = (c, mult)
        new_idx += 1
    for new_c, (old_c, mult) in col_map.items():
        b = old_c // 7
        t = old_c % 7
        u_loc, v_loc, w_loc = block_terms[b][t]
        for idx, c2 in u_loc.items():
            U[idx, new_c] += mult * c2
        for idx, c2 in v_loc.items():
            V[idx, new_c] += mult * c2
        for idx, c2 in w_loc.items():
            W[idx, new_c] += mult * c2
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100.0  # safety margin before 120s (and 150s kill)

    # Fallback: rank 28
    block_terms = _build_blocks()
    U28 = np.zeros((n * m, 28), dtype=int)
    V28 = np.zeros((m * p, 28), dtype=int)
    W28 = np.zeros((p * n, 28), dtype=int)
    for b in range(4):
        for t in range(7):
            col = b * 7 + t
            u_loc, v_loc, w_loc = block_terms[b][t]
            for idx, c in u_loc.items():
                U28[idx, col] += c
            for idx, c in v_loc.items():
                V28[idx, col] += c
            for idx, c in w_loc.items():
                W28[idx, col] += c

    # Try rank 27
    result27 = None
    if time.time() < deadline:
        try:
            result27 = _try_rank27(deadline)
        except Exception:
            result27 = None

    if result27 is not None:
        U, V, W = result27
        # Verify
        A = np.arange(n * m).reshape(n, m) + 1.0
        B = np.arange(m * p).reshape(m, p) + 2.0
        C = A @ B
        if _matmul_check(U, V, W, A, B, C):
            # Second verification with different random matrices
            rng = np.random.default_rng(42)
            for _ in range(5):
                A2 = rng.standard_normal((n, m))
                B2 = rng.standard_normal((m, p))
                C2 = A2 @ B2
                if not _matmul_check(U, V, W, A2, B2, C2):
                    result27 = None
                    break
        if result27 is not None:
            U, V, W = result27
            # Convert to int if possible, else keep as float (half-integers)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
