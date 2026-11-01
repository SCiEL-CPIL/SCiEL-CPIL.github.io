"""Tensor decomposition for 2x2 x 2x9 matrix multiplication."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 130  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _build_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _validate(U, V, W, n, m, p, atol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T, T_recon, atol=atol)


def _strassen_block(T_block, u_base, v_base, w_base):
    """
    Add Strassen (rank 7) decomposition for a 2x2x2 block.
    T_block has shape (2,2,2) with T_block[i,j,k]=1 (full 2x2x2 multiplication).
    u_base: index offset in U (U has 4 rows: i*m+j)
    v_base: (v_row0, v_row1) base indices for the 2 rows of B block
    w_base: base index for W (w = k*n+i)
    Returns list of (u_vec, v_vec, w_vec) each of appropriate length.
    """
    # Strassen terms for 2x2x2:
    # P0=(A00+A11)(B00+B11)
    # P1=(A10+A11)B00
    # P2=A00(B01-B11)
    # P3=A11(B10-B00)
    # P4=(A00+A01)B11
    # P5=(A10-A00)(B00+B01)
    # P6=(A01-A11)(B10+B11)
    # C00=P0-P1-P4+P6, C01=P4+P6, C10=P1+P5, C11=P0-P2+P3-P5
    # U rows: 0=A00,1=A01,2=A10,3=A11
    # V rows: 0=B00,1=B01,2=B10,3=B11
    # W rows: 0=C00,1=C01,2=C10,3=C11
    terms = [
        # (U coeffs, V coeffs, W coeffs)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1), (1, 1), (2, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, -1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1), (2, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)]),
    ]
    out = []
    for uc, vc, wc in terms:
        u = np.zeros(4)
        for idx, c in uc:
            u[idx] += c
        v = np.zeros(4)
        for idx, c in vc:
            v[idx] += c
        w = np.zeros(4)
        for idx, c in wc:
            w[idx] += c
        out.append((u, v, w))
    return out


def _greedy_rank1(T, n, m, p, max_iter=2000):
    """
    Greedily extract rank-1 terms from residual tensor T.
    Returns list of (u, v, w) vectors.
    """
    T = T.copy()
    terms = []
    for it in range(max_iter):
        if _elapsed() > _TIMEOUT:
            break
        # Find max entry
        idx = np.unravel_index(np.argmax(np.abs(T)), T.shape)
        if abs(T[idx]) < 1e-9:
            break
        i, j, k = idx
        # Solve for u (len n*m), v (len m*p), w (len p*n) with u_i v_j w_k = T[i,j,k]
        # and u ⊗ v ⊗ w = T (residual).
        # Strategy: set u_i=1, v_j=1, then w_k = T[i,j,k]. Then for all (i',j',k'),
        # we need u_{i'} v_{j'} w_{k'} = T[i',j',k'].
        # This is a rank-1 approximation. Use: u_{i'} = T[i',j,k] / (v_j w_k) but v_j=1,w_k=T[i,j,k]
        # Let's use a cleaner approach: pick the max entry, set it as the "anchor".
        # u = T[:, j, k] / (v_j * w_k). Set v_j=1, w_k=1 -> u = T[:,j,k].
        # Then v = T[i, :, k] / (u_i * w_k) = T[i,:,k]/u_i.
        # Then w = T[i, j, :] / (u_i * v_j) = T[i,j,:]/u_i.
        u = T[:, j, k].copy()
        if abs(u[i]) < 1e-12:
            # pivot problem; try different anchor
            continue
        u = u / u[i]
        v = T[i, :, k].copy() / u[i]
        w = T[i, j, :].copy() / u[i]
        # Now check: does u⊗v⊗w match T well? Subtract the best rank-1.
        recon = np.einsum('i,j,k->ijk', u, v, w)
        # Scale to best fit: T = alpha * recon + residual
        # alpha = <T, recon> / <recon, recon>
        num = np.sum(T * recon)
        den = np.sum(recon * recon)
        if den < 1e-12:
            continue
        alpha = num / den
        T_new = T - alpha * recon
        # If it reduces the residual significantly, accept
        if np.sum(T_new * T_new) < np.sum(T * T) * 0.999:
            T = T_new
            terms.append((alpha * u, v, w))
        else:
            break
    return terms


def _build_strassen_plus_greedy(n, m, p, max_greedy=60):
    """
    Build decomposition: Strassen for k=0,1 block (rank 7) + greedy for rest.
    """
    T = _build_tensor(n, m, p)
    # U shape (n*m=4, rank), V (m*p=18, rank), W (p*n=18, rank)
    U_terms = []
    V_terms = []
    W_terms = []

    # Strassen for the 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # This block's tensor entries: T[i*2+j, j*9+k, k*2+i] = 1 for k in {0,1}
    # We'll handle this block with Strassen.
    strassen = _strassen_block(None, 0, 0, 0)
    for u, v, w in strassen:
        U_terms.append(u)  # u has 4 entries: rows i*2+j for i,j in {0,1}
        # v: rows j*9+k for j in {0,1}, k in {0,1} -> indices 0,1,9,10
        v_full = np.zeros(m * p)
        v_full[0] = v[0]  # B00
        v_full[1] = v[1]  # B01
        v_full[9] = v[2]  # B10
        v_full[10] = v[3]  # B11
        V_terms.append(v_full)
        # w: rows k*2+i for k in {0,1}, i in {0,1} -> indices 0,1,2,3
        w_full = np.zeros(p * n)
        w_full[0] = w[0]  # C00
        w_full[1] = w[1]  # C01
        w_full[2] = w[2]  # C10
        w_full[3] = w[3]  # C11
        W_terms.append(w_full)

    # Subtract Strassen contribution from T
    for u, v, w in zip(U_terms, V_terms, W_terms):
        T -= np.einsum('i,j,k->ijk', u, v, w)

    # Now greedy on remaining T (k=2..8 blocks)
    greedy = _greedy_rank1(T, n, m, p, max_iter=max_greedy)
    for u, v, w in greedy:
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)

    if not U_terms:
        return None

    U = np.array(U_terms).T  # (4, rank)
    V = np.array(V_terms).T  # (18, rank)
    W = np.array(W_terms).T  # (18, rank)
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS

    best = None
    best_rank = None

    # Try Strassen + greedy
    try:
        res = _build_strassen_plus_greedy(n, m, p, max_greedy=80)
        if res is not None:
            U, V, W = res
            if _validate(U, V, W, n, m, p):
                rank = U.shape[1]
                if best_rank is None or rank < best_rank:
                    best = (U, V, W)
                    best_rank = rank
    except Exception:
        pass

    if best is not None:
        U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
