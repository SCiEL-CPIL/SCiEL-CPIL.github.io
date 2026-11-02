import time
import numpy as np
import itertools

DIMENSIONS = [2, 2, 9]


def _build_rank32(n, m, p):
    """Build the rank-32 decomposition: 4 Strassen blocks + 1 trivial block."""
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0, col] = 1
        U[3, col] = 1
        V[0 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        # M2
        U[2, col] = 1
        U[3, col] = 1
        V[0 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        W[k1 * n + 1, col] = -1
        col += 1
        # M3
        U[0, col] = 1
        V[0 * p + k1, col] = 1
        V[1 * p + k1, col] = -1
        W[k1 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        # M4
        U[3, col] = 1
        V[1 * p + k0, col] = 1
        V[0 * p + k0, col] = -1
        W[k0 * n + 0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1
        # M5
        U[0, col] = 1
        U[1, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 0, col] = -1
        W[k1 * n + 0, col] = 1
        col += 1
        # M6
        U[2, col] = 1
        U[0, col] = -1
        V[0 * p + k0, col] = 1
        V[0 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        # M7
        U[1, col] = 1
        U[3, col] = -1
        V[1 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1
        V[0 * p + k, col] = 1
        W[k * n + 0, col] = 1
        col += 1

        U[1, col] = 1
        V[1 * p + k, col] = 1
        W[k * n + 0, col] = 1
        col += 1

        U[2, col] = 1
        V[0 * p + k, col] = 1
        W[k * n + 1, col] = 1
        col += 1

        U[3, col] = 1
        V[1 * p + k, col] = 1
        W[k * n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W


def _build_target(n, m, p):
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx, b_idx, c_idx] = 1
    return target


def _decomp_tensor(U, V, W):
    r = U.shape[1]
    out = np.zeros(U.shape[0] * V.shape[0] * W.shape[0])
    for c in range(r):
        out += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]).flatten()
    return out


def _verify(U, V, W, n, m, p, tol=1e-9):
    target = _build_target(n, m, p)
    decomp = _decomp_tensor(U, V, W).reshape(target.shape)
    residual = np.max(np.abs(decomp - target))
    return residual < tol, residual


def _try_remove_one_term(U, V, W, n, m, p, deadline):
    """Try to find rank-31 by checking if removing one term + adjusting others works."""
    r = U.shape[1]
    target = _build_target(n, m, p).flatten()
    
    # For each term to drop, check if the remaining r-1 terms span the target
    # in the tensor space. If the target is in the span of the remaining rank-1
    # tensors (with free coefficients), we can rescale.
    # Actually, we need exact rank-1 decomposition, so we need the target to be
    # representable as sum of r-1 rank-1 tensors.
    # 
    # A necessary condition: target must be in the linear span of the r-1
    # remaining rank-1 tensors. If yes, we get a solution (with possibly
    # non-unit scalings, but that's fine since rank-1 tensors can be scaled).
    
    for drop in range(r):
        if time.time() > deadline:
            return None
        idx = [i for i in range(r) if i != drop]
        # Build matrix: each column is a rank-1 tensor (flattened)
        # We want: sum_{c in idx} alpha_c * T_c = target
        # This is a linear system: A @ alpha = target
        A = np.zeros((target.shape[0], len(idx)))
        for j, c in enumerate(idx):
            A[:, j] = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]).flatten()
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A, target, rcond=None)
            # Check exactness
            recon = A @ alpha
            if np.max(np.abs(recon - target)) < 1e-8:
                # Build new U, V, W with r-1 columns
                new_r = r - 1
                newU = np.zeros((U.shape[0], new_r))
                newV = np.zeros((V.shape[0], new_r))
                newW = np.zeros((W.shape[0], new_r))
                for j, c in enumerate(idx):
                    newU[:, j] = U[:, c] * alpha[j] ** (1/3)
                    newV[:, j] = V[:, c] * alpha[j] ** (1/3)
                    newW[:, j] = W[:, c] * alpha[j] ** (1/3)
                # Verify
                ok, res = _verify(newU, newV, newW, n, m, p, tol=1e-8)
                if ok:
                    return newU, newV, newW
        except Exception:
            pass
    return None


def _try_merge_trivial_with_strassen(n, m, p, deadline):
    """Try to merge the trivial block (col 8) with an adjacent Strassen block.
    The idea: instead of 7 + 4 = 11 terms for cols 6,7,8, find a rank-10
    decomposition for the 2x2x3 sub-problem."""
    
    # Build the 2x2x3 sub-problem: A is 2x2, B is 2x3, C is 2x3
    # We need a decomposition of M_{2,2,3}
    # 
    # Let's try: 3 Strassen-like terms + 7 other terms = 10
    # Actually, let's just do a brute force search over small integer/half-integer
    # coefficients for a rank-10 decomposition of M_{2,2,3}.
    
    # M_{2,2,3}: A (2x2), B (2x3), C (2x3)
    # Tensor indices: a_idx in [0,3], b_idx in [0,5], c_idx in [0,5]
    # T[a_idx, b_idx, c_idx] = 1 if a_idx=i*m+j, b_idx=j*p+k, c_idx=k*n+i
    # with n=2, m=2, p=3
    # a_idx = i*2+j, b_idx = j*3+k, c_idx = k*2+i
    target = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a_idx = i * 2 + j
                b_idx = j * 3 + k
                c_idx = k * 2 + i
                target[a_idx, b_idx, c_idx] = 1
    
    target_flat = target.flatten()
    tensor_dim = target_flat.shape[0]
    
    # We'll try to find rank-10 by starting from a rank-11 (Strassen+trivial)
    # and trying to merge.
    
    # Build the rank-11 decomposition for M_{2,2,3}
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    # Strassen for cols 0,1
    def add_strassen(k0, k1, offset):
        col = offset
        U11[0, col] = 1
        U11[3, col] = 1
        V11[0 * 3 + k0, col] = 1
        V11[1 * 3 + k1, col] = 1
        W11[k0 * 2 + 0, col] = 1
        W11[k1 * 2 + 1, col] = 1
        col += 1
        U11[2, col] = 1
        U11[3, col] = 1
        V11[0 * 3 + k0, col] = 1
        W11[k0 * 2 + 1, col] = 1
        W11[k1 * 2 + 1, col] = -1
        col += 1
        U11[0, col] = 1
        V11[0 * 3 + k1, col] = 1
        V11[1 * 3 + k1, col] = -1
        W11[k1 * 2 + 0, col] = 1
        W11[k1 * 2 + 1, col] = 1
        col += 1
        U11[3, col] = 1
        V11[1 * 3 + k0, col] = 1
        V11[0 * 3 + k0, col] = -1
        W11[k0 * 2 + 0, col] = 1
        W11[k0 * 2 + 1, col] = 1
        col += 1
        U11[0, col] = 1
        U11[1, col] = 1
        V11[1 * 3 + k1, col] = 1
        W11[k0 * 2 + 0, col] = -1
        W11[k1 * 2 + 0, col] = 1
        col += 1
        U11[2, col] = 1
        U11[0, col] = -1
        V11[0 * 3 + k0, col] = 1
        V11[0 * 3 + k1, col] = 1
        W11[k1 * 2 + 1, col] = 1
        col += 1
        U11[1, col] = 1
        U11[3, col] = -1
        V11[1 * 3 + k0, col] = 1
        V11[1 * 3 + k1, col] = 1
        W11[k0 * 2 + 0, col] = 1
        col += 1
        return col
    
    col = add_strassen(0, 1, 0)
    # Trivial for col 2
    U11[0, col] = 1
    V11[0 * 3 + 2, col] = 1
    W11[2 * 2 + 0, col] = 1
    col += 1
    U11[1, col] = 1
    V11[1 * 3 + 2, col] = 1
    W11[2 * 2 + 0, col] = 1
    col += 1
    U11[2, col] = 1
    V11[0 * 3 + 2, col] = 1
    W11[2 * 2 + 1, col] = 1
    col += 1
    U11[3, col] = 1
    V11[1 * 3 + 2, col] = 1
    W11[2 * 2 + 1, col] = 1
    col += 1
    
    assert col == 11
    
    # Verify rank-11
    decomp = np.zeros(tensor_dim)
    for c in range(11):
        decomp += np.einsum('i,j,k->ijk', U11[:, c], V11[:, c], W11[:, c]).flatten()
    assert np.max(np.abs(decomp - target_flat)) < 1e-10
    
    # Now try to find rank-10 by removing one term and solving
    for drop in range(11):
        if time.time() > deadline:
            return None
        idx = [i for i in range(11) if i != drop]
        A = np.zeros((tensor_dim, len(idx)))
        for j, c in enumerate(idx):
            A[:, j] = np.einsum('i,j,k->ijk', U11[:, c], V11[:, c], W11[:, c]).flatten()
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A, target_flat, rcond=None)
            recon = A @ alpha
            if np.max(np.abs(recon - target_flat)) < 1e-8:
                # Found rank-10!
                # Build full rank-31 decomposition
                U31 = np.zeros((4, 31))
                V31 = np.zeros((18, 31))
                W31 = np.zeros((18, 31))
                
                # First 3 Strassen blocks (cols 0-1, 2-3, 4-5)
                col = 0
                for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
                    # Copy from rank-32 structure
                    U31[0, col] = 1
                    U31[3, col] = 1
                    V31[0 * 9 + k0, col] = 1
                    V31[1 * 9 + k1, col] = 1
                    W31[k0 * 2 + 0, col] = 1
                    W31[k1 * 2 + 1, col] = 1
                    col += 1
                    U31[2, col] = 1
                    U31[3, col] = 1
                    V31[0 * 9 + k0, col] = 1
                    W31[k0 * 2 + 1, col] = 1
                    W31[k1 * 2 + 1, col] = -1
                    col += 1
                    U31[0, col] = 1
                    V31[0 * 9 + k1, col] = 1
                    V31[1 * 9 + k1, col] = -1
                    W31[k1 * 2 + 0, col] = 1
                    W31[k1 * 2 + 1, col] = 1
                    col += 1
                    U31[3, col] = 1
                    V31[1 * 9 + k0, col] = 1
                    V31[0 * 9 + k0, col] = -1
                    W31[k0 * 2 + 0, col] = 1
                    W31[k0 * 2 + 1, col] = 1
                    col += 1
                    U31[0, col] = 1
                    U31[1, col] = 1
                    V31[1 * 9 + k1, col] = 1
                    W31[k0 * 2 + 0, col] = -1
                    W31[k1 * 2 + 0, col] = 1
                    col += 1
                    U31[2, col] = 1
                    U31[0, col] = -1
                    V31[0 * 9 + k0, col] = 1
                    V31[0 * 9 + k1, col] = 1
                    W31[k1 * 2 + 1, col] = 1
                    col += 1
                    U31[1, col] = 1
                    U31[3, col] = -1
                    V31[1 * 9 + k0, col] = 1
                    V31[1 * 9 + k1, col] = 1
                    W31[k0 * 2 + 0, col] = 1
                    col += 1
                
                # Last 10 terms: the rank-10 decomposition for cols 6,7,8
                for j, c in enumerate(idx):
                    newU = U11[:, c] * alpha[j] ** (1/3)
                    newV = V11[:, c] * alpha[j] ** (1/3)
                    newW = W11[:, c] * alpha[j] ** (1/3)
                    U31[:, col] = newU
                    V31[6 * 3:9 * 3, col] = newV  # rows 18-23 for B cols 6,7,8
                    W31[6 * 2:9 * 2, col] = newW  # rows 12-17 for C cols 6,7,8
                    col += 1
                
                assert col == 31
                ok, res = _verify(U31, V31, W31, 2, 2, 9, tol=1e-8)
                if ok:
                    return U31, V31, W31
        except Exception:
            pass
    return None


def _try_random_search(n, m, p, U32, V32, W32, deadline):
    """Random perturbation search for rank-31."""
    target = _build_target(n, m, p).flatten()
    r = U32.shape[1]
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
        # Drop a random term
        drop = np.random.randint(0, r)
        idx = [i for i in range(r) if i != drop]
        
        # Perturb the remaining terms slightly
        U_p = U32[:, idx].copy()
        V_p = V32[:, idx].copy()
        W_p = W32[:, idx].copy()
        
        # Add small random noise
        noise = 0.1
        U_p += noise * np.random.randn(*U_p.shape)
        V_p += noise * np.random.randn(*V_p.shape)
        W_p += noise * np.random.randn(*W_p.shape)
        
        # Try to solve for scalings
        A = np.zeros((target.shape[0], len(idx)))
        for j in range(len(idx)):
            A[:, j] = np.einsum('i,j,k->ijk', U_p[:, j], V_p[:, j], W_p[:, j]).flatten()
        
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A, target, rcond=None)
            recon = A @ alpha
            if np.max(np.abs(recon - target)) < 1e-8:
                newU = U_p * alpha ** (1/3)
                newV = V_p * alpha ** (1/3)
                newW = W_p * alpha ** (1/3)
                ok, res = _verify(newU, newV, newW, n, m, p, tol=1e-8)
                if ok:
                    return newU, newV, newW
        except Exception:
            pass
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS

    # Build the rank-32 solution
    U32, V32, W32 = _build_rank32(n, m, p)

    # Verify it works
    valid, residual = _verify(U32, V32, W32, n, m, p)
    if not valid:
        # Shouldn't happen
        pass

    # Try to find a rank-31 solution
    best_result = (U32, V32, W32)

    if time.time() < deadline - 5:
        # First try: remove one term from rank-32
        result = _try_remove_one_term(U32, V32, W32, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            valid, residual = _verify(U, V, W, n, m, p)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() < deadline - 5:
        # Try to merge trivial with Strassen
        result = _try_merge_trivial_with_strassen(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            valid, residual = _verify(U, V, W, n, m, p)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() < deadline - 5:
        # Random search
        result = _try_random_search(n, m, p, U32, V32, W32, deadline)
        if result is not None:
            U, V, W = result
            valid, residual = _verify(U, V, W, n, m, p)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return the rank-32 solution
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
