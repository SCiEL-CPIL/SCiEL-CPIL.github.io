"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition.

Combines:
- Structured random search + Adam polishing (current program structure)
- Winograd / cyclic-group Fourier basis construction over small finite fields
  to produce low-rank integer/half-integer candidate decompositions
- Coordinate-descent local refinement on the half-integer lattice
- Fallback to schoolbook rank-36

The evaluator measures raw rank (lower is better). Score = seed / rank.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 110s deadline, 10s safety margin


def _time_left():
    return _DEADLINE - time.time()


def _schoolbook():
    """Rank-36 schoolbook construction."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _evaluate(U, V, W, n, m, p):
    """Compute max |sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1| over all (i,j,k)."""
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _is_exact(U, V, W, n, m, p, tol=1e-9):
    return _evaluate(U, V, W, n, m, p) < tol


def _try_rank_candidate(U, V, W, n, m, p):
    """Check if a candidate (list-of-lists) is exact. Returns True if so."""
    return _is_exact(U, V, W, n, m, p)


# ---------------------------------------------------------------------------
# Winograd / cyclic-group construction
# ---------------------------------------------------------------------------

def _winograd_construction(n, m, p, deadline):
    """
    Attempt to construct a low-rank decomposition by mapping the matmul tensor
    to a cyclic group structure over Z/mod.

    The key insight: for the all-ones tensor T[i,j,k]=1, we can use the fact
    that the product of three DFT vectors gives a Kronecker delta. By choosing
    appropriate group homomorphisms from Z_L to the index sets, we can decompose
    the constant-1 tensor into a sum of rank-1 terms.

    Strategy:
    1. Choose a cyclic group of order L.
    2. For each "frequency" r in 0..L-1, define:
       U_r[i*m+j] = omega^{r * f(i,j)}
       V_r[j*p+k] = omega^{r * g(j,k)}
       W_r[k*n+i] = omega^{r * h(k,i)}
    3. Choose f, g, h such that f(i,j) + g(j,k) + h(k,i) = 0 mod L
       for all valid (i,j,k), and the sum over r gives 1.

    Since we need integer/half-integer entries, we use real-valued DFT
    (cosine/sine) or work with specific small moduli where the DFT has
    integer structure (e.g., Hadamard for L=4,8,16).
    """
    if time.time() > deadline:
        return None

    # Try Hadamard-based construction (integer entries, L = power of 2)
    for L in [4, 8, 16]:
        if time.time() > deadline:
            break

        # Hadamard matrix H of order L: H[a,b] = (-1)^{popcount(a&b)}
        H = np.ones((L, L))
        for a in range(L):
            for b in range(L):
                # Count common set bits
                x = a & b
                # Popcount
                pc = 0
                while x:
                    pc += 1
                    x &= x - 1
                H[a, b] = 1.0 if pc % 2 == 0 else -1.0

        # H * H^T = L * I, so (1/L) * H * H^T = I
        # This means: sum_{r=0}^{L-1} H[a,r] * H[b,r] = L * delta(a,b)
        # And: sum_{r=0}^{L-1} H[a,r] * H[b,r] * H[c,r] = L * delta(a,b,c) in some sense

        # For the matmul tensor, we need:
        # T[i,j,k] = 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

        # Map (i,j) -> a(i,j), (j,k) -> b(j,k), (k,i) -> c(k,i) in Z_L
        # such that a(i,j) + b(j,k) + c(k,i) = 0 mod L for all (i,j,k)
        # Then sum_r H[a,r]*H[b,r]*H[c,r] = L * delta(a+b+c=0) = L * 1 = L
        # So we need U[i*m+j,r] = H[a(i,j), r] / L^{1/3}, etc.
        # But L^{1/3} is not integer in general.

        # Alternative: use the identity directly.
        # If a+b+c = 0 mod L, then H[a,r]*H[b,r]*H[c,r] = H[a,r]*H[b,r]*H[-(a+b),r]
        # = H[a,r]*H[b,r]*H[a+b,r] (since H[-x,r] = H[x,r] for Hadamard)
        # = H[a,r]^2 * H[b,r] * H[b,r] / H[b,r] ... this is getting complicated.

        # Let's try a direct approach: enumerate all possible mappings
        # a(i,j) = alpha*i + beta*j mod L
        # b(j,k) = gamma*j + delta*k mod L
        # c(k,i) = epsilon*k + zeta*i mod L
        # and require a+b+c = 0 for all (i,j,k).
        # This gives: (alpha+zeta)*i + (beta+gamma)*j + (delta+epsilon)*k = 0 mod L
        # for all i in [0,n), j in [0,m), k in [0,p).
        # So we need alpha+zeta = 0, beta+gamma = 0, delta+epsilon = 0 mod L.

        # This is always satisfiable. Now the question is whether the resulting
        # decomposition has the right structure.

        # For each valid mapping, construct U, V, W and check.
        # a(i,j) = alpha*i + beta*j
        # b(j,k) = -beta*j + delta*k
        # c(k,i) = -delta*k - alpha*i

        for alpha in range(L):
            if time.time() > deadline:
                break
            for beta in range(L):
                if time.time() > deadline:
                    break
                for delta in range(L):
                    if time.time() > deadline:
                        break

                    # Build the mapping
                    a_map = {}
                    b_map = {}
                    c_map = {}
                    for i in range(n):
                        for j in range(m):
                            a_map[(i, j)] = (alpha * i + beta * j) % L
                    for j in range(m):
                        for k in range(p):
                            b_map[(j, k)] = (-beta * j + delta * k) % L
                    for k in range(p):
                        for i in range(n):
                            c_map[(k, i)] = (-delta * k - alpha * i) % L

                    # Check that a+b+c = 0 mod L for all (i,j,k)
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if (a_map[(i, j)] + b_map[(j, k)] + c_map[(k, i)]) % L != 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        continue

                    # Construct U, V, W using Hadamard rows
                    rank = L
                    U = np.zeros((n * m, rank))
                    V = np.zeros((m * p, rank))
                    W = np.zeros((p * n, rank))

                    for i in range(n):
                        for j in range(m):
                            U[i * m + j] = H[a_map[(i, j)]]
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k] = H[b_map[(j, k)]]
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i] = H[c_map[(k, i)]]

                    # Scale: we need sum_r U*V*W = 1, but currently it's L.
                    # So divide one factor by L.
                    # But we need integer/half-integer entries.
                    # L is 4, 8, or 16. Dividing by L gives fractions.
                    # Instead, try to absorb the scaling differently.

                    # Check if the unscaled version works (it won't, gives L)
                    err = _evaluate(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                    if err < 1e-9:
                        return U.tolist(), V.tolist(), W.tolist()

                    # Try dividing U by L
                    U_scaled = U / L
                    U_scaled = np.round(U_scaled * 2.0) / 2.0
                    err = _evaluate(U_scaled.tolist(), V.tolist(), W.tolist(), n, m, p)
                    if err < 1e-9:
                        return U_scaled.tolist(), V.tolist(), W.tolist()

                    # Try dividing V by L
                    V_scaled = V / L
                    V_scaled = np.round(V_scaled * 2.0) / 2.0
                    err = _evaluate(U.tolist(), V_scaled.tolist(), W.tolist(), n, m, p)
                    if err < 1e-9:
                        return U.tolist(), V_scaled.tolist(), W.tolist()

                    # Try dividing W by L
                    W_scaled = W / L
                    W_scaled = np.round(W_scaled * 2.0) / 2.0
                    err = _evaluate(U.tolist(), V.tolist(), W_scaled.tolist(), n, m, p)
                    if err < 1e-9:
                        return U.tolist(), V.tolist(), W_scaled.tolist()

    return None


def _fourier_basis_decomposition(n, m, p, deadline):
    """
    Try to build a decomposition using Fourier basis vectors.

    For the matmul tensor T[i,j,k]=1 (with the W[k*n+i] convention),
    we can write:
      T[i,j,k] = (1/L) * sum_{r=0}^{L-1} omega^{r*(i+j+k)} * omega^{-r*(i+j+k)}
    but this doesn't match the U,V,W structure.

    Instead, note that:
      1 = (1/L) * sum_{r=0}^{L-1} sum_{s=0}^{L-1} omega^{r*i} * omega^{s*j} * omega^{-(r+s)*k}
    if we can make the indices work out.

    This is getting complex. Let's try a more direct approach:
    Use the fact that for small dimensions, we can enumerate structured patterns.
    """
    if time.time() > deadline:
        return None

    # Try specific structured patterns with small integer/half-integer entries
    # Pattern: each rank dimension handles a "diagonal" or "block" of the tensor

    for rank in range(12, 36):
        if time.time() > deadline:
            break

        # Try a "cyclic" pattern:
        # U[i*m+j, r] = 1 if (i*m+j) % rank == r else 0  (too sparse, won't work)

        # Better: use a structured assignment based on (i+j+k) mod rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for r in range(rank):
            # Assign this rank to handle (i,j,k) where (i + j + k) % rank == r
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i + j + k) % rank == r:
                            U[i * m + j, r] += 1.0
                            V[j * p + k, r] += 1.0
                            W[k * n + i, r] += 1.0

        # Normalize: each (i,j,k) should sum to 1, so divide by count
        # But this might not give integer/half-integer entries.
        # Check if it works as-is:
        err = _evaluate(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        if err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()

    return None


# ---------------------------------------------------------------------------
# Adam polishing (from current program, kept)
# ---------------------------------------------------------------------------

def _smooth_polish(U, V, W, n, m, p, deadline, tol=1e-9, max_iters=300):
    """Smooth-surrogate polish using Adam optimizer."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]

    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    def _obj_and_grad(Ua, Va, Wa):
        loss = 0.0
        gU = np.zeros_like(Ua)
        gV = np.zeros_like(Va)
        gW = np.zeros_like(Wa)
        for (i, j, k) in triples:
            ui = i * m + j
            vj = j * p + k
            wk = k * n + i
            s = float(np.dot(Ua[ui], Va[vj] * Wa[wk]))
            d = s - 1.0
            loss += d * d
            gU[ui] += 2.0 * d * (Va[vj] * Wa[wk])
            gV[vj] += 2.0 * d * (Ua[ui] * Wa[wk])
            gW[wk] += 2.0 * d * (Ua[ui] * Va[vj])
        return loss, gU, gV, gW

    b1, b2, eps = 0.9, 0.999, 1e-8
    mU = np.zeros_like(Ua)
    vU = np.zeros_like(Ua)
    mV = np.zeros_like(Va)
    vV = np.zeros_like(Va)
    mW = np.zeros_like(Wa)
    vW = np.zeros_like(Wa)
    t = 0
    lr = 0.05

    cur_loss, _, _, _ = _obj_and_grad(Ua, Va, Wa)
    best_loss = cur_loss
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()

    for it in range(max_iters):
        if time.time() > deadline:
            break
        t += 1
        loss, gU, gV, gW = _obj_and_grad(Ua, Va, Wa)

        mU = b1 * mU + (1 - b1) * gU
        vU = b2 * vU + (1 - b2) * gU ** 2
        mU_hat = mU / (1 - b1 ** t)
        vU_hat = vU / (1 - b2 ** t)

        mV = b1 * mV + (1 - b1) * gV
        vV = b2 * vV + (1 - b2) * gV ** 2
        mV_hat = mV / (1 - b1 ** t)
        vV_hat = vV / (1 - b2 ** t)

        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW ** 2
        mW_hat = mW / (1 - b1 ** t)
        vW_hat = vW / (1 - b2 ** t)

        Ua -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        Va -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        Wa -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        cur_loss, _, _, _ = _obj_and_grad(Ua, Va, Wa)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        if cur_loss < 1e-14:
            break

    # Snap to half-integers
    Ua = np.round(best_U * 2.0) / 2.0
    Va = np.round(best_V * 2.0) / 2.0
    Wa = np.round(best_W * 2.0) / 2.0

    U = Ua.tolist()
    V = Va.tolist()
    W = Wa.tolist()
    err = _evaluate(U, V, W, n, m, p)
    if err < tol:
        return U, V, W
    return None


# ---------------------------------------------------------------------------
# Coordinate descent on half-integer lattice (from inspiration 2)
# ---------------------------------------------------------------------------

def _coordinate_descent(U, V, W, n, m, p, deadline, max_iters=2000):
    """Local search: try single-coordinate half-integer steps."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]

    def _residual():
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    cur_res = _residual()
    if cur_res < 1e-9:
        return Ua.tolist(), Va.tolist(), Wa.tolist()

    for it in range(max_iters):
        if time.time() > deadline:
            break

        factor_choice = random.randint(0, 2)
        if factor_choice == 0:
            mat, nrows = Ua, n * m
        elif factor_choice == 1:
            mat, nrows = Va, m * p
        else:
            mat, nrows = Wa, p * n

        row = random.randint(0, nrows - 1)
        col = random.randint(0, rank - 1)
        step = random.choice([-0.5, 0.5, -1.0, 1.0])

        old_val = mat[row, col]
        mat[row, col] = old_val + step

        new_res = _residual()
        if new_res < cur_res:
            cur_res = new_res
            if cur_res < 1e-9:
                return Ua.tolist(), Va.tolist(), Wa.tolist()
        else:
            mat[row, col] = old_val

    return None


# ---------------------------------------------------------------------------
# Structured initialization (from current program)
# ---------------------------------------------------------------------------

def _structured_init(n, m, p, rank, seed=0):
    """Create a structured initialization that exploits matmul structure."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r in range(rank):
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p

        if rng.random() < 0.3:
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)

        U[i * m + j, r] = 1.0
        V[j * p + k, r] = 1.0
        W[k * n + i, r] = 1.0

    U += rng.normal(0, 0.1, U.shape)
    V += rng.normal(0, 0.1, V.shape)
    W += rng.normal(0, 0.1, W.shape)

    return U.tolist(), V.tolist(), W.tolist()


def _try_structured_rank(n, m, p, rank, deadline, num_restarts=50):
    """Attempt to find a rank-r decomposition via structured init + Adam + coord descent."""
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break

        U, V, W = _structured_init(n, m, p, rank, seed=restart)

        # Polish with Adam
        result = _smooth_polish(U, V, W, n, m, p, deadline)
        if result is not None:
            return result

        # Try coordinate descent on the snapped result
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        Ua = np.round(Ua * 2.0) / 2.0
        Va = np.round(Va * 2.0) / 2.0
        Wa = np.round(Wa * 2.0) / 2.0

        result = _coordinate_descent(Ua.tolist(), Va.tolist(), Wa.tolist(), n, m, p, deadline)
        if result is not None:
            return result

    return None


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def _try_low_rank_search(n, m, p, deadline):
    """Search for low-rank decompositions starting from a reasonable lower bound."""
    for rank in range(14, 36):
        if time.time() > deadline:
            break

        num_restarts = 40 if rank <= 20 else 25
        result = _try_structured_rank(n, m, p, rank, deadline, num_restarts=num_restarts)
        if result is not None:
            return result

    return None


def _try_fourier_patterns(n, m, p, deadline):
    """Try Fourier/cyclic-group based structured patterns."""
    if time.time() > deadline:
        return None

    # Try cyclic patterns with different group orders
    for L in [12, 15, 20, 24, 30]:
        if time.time() > deadline:
            break

        for rank in range(max(12, L - 2), min(36, L + 4)):
            if time.time() > deadline:
                break

            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))

            # Assign each (i,j,k) to a rank via (i + j + k) % rank
            # and use Fourier-like weights
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = (i + j + k) % rank
                        # Use a simple weight: 1.0
                        U[i * m + j, r] += 1.0 / L
                        V[j * p + k, r] += 1.0
                        W[k * n + i, r] += 1.0 / L

            # Normalize to get integer/half-integer
            U = np.round(U * 2.0) / 2.0
            V = np.round(V * 2.0) / 2.0
            W = np.round(W * 2.0) / 2.0

            err = _evaluate(U.tolist(), V.tolist(), W.tolist(), n, m, p)
            if err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()

    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matmul.

    Returns {'U', 'V', 'W'} with factor matrices of shapes:
    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (12, rank)
    W: (p*n, rank) = (12, rank)

    Entries are integer or half-integer.
    """
    n, m, p = DIMENSIONS

    # Try Winograd/Hadamard-based construction first (fast, structured)
    result = _winograd_construction(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Try Fourier/cyclic patterns next
    result = _try_fourier_patterns(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Try structured search for low-rank solutions
    result = _try_low_rank_search(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank-36
    return _schoolbook()
