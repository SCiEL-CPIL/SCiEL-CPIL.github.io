import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _lag_basis_values_float(n, a, xs):
    """Return list of length n+1, each a numpy array of L_k^(a)(x) for k=0..n at xs (float)."""
    xs = np.asarray(xs, dtype=np.float64)
    N = int(n)
    out = [np.zeros_like(xs) for _ in range(N + 1)]
    out[0] = np.ones_like(xs)
    if N >= 1:
        out[1] = (1 + a - xs).astype(np.float64)
    for k in range(1, N):
        kp1 = k + 1
        num = (2*k + 1 + a - xs) * out[k] - (k + a) * out[k - 1]
        out[kp1] = num / kp1
    return out


def _score_float(z_list):
    """Fast float-based score. Returns (B_val, ok, q_degree, B_candidates)."""
    zs = np.asarray(z_list, dtype=np.float64)
    if np.any(~np.isfinite(zs)) or np.any(zs <= 0):
        return None
    if len(np.unique(zs)) != len(zs):
        return None
    alpha = -0.5
    # basis at 0 and at each z
    basis0 = _lag_basis_values_float(50, alpha, np.array([0.0]))
    L0 = np.array([basis0[k][0] for k in range(26)])          # L_{2k}(0)
    dL0 = np.array([_lag_derivative_float(2*k, alpha, 0.0) for k in range(26)])

    # at each z: L_{2k}(z) and L'_{2k}(z)
    zvals = []
    dzvals = []
    for z in zs:
        basis = _lag_basis_values_float(50, alpha, np.array([z]))
        Lz = np.array([basis[2*k][0] for k in range(26)])
        dLz = np.array([_lag_derivative_float(2*k, alpha, float(z)) for k in range(26)])
        zvals.append(Lz)
        dzvals.append(dLz)
    zvals = np.array(zvals)      # (12,26)
    dzvals = np.array(dzvals)    # (12,26)

    # build A (26x26), b
    A = np.zeros((26, 26))
    b = np.zeros(26)
    A[0] = L0; b[0] = 0.0
    A[1] = dL0; b[1] = 1.0
    for i in range(12):
        A[2 + 2*i] = zvals[i]
        A[3 + 2*i] = dzvals[i]

    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    # g coefficients in power basis
    degs = [2*k for k in range(26)]
    gcoeff = np.zeros(51)
    for k in range(26):
        d = degs[k]
        gcoeff[:d+1] += c[k] * _lag_coeff_float(d, alpha)

    # denom = x * prod (x - z_i)^2
    denom = np.array([0.0, 1.0])  # x
    for z in zs:
        denom = np.convolve(denom, np.array([1.0, -2*z, z*z]))

    # exact division
    deg_g = len(gcoeff) - 1
    deg_d = len(denom) - 1
    deg_q = deg_g - deg_d
    if deg_q < 0:
        return None
    qcoeff = np.zeros(deg_q + 1)
    rem = gcoeff.copy()
    lead_d = denom[-1]
    for j in range(deg_q, -1, -1):
        factor = rem[deg_g - j] / lead_d
        qcoeff[j] = factor
        rem[deg_g - j:deg_g - j + len(denom)] -= factor * denom
    if np.max(np.abs(rem)) > 1e-7:
        return None

    # real roots of q via companion
    if deg_q == 0:
        B = 0.0
        return (0.0, True, 0, [])
    comp = np.zeros((deg_q, deg_q))
    np.fill_diagonal(comp, 1.0)
    comp[0, :] = -qcoeff[1:deg_q+1] / qcoeff[deg_q]
    try:
        roots = np.linalg.eigvals(comp)
    except Exception:
        return None
    real_roots = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-8 and r.real > 0)
    B = 0.0
    for r in real_roots:
        try:
            qm = _eval_poly(qcoeff, r - 1e-198)
            qp = _eval_poly(qcoeff, r + 1e-198)
        except Exception:
            continue
        if qm * qp < 0:
            if r > B:
                B = r
    return (B, True, deg_q, real_roots)


def _lag_coeff_float(n, a):
    """Coefficients of L_n^(a)(x) = sum_{k=0}^n c_k x^k (float)."""
    c = np.zeros(n + 1)
    for k in range(n + 1):
        c[k] = ((-1.0) ** k) * math.comb(int(n + a + 0.5), n - k) / math.factorial(k)
    return c


def _lag_derivative_float(n, a, x):
    """d/dx L_n^(a)(x) at x (float)."""
    # d/dx L_n^(a) = -L_{n-1}^{(a+1)}
    if n == 0:
        return 0.0
    return -_lag_value_float(n - 1, a + 1, x)


def _lag_value_float(n, a, x):
    """L_n^(a)(x) (float) via recurrence."""
    if n == 0:
        return 1.0
    if n == 1:
        return 1 + a - x
    L0 = 1.0
    L1 = 1 + a - x
    for k in range(1, n):
        L2 = ((2*k + 1 + a - x) * L1 - (k + a) * L0) / (k + 1)
        L0, L1 = L1, L2
    return L1


def _eval_poly(coeff, x):
    """Evaluate polynomial with coeff[0]=constant using Horner."""
    r = 0.0
    for c in coeff[::-1]:
        r = r * x + c
    return r


def _compute_score_exact(z_list):
    """Exact sympy score. Returns B/(2*pi) or None."""
    from sympy import symbols, Rational, Poly, diff, real_roots
    from sympy import Matrix, binomial, factorial

    z_rationals = sorted(Rational(float(z)) for z in z_list)
    x = symbols('x')
    alpha = Rational(-1, 2)

    def lag_gen(n, a, xx):
        s = 0
        for k in range(n + 1):
            s += (-1)**k * binomial(n + a, n - k) * xx**k / factorial(k)
        return s

    degrees = [2*k for k in range(26)]
    c = [symbols(f'c_{i}') for i in range(26)]
    g = sum(c[k] * lag_gen(deg, alpha, x) for k, deg in enumerate(degrees))
    g = g.expand()
    gp = diff(g, x)

    eqs = [g.subs(x, 0), gp.subs(x, 0) - 1]
    for zi in z_rationals:
        eqs.append(g.subs(x, zi))
        eqs.append(gp.subs(x, zi))

    if len(eqs) != 26:
        return None
    A = []
    b = []
    for eq in eqs:
        eq = eq.expand()
        row = []
        rest = eq
        for i in range(26):
            coef = rest.coeff(c[i])
            rest = rest - coef * c[i]
            row.append(coef)
        if rest != 0:
            return None
        A.append(row)
        b.append(0)

    try:
        sol = Matrix(A).LUsolve(Matrix(b))
    except Exception:
        return None
    if any(v is None or v.has(symbols('c')) for v in sol):
        return None

    g_s = sum(sol[k] * lag_gen(deg, alpha, x) for k, deg in enumerate(degrees)).expand()
    denom = x
    for zi in z_rationals:
        denom *= (x - zi)**2
    denom = denom.expand()
    try:
        q_poly, rem = Poly(g_s, x).div(Poly(denom, x))
    except Exception:
        return None
    if rem != 0:
        return None
    q_expr = q_poly.as_expr()
    rr = real_roots(q_expr, x)
    B = 0
    for r in rr:
        rv = float(r)
        if rv <= 0:
            continue
        qm = q_expr.subs(x, r - Rational(1, 10)**198)
        qp = q_expr.subs(x, r + Rational(1, 10)**198)
        if qm * qp < 0 and rv > B:
            B = rv
    return B / (2 * math.pi)


def construct():
    """Return 12 double-root locations z_1..z_12."""
    start = time.time()
    deadline = start + 560.0  # safety margin before 600/630s

    def rem():
        return deadline - time.time()

    # Initial: Bessel-type zeros (k*pi)^2, i=1..12
    z_best = [ (i * math.pi) ** 2 for i in range(1, 13) ]
    score_best = _score_float(z_best)
    if score_best is None:
        z_best = [i**2 * math.pi**2 for i in range(1, 13)]
        score_best = _score_float(z_best)
    B_best = score_best[0] if score_best else float('inf')

    # Coordinate descent + random perturbations
    rng = np.random.default_rng(12345)
    it = 0
    while rem() > 120:
        improved = False
        # coordinate descent
        for i in range(12):
            if rem() < 120:
                break
            base = z_best[:]
            for delta in [0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                z_try = base[:]
                z_try[i] = base[i] * (1 + delta)
                if z_try[i] <= 0:
                    continue
                if len(set(z_try)) != 12:
                    continue
                s = _score_float(z_try)
                if s is not None and s[0] < B_best - 1e-9:
                    z_best = z_try
                    B_best = s[0]
                    improved = True
                    break
        # random perturbation of a couple of coords
        if rem() < 120:
            break
        for _ in range(3):
            if rem() < 120:
                break
            z_try = z_best[:]
            idx = list(rng.choice(12, size=2, replace=False))
            for i in idx:
                z_try[i] = z_best[i] * (1 + rng.uniform(-0.15, 0.15))
            if any(z <= 0 for z in z_try) or len(set(z_try)) != 12:
                continue
            s = _score_float(z_try)
            if s is not None and s[0] < B_best - 1e-9:
                z_best = z_try
                B_best = s[0]
                improved = True
        it += 1
        if not improved and it > 200:
            break

    # Exact verification of best; if it fails, fall back to base
    try:
        exact = _compute_score_exact(z_best)
        if exact is None:
            z_best = [ (i * math.pi) ** 2 for i in range(1, 13) ]
    except Exception:
        z_best = [ (i * math.pi) ** 2 for i in range(1, 13) ]

    # Validate
    res = list(z_best)
    for i in range(12):
        if not math.isfinite(res[i]) or res[i] <= 0:
            res = [ (i * math.pi) ** 2 for i in range(1, 13) ]
            break
    for i in range(12):
        for j in range(i+1, 12):
            if abs(res[i] - res[j]) < 1e-12:
                res = [ (i * math.pi) ** 2 for i in range(1, 13) ]
                break
    return res

# EVOLVE-BLOCK-END
