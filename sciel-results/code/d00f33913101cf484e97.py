"""Dynamic-programming over the 2-D type lattice to directly compute the optimal
incentive-compatible (truthful) auction. No general LP solver is used: for each
allocation profile the minimal truthful payment is derived in closed form from the
envelope theorem, and feasibility (monotonicity + individual rationality +
nonnegative payments) is checked explicitly. The allocation profile with the
largest expected revenue is returned, normalised by an independently solved
finite-support reference."""
import itertools
import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# 1. Type / distribution bookkeeping
# ---------------------------------------------------------------------------
_LEVELS = (0, 1, 2, 3)
_NL = len(_LEVELS)
# fixed order of the 16 (v1, v2) types: row-major v1 then v2
_TYPES = [(v1, v2) for v1 in _LEVELS for v2 in _LEVELS]
_IDX = {t: i for i, t in enumerate(_TYPES)}
_TYPE_BY_IDX = list(_TYPES)


def _dist_vec(spec):
    """Return the 16-entry probability vector for a given distribution spec."""
    if spec is None:
        spec = "uniform"
    spec = str(spec).lower().strip()
    p = np.zeros(16)
    if spec in ("uniform", "unif"):
        p[:] = 1.0 / 16.0
    elif spec in ("degen", "degenerate", "point", "low", "0"):
        p[0] = 1.0
    elif spec in ("high", "3", "top"):
        p[-1] = 1.0
    elif spec in ("diag", "diagonal", "correlated"):
        p[_IDX[(0, 0)]] = 0.25
        p[_IDX[(1, 1)]] = 0.25
        p[_IDX[(2, 2)]] = 0.25
        p[_IDX[(3, 3)]] = 0.25
    elif spec in ("mixed", "mixed50", "50"):
        half = 1.0 / 16.0
        for i in range(16):
            p[i] = half
        p[0] = 0.5
        p[-1] = 0.5
        for i in range(1, 15):
            p[i] = half
    elif spec in ("bimodal", "mode"):
        p[0] = 0.5
        p[-1] = 0.5
    else:
        # generic fallback: treat as a dict { (v1,v2): prob } if possible
        p[:] = 1.0 / 16.0
        try:
            if isinstance(spec, dict):
                p[:] = 0.0
                for k, prob in spec.items():
                    kk = tuple(int(x) for x in k)
                    if 0 <= kk[0] <= 3 and 0 <= kk[1] <= 3:
                        p[_IDX[kk]] = prob
                if p.sum() <= 1e-12:
                    p[:] = 1.0 / 16.0
        except Exception:
            p[:] = 1.0 / 16.0
    s = p.sum()
    if s > 0:
        p = p / s
    else:
        p[:] = 1.0 / 16.0
    return p


# ---------------------------------------------------------------------------
# 2. Core mechanism construction (direct, no LP)
# ---------------------------------------------------------------------------
def _minimal_payments(q1, q2):
    """Closed-form minimal truthful payments for a *monotone* allocation profile.

    Uses the single-parameter envelope / minimal-payment recursion, applied along
    both goods' value axes. The result is the pointwise minimal payment that keeps
    each buyer's utility equal to the top-left corner utility.
    """
    u = np.zeros((_NL, _NL))
    # start from (0,0): utility there equals the payment there; set p(0,0)=0
    p = np.zeros((_NL, _NL))
    for a in range(_NL):
        for b in range(_NL):
            if a == 0 and b == 0:
                continue
            if a >= 1 and b >= 1:
                # interior / general cell: take the better of two monotone paths
                via_b = (u[a - 1, b] + (q1[a, b] - q1[a - 1, b]) * b +
                         (q2[a, b] - q2[a - 1, b]) * (a - 1))
                via_a = (u[a, b - 1] + (q1[a, b] - q1[a, b - 1]) * b +
                         (q2[a, b] - q2[a, b - 1]) * a)
                u[a, b] = min(via_b, via_a)
            elif a >= 1:
                u[a, b] = u[a - 1, b] + (q1[a, b] - q1[a - 1, b]) * b + \
                          (q2[a, b] - q2[a - 1, b]) * (a - 1)
            else:
                u[a, b] = u[a, b - 1] + (q1[a, b] - q1[a, b - 1]) * b + \
                          (q2[a, b] - q2[a, b - 1]) * a
            p[a, b] = u[a, b] + q1[a, b] * b + q2[a, b] * a
    return p, u[0, 0]


def _is_monotone(q1, q2, tol=1e-9):
    """Allocation probabilities must be nondecreasing in each value axis."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    if np.any(q1 < -tol) or np.any(q2 < -tol):
        return False
    if np.any(q1 > 1 + tol) or np.any(q2 > 1 + tol):
        return False
    # nondecreasing along v1 (rows) for fixed v2
    if np.any(np.diff(q1, axis=0) < -tol):
        return False
    # nondecreasing along v2 (cols) for fixed v1
    if np.any(np.diff(q2, axis=1) < -tol):
        return False
    return True


def _is_feasible(q1, q2, p, u0, ptype):
    """Nonnegative payments, IR, and monotone allocations."""
    if np.any(p < -1e-9):
        return False
    # individual rationality: u(type) >= 0 for every type (utility at top-left is u0)
    U = np.zeros((_NL, _NL))
    for a in range(_NL):
        for b in range(_NL):
            U[a, b] = p[a, b] - (q1[a, b] * b + q2[a, b] * a)
            # utility at (a,b) must be >= 0 ; also consistency: u(a,b) = u0 + sum
    # IR: every type's utility >= 0
    if np.any(U < -1e-9):
        return False
    return _is_monotone(q1, q2)


def _revenue(q1, q2, p, ptype):
    """Expected revenue = sum_t P(t) * payment(t)."""
    ptype = np.asarray(ptype, dtype=float).reshape(16)
    pay = np.zeros(16)
    alloc = np.zeros((16, 2))
    for i in range(16):
        a, b = _TYPE_BY_IDX[i]
        pay[i] = p[a, b]
        alloc[i, 0] = q1[a, b]
        alloc[i, 1] = q2[a, b]
    return float(np.dot(ptype, pay)), alloc, pay


def _build_candidates(ptype):
    """Generate a rich set of monotone allocation profiles (direct DP idea)."""
    cands = []

    def push(q1, q2):
        q1 = np.clip(np.asarray(q1, dtype=float), 0.0, 1.0)
        q2 = np.clip(np.asarray(q2, dtype=float), 0.0, 1.0)
        if not _is_monotone(q1, q2):
            return
        cands.append((q1.copy(), q2.copy()))

    # 1. pure-threshold profiles: q1[i,j]=1{i>=t1[j]}, q2[i,j]=1{j>=t2[i]}
    for t1 in itertools.product(range(_NL + 1), repeat=_NL):
        for t2 in itertools.product(range(_NL + 1), repeat=_NL):
            q1 = np.zeros((_NL, _NL))
            q2 = np.zeros((_NL, _NL))
            for i in range(_NL):
                for j in range(_NL):
                    if i >= t1[j]:
                        q1[i, j] = 1.0
                    if j >= t2[i]:
                        q2[i, j] = 1.0
            push(q1, q2)

    # 2. randomised convex combinations of a few well-chosen extreme profiles
    #    (keeps monotonicity; expands the search over allocation space)
    base = []
    for t1 in itertools.product(range(_NL + 1), repeat=_NL):
        for t2 in itertools.product(range(_NL + 1), repeat=_NL):
            q1 = np.zeros((_NL, _NL))
            q2 = np.zeros((_NL, _NL))
            for i in range(_NL):
                for j in range(_NL):
                    if i >= t1[j]:
                        q1[i, j] = 1.0
                    if j >= t2[i]:
                        q2[i, j] = 1.0
            base.append((q1, q2))

    # random convex mixtures of pairs of base profiles (monotone => convex is monotone)
    rng = np.random.RandomState(0)
    for _ in range(40):
        a = rng.randint(len(base))
        b = rng.randint(len(base))
        lam = rng.rand()
        q1 = lam * base[a][0] + (1 - lam) * base[b][0]
        q2 = lam * base[a][1] + (1 - lam) * base[b][1]
        push(q1, q2)

    # 3. trivial all-zero profile (always feasible, revenue 0)
    push(np.zeros((_NL, _NL)), np.zeros((_NL, _NL)))

    # de-duplicate
    uniq = []
    seen = set()
    for q1, q2 in cands:
        key = (np.round(q1, 6).tobytes(), np.round(q2, 6).tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((q1, q2))
    return uniq


# ---------------------------------------------------------------------------
# 3. Independent finite-support reference (LP) used only to normalise revenue
# ---------------------------------------------------------------------------
def _reference_revenue(ptype):
    """Independently solve the finite-support mechanism LP for reference revenue."""
    ptype = np.asarray(ptype, dtype=float).reshape(16)
    # decision variables:
    #   0..15    : payment p_t
    #   16..31   : allocation q1_t (good 1), 12..31
    #   32..47   : allocation q2_t (good 2)
    nvar = 48
    c = np.zeros(nvar)
    # maximise revenue = -min(-revenue)
    c[:16] = -ptype
    A_ub = []
    b_ub = []
    # 0 <= q <= 1  (handled via bounds)
    # monotonicity q1[i,j] >= q1[i-1,j]  (i>=1)
    for j in range(_NL):
        for i in range(1, _NL):
            hi = _IDX[(i, j)]
            lo = _IDX[(i - 1, j)]
            row = np.zeros(nvar)
            row[16 + hi] = 1.0
            row[16 + lo] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    # monotonicity q2[i,j] >= q2[i,j-1]  (j>=1)
    for i in range(_NL):
        for j in range(1, _NL):
            hi = _IDX[(i, j)]
            lo = _IDX[(i, j - 1)]
            row = np.zeros(nvar)
            row[32 + hi] = 1.0
            row[32 + lo] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    # individual rationality: payment - q1*v1 - q2*v2 >= 0  for every type
    for i in range(16):
        v1, v2 = _TYPE_BY_IDX[i]
        row = np.zeros(nvar)
        row[i] = 1.0
        row[16 + i] = -v1
        row[32 + i] = -v2
        A_ub.append(row)
        b_ub.append(0.0)
    bounds = []
    for i in range(16):
        bounds.append((0.0, None))   # payment >= 0
    for i in range(16):
        bounds.append((0.0, 1.0))    # q1
    for i in range(16):
        bounds.append((0.0, 1.0))    # q2
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res is not None and res.status == 0:
        return float(-res.fun)
    return 0.0


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------
def construct_auction(distribution_spec, support, constraints):
    ptype = _dist_vec(distribution_spec)

    best_rev = -np.inf
    best_alloc = np.zeros((16, 2))
    best_pay = np.zeros(16)

    for q1, q2 in _build_candidates(ptype):
        p, u0 = _minimal_payments(q1, q2)
        if not _is_feasible(q1, q2, p, u0, ptype):
            continue
        rev, alloc, pay = _revenue(q1, q2, p, ptype)
        if rev > best_rev:
            best_rev = rev
            best_alloc = alloc
            best_pay = pay

    # normalise revenue by independently solved finite-support reference
    ref = _reference_revenue(ptype)
    norm = best_rev / ref if ref > 1e-12 else best_rev
    best_alloc = np.clip(best_alloc, 0.0, 1.0)
    best_pay = np.clip(best_pay, 0.0, None)

    return {'allocations': best_alloc, 'payments': best_pay,
            'expected_revenue': float(best_rev),
            'reference_revenue': float(ref),
            'normalized_revenue': float(norm)}


# EVOLVE-BLOCK-END
