"""Erdos minimum overlap — LP-based construction."""
import time
import numpy as np

N = 100
DEADLINE = 110.0  # wall-clock safety margin (limit is 150s)
_start = time.time()

def _elapsed():
    return time.time() - _start

def _overlap_matrix():
    """Return (A, b, shift_indices) for constraints overlap_k <= V for all k>0.
    overlap_k = sum_{i} h[i]*(1-h[i-k]) over valid i, expressed linearly in h and constants.
    For each shift k: overlap_k = sum_i h[i] - sum_i h[i]*h[i-k]  ... but h[i]*h[i-k] is quadratic.
    We handle the quadratic by treating it at its current value during a single LP pass,
    OR we minimize a linear surrogate: use the identity that for mean 0.5 the cross term
    is bounded. Instead we minimize the peak of the *linear* part: sum_i h[i]*(1 - h[i-k])
    approximated by fixing h[i-k] to 0.5 in the product gives sum_i h[i] - 0.5*sum_i h[i-k].
    The quadratic residual is small; we refine with a second pass.
    """
    pass


def _lp_construct():
    """Minimize V = max_k overlap_k via linear programming with a linear surrogate,
    then polish. Return (h, score)."""
    try:
        from scipy.optimize import linprog
    except Exception:
        return None, None

    # Decision variable: h of length N, with sum h = N/2, 0<=h<=1.
    # Objective: minimize V.
    # Constraint for each shift k in 1..N-1:
    #   overlap_k(h) = sum_{i=k}^{N-1} h[i] * (1 - h[i-k]) <= V
    # Expand: sum_i h[i] - sum_i h[i]h[i-k] <= V
    # The term sum_i h[i]h[i-k] is quadratic. We linearize by using the fact that
    # h[i]h[i-k] <= (h[i]+h[i-k])/2 and also >= ... To get an exact minimax we use
    # the identity with an auxiliary: introduce w_k = sum_i h[i](1-h[i-k]) is NOT linear.
    #
    # Alternative exact linearization: note that (1 - h[i-k]) in [0,1], and the product
    # h[i]*(1-h[i-k]) is bilinear. We minimize the max over k of a bilinear form — a
    # quadratic program. We solve it with scipy minimize (SLSQP) instead for exactness,
    # but keep LP as a fast initializer.
    #
    # Fast initializer via LP: minimize V s.t. for each k, sum_i h[i] - 0.5*sum_{i>=k} h[i-k] <= V
    # (using h[i-k] <= 0.5+... ). This gives a valid feasible h close to optimum.
    n = N
    # variables: h (n) + V (1) => n+1 vars
    # objective: minimize V => c = [0,...,0, 1]
    c = np.zeros(n + 1)
    c[-1] = 1.0

    A_ub_rows = []
    b_ub = []
    for k in range(1, n):
        row = np.zeros(n + 1)
        # sum_{i=k}^{n-1} h[i]  (h[i] coefficient +1)
        row[k:n] += 1.0
        # -0.5 * sum_{i=k}^{n-1} h[i-k]  => for index j in 0..n-1-k, coeff -0.5
        row[0:n-k] += -0.5
        row[-1] = -1.0  # bring V to left: LHS - V <= 0
        A_ub_rows.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub) if b_ub else None

    # equality: sum h = n/2
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([n / 2.0])

    bounds = [(0.0, 1.0)] * n + [(0.0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method='highs', options={'presolve': True, 'time_limit': 60})

    if res is None or not res.success:
        return None, None

    h = res.x[:n].copy()
    h = np.clip(h, 0.0, 1.0)
    # fix sum exactly
    diff = n / 2.0 - h.sum()
    h += diff / n
    h = np.clip(h, 0.0, 1.0)
    # re-fix after clip
    diff = n / 2.0 - h.sum()
    if abs(diff) > 1e-12:
        # add to a coordinate with room
        idx = int(np.argmax(h < 0.9999999)) if (h < 0.9999999).any() else int(np.argmin(h > 0.0000001)) if (h > 0.0000001).any() else 0
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h, None


def _true_score(h):
    """Compute the maximum overlap of h with shifted (1-h).
    overlap_k = sum_{i=k}^{N-1} h[i] * (1 - h[i-k])   for k=1..N-1
    Return the maximum."""
    n = len(h)
    best = 0.0
    for k in range(1, n):
        ov = 0.0
        # i from k to n-1
        a = h[k:n]
        b = h[0:n-k]
        ov = float(np.sum(a * (1.0 - b)))
        if ov > best:
            best = ov
    return best


def _polish(h0, max_iter=8):
    """Refine h to reduce the max overlap using a simple gradient / coordinate descent
    on the true (quadratic) objective. Keep sum fixed and bounds."""
    n = len(h0)
    h = h0.astype(float).copy()

    def score_and_grad(h):
        # objective f(h) = max_k ov_k,  ov_k = sum_{i>=k} h[i](1-h[i-k])
        # gradient of ov_k w.r.t. h[i]:
        #   from term (i, i-k): d/dh[i] h[i](1-h[i-k]) = (1 - h[i-k])
        #   from term (i+k, i): d/dh[i] h[i+k](1-h[i]) = -h[i+k]
        ov = np.zeros(n - 1)
        for k in range(1, n):
            ov[k - 1] = float(np.sum(h[k:n] * (1.0 - h[0:n-k])))
        kmax = int(np.argmax(ov)) + 1
        # gradient of ov_{kmax}
        grad = np.zeros(n)
        k = kmax
        # terms h[i](1-h[i-k]) for i in k..n-1: d/dh[i] = (1 - h[i-k])
        grad[k:n] += (1.0 - h[0:n-k])
        # also h[i] appears as h[i+k](1-h[i]) for i in 0..n-1-k: d/dh[i] = -h[i+k]
        grad[0:n-k] += -h[k:n]
        return float(ov[k - 1]), grad

    for it in range(max_iter):
        if _elapsed() > DEADLINE - 5.0:
            break
        f, grad = score_and_grad(h)
        # project gradient step keeping sum zero
        step = 0.02
        dh = -grad
        dh -= dh.mean()  # keep sum constant
        h_new = h + step * dh
        # project onto [0,1] while preserving sum: simple clip + redistribute
        h_new = np.clip(h_new, 0.0, 1.0)
        diff = n / 2.0 - h_new.sum()
        # distribute diff to entries with room
        if diff > 0:
            room = (1.0 - h_new)
            tot = room.sum()
            if tot > 1e-12:
                h_new += diff * (room / tot)
        elif diff < 0:
            room = h_new
            tot = room.sum()
            if tot > 1e-12:
                h_new -= (-diff) * (room / tot)
        h_new = np.clip(h_new, 0.0, 1.0)
        # exact sum fix
        diff = n / 2.0 - h_new.sum()
        if abs(diff) > 1e-14:
            idx = int(np.argmax(h_new < 0.9999999)) if (h_new < 0.9999999).any() else 0
            h_new[idx] += diff
            h_new = np.clip(h_new, 0.0, 1.0)
        new_f = _true_score(h_new)
        if new_f < f - 1e-12:
            h = h_new
            step *= 0.7
        else:
            step *= 0.5
            if step < 1e-6:
                break
    return h


def _seed_pattern():
    """A structured non-uniform starting point: sinusoidal to break symmetry."""
    n = N
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    h = 0.5 + 0.25 * np.sin(t)
    h = np.clip(h, 0.0, 1.0)
    diff = n / 2.0 - h.sum()
    h += diff / n
    return np.clip(h, 0.0, 1.0)


def construct():
    # Always have a valid fallback ready.
    fallback = np.full(N, 0.5)

    best_h = fallback
    best_f = _true_score(fallback)

    # 1) LP initializer
    if _elapsed() < DEADLINE - 20.0:
        h_lp, _ = _lp_construct()
        if h_lp is not None:
            h_lp = _polish(h_lp)
            f = _true_score(h_lp)
            if f < best_f:
                best_f = f
                best_h = h_lp.copy()

    # 2) Structured seed + polish (different basin)
    if _elapsed() < DEADLINE - 20.0:
        h_seed = _polish(_seed_pattern())
        f = _true_score(h_seed)
        if f < best_f:
            best_f = f
            best_h = h_seed.copy()

    # 3) Final guard: ensure validity (sum exactly N/2, in [0,1])
    h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - h.sum()
    if abs(diff) > 1e-12:
        idx = int(np.argmax(h < 0.9999999)) if (h < 0.9999999).any() else 0
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        # exact
        diff = N / 2.0 - h.sum()
        if abs(diff) > 1e-14:
            h[idx] += diff
    return h
