import numpy as np
import time

N = 32
_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)

def _validity_check(arr, eps=1e-9):
    """Return True if all circles are disjoint (interiors) and inside [0,1]^2."""
    if arr is None or len(arr) != N:
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r <= 0):
        return False
    # inside square
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    # disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - eps:
                return False
    return True

def _score(arr):
    """Sum of radii if valid, else -inf."""
    if _validity_check(arr):
        return float(np.sum(arr[:, 2]))
    return -np.inf

def _hex_seed():
    """Staggered hexagonal-ish lattice trimmed to N circles."""
    pts = []
    # Use a hex packing: rows offset, vertical spacing = r*sqrt(3), horizontal = 2r
    # We'll lay out a grid of candidate centers and pick N, then compute a base radius.
    # Simpler: place on a hex lattice with spacing 1.0, scale later.
    row_h = 0.8660254  # sqrt(3)/2
    cols = 0
    centers = []
    while len(centers) < N + 10:
        y = row_h * cols
        xoff = 0.5 if (cols % 2) else 0.0
        for i in range(0, 20):
            centers.append((xoff + i, y))
            if len(centers) >= N + 10:
                break
        cols += 1
    centers = centers[:N]
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    # normalize into [0,1] with margin
    cx = (cx - cx.min()) / (cx.max() - cx.min() + 1e-12)
    cy = (cy - cy.min()) / (cy.max() - cy.min() + 1e-12)
    # base radius: min distance to boundary /2, and half min pair distance
    r = np.full(N, 0.0)
    for i in range(N):
        dmin = min(cx[i], 1 - cx[i], cy[i], 1 - cy[i])
        r[i] = dmin * 0.5
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            r[i] = min(r[i], d * 0.5)
    return np.column_stack([cx, cy, r])

def _repulsion_optimize(x0, y0, r0, max_iter=4000, dt=0.02, grow=0.0008,
                        wall_k=40.0, pair_k=60.0, max_r=0.25, seed=0):
    """
    Physics-style optimization: repel overlapping circles and grow radii.
    Returns best valid (x,y,r) found during the run.
    """
    rng = np.random.default_rng(seed)
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = np.minimum(r0.copy().astype(float), max_r)

    best = None
    best_score = -np.inf

    # soft objective: minimize overlaps + wall violations, maximize sum r
    def total_penalty():
        pen = 0.0
        # wall
        for i in range(N):
            if x[i] - r[i] < 0: pen += wall_k * (0 - (x[i] - r[i])) ** 2
            if x[i] + r[i] > 1: pen += wall_k * ((x[i] + r[i]) - 1) ** 2
            if y[i] - r[i] < 0: pen += wall_k * (0 - (y[i] - r[i])) ** 2
            if y[i] + r[i] > 1: pen += wall_k * ((y[i] + r[i]) - 1) ** 2
        # pair overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if d < s:
                    pen += pair_k * (s - d) ** 2
        return pen

    start_pen = total_penalty()

    for it in range(max_iter):
        # compute forces
        fx = np.zeros(N); fy = np.zeros(N)
        for i in range(N):
            # wall forces
            if x[i] - r[i] < 0: fx[i] += wall_k * (0 - (x[i] - r[i]))
            if x[i] + r[i] > 1: fx[i] -= wall_k * ((x[i] + r[i]) - 1)
            if y[i] - r[i] < 0: fy[i] += wall_k * (0 - (y[i] - r[i]))
            if y[i] + r[i] > 1: fy[i] -= wall_k * ((y[i] + r[i]) - 1)
            for j in range(i + 1, N):
                dx = x[i] - x[j]; dy = y[i] - y[j]
                d = np.hypot(dx, dy) + 1e-12
                s = r[i] + r[j]
                if d < s:
                    f = pair_k * (s - d) / d
                    fx[i] += f * dx; fy[i] += f * dy
                    fx[j] -= f * dx; fy[j] -= f * dy
        # update positions
        x += dt * fx
        y += dt * fy
        # grow radii
        r = np.minimum(r + grow, max_r)
        # clamp to square (centers)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        if it % 20 == 0 or it == max_iter - 1:
            arr = np.column_stack([x, y, r])
            sc = _score(arr)
            if sc > best_score:
                best_score = sc
                best = arr.copy()

    if best is None:
        best = np.column_stack([x, y, r])
    return best, best_score

def _shrink_to_valid(arr):
    """Given a candidate, shrink radii minimally to guarantee validity."""
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = arr[:, 2].copy()
    # ensure inside
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
    # ensure disjoint: iteratively shrink
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if d < s:
                    # shrink the larger one (or both half)
                    need = s - d
                    # reduce r[i] and r[j] so that r[i]+r[j] <= d
                    # keep ratio
                    tot = r[i] + r[j]
                    if tot <= 0:
                        continue
                    new_tot = max(0.0, d)
                    r[i] = r[i] * new_tot / tot
                    r[j] = r[j] * new_tot / tot
                    changed = True
        if not changed:
            break
    return np.column_stack([x, y, r])

def construct():
    t0 = time.time()
    global _DEADLINE

    # baseline from hex seed
    base = _hex_seed()
    base = _shrink_to_valid(base)
    best = base
    best_score = _score(best)

    rng = np.random.default_rng(12345)

    restart = 0
    while time.time() < _DEADLINE - 5:
        # build a random-ish initial configuration
        # jitter the hex seed
        x0 = base[:, 0].copy()
        y0 = base[:, 1].copy()
        r0 = base[:, 2].copy()
        # random jitter
        jit = 0.05
        x0 += rng.uniform(-jit, jit, N)
        y0 += rng.uniform(-jit, jit, N)
        x0 = np.clip(x0, 0.0, 1.0)
        y0 = np.clip(y0, 0.0, 1.0)
        # random initial radius scale
        scale = rng.uniform(0.7, 1.3)
        r0 = r0 * scale

        # randomize order / positions for diversity occasionally
        if restart % 5 == 0:
            perm = rng.permutation(N)
            x0 = x0[perm]; y0 = y0[perm]; r0 = r0[perm]

        arr, sc = _repulsion_optimize(
            x0, y0, r0,
            max_iter=2500,
            dt=0.02,
            grow=0.0008,
            wall_k=40.0,
            pair_k=60.0,
            max_r=0.25,
            seed=restart
        )
        # shrink to valid
        arr = _shrink_to_valid(arr)
        sc = _score(arr)
        if sc > best_score:
            best_score = sc
            best = arr.copy()
        restart += 1

    # final safety: ensure validity
    best = _shrink_to_valid(best)
    if not _validity_check(best):
        # fall back to a guaranteed-valid trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])

    return best
