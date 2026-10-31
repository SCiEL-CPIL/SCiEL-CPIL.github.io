"""Combined program: sequential greedy construction + local search + SLSQP polish,
with a free (searchable) rectangle of perimeter 4."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # hard safety margin before the 150s kill
    best = None
    best_score = -np.inf

    # ---------- low-level packing helpers (fixed w,h) ----------
    def min_pair_dist(x, y):
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1)

    def max_radius(x, y, r, w, h):
        cap = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        cap = np.minimum(cap, min_pair_dist(x, y) - r)
        return np.clip(cap, 0.0, None)

    def feasible(x, y, r, w, h, tol=1e-8):
        if r.size == 0:
            return True
        if np.any(r < -tol):
            return False
        if np.min(x - r) < -tol or np.max(x + r) > w + tol:
            return False
        if np.min(y - r) < -tol or np.max(y + r) > h + tol:
            return False
        if r.size < 2:
            return True
        D = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        S = r[:, None] + r[None, :]
        if np.max(S - D) > tol:
            return False
        return True

    def rect_of(x, y, r):
        w = np.max(x + r) if x.size else 0.0
        h = np.max(y + r) if y.size else 0.0
        return w, h

    # ---------- strategy 1: sequential greedy construction ----------
    def greedy_place(w, h, order, seed):
        rng = np.random.default_rng(seed)
        x = np.empty(N); y = np.empty(N); r = np.empty(N)
        for idx, ci in enumerate(order):
            if idx == 0:
                x[ci] = w / 2.0; y[ci] = h / 2.0; r[ci] = min(w, h) / 2.0
            else:
                cand = rng.uniform(0.0, w, 1500)
                cand2 = rng.uniform(0.0, h, 1500)
                X = np.repeat(cand, 1500); Y = np.tile(cand2, 1500)
                cap = max_radius(X, Y, r[:idx], w, h)
                best_i = int(np.argmax(cap))
                if cap[best_i] <= 1e-9:
                    cap = max_radius(X, Y, r[:idx], w, h)
                    best_i = int(np.argmax(cap))
                x[ci] = X[best_i]; y[ci] = Y[best_i]; r[ci] = cap[best_i]
        return x, y, r

    # ---------- strategy 2: local search (grow, jitter, swap) ----------
    def local_search(x, y, r, w, h, iters, seed):
        rng = np.random.default_rng(seed)
        x = x.copy(); y = y.copy(); r = r.copy()
        for it in range(iters):
            if time.time() > deadline - 8:
                break
            op = it % 3
            if op == 0:
                r = np.maximum(r, max_radius(x, y, r, w, h))
            elif op == 1:
                i = int(rng.integers(N))
                x[i] = np.clip(x[i] + rng.normal(0, 0.04 * w), 0.0, w)
                y[i] = np.clip(y[i] + rng.normal(0, 0.04 * h), 0.0, h)
                r[i] = max_radius(x, y, r, w, h)[i]
            else:
                a, b = rng.integers(N, 2)
                x[a], x[b] = x[b], x[a]
                y[a], y[b] = y[b], y[a]
                r[a], r[b] = r[b], r[a]
                r = np.maximum(r, max_radius(x, y, r, w, h))
        r = np.maximum(r, max_radius(x, y, r, w, h))
        return x, y, r

    # ---------- strategy 3: SLSQP polish with a free rectangle ----------
    def polish(x, y, r, w, h, maxiter=600, ftol=1e-9):
        w0 = np.clip(w, 0.05, 1.95)
        h0 = np.clip(h, 0.05, 1.95)
        v0 = np.concatenate([x, y, r, [w0, h0]])
        P = 1000.0

        def obj(v):
            xi = v[0::3]; yi = v[1::3]; ri = v[2::3]
            wi = v[-2]; hi = v[-1]
            pen = 0.0
            pen += P * np.sum(np.maximum(0.0, ri - xi) ** 2)
            pen += P * np.sum(np.maximum(0.0, ri - (wi - xi)) ** 2)
            pen += P * np.sum(np.maximum(0.0, ri - yi) ** 2)
            pen += P * np.sum(np.maximum(0.0, ri - (hi - yi)) ** 2)
            pen += P * np.sum(np.maximum(0.0, -ri) ** 2)
            for i in range(N):
                for j in range(i + 1, N):
                    dd = np.sqrt((xi[i] - xi[j]) ** 2 + (yi[i] - yi[j]) ** 2)
                    ov = ri[i] + ri[j] - dd
                    if ov > 0:
                        pen += P * ov * ov
            pen += 10.0 * (2.0 - (wi + hi)) ** 2
            return pen - np.sum(ri)

        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.95))
            bounds.append((0.0, 1.95))
            bounds.append((0.0, 0.975))
        bounds.append((0.05, 1.95))
        bounds.append((0.05, 1.95))
        try:
            res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})
        except Exception:
            return None
        xi = res.x[0::3]; yi = res.x[1::3]; ri = res.x[2::3]
        wi = res.x[-2]; hi = res.x[-1]
        xi = np.clip(xi, 0.0, wi); yi = np.clip(yi, 0.0, hi); ri = np.clip(ri, 0.0, None)
        if not feasible(xi, yi, ri, wi, hi, tol=1e-6):
            return None
        return xi, yi, ri, wi, hi

    # ---------- driver ----------
    aspect_ratios = [1.0, 1.2, 1.3, 1.5, 1.25, 1.1, 1.4, 1.6, 0.9, 1.7, 1.35, 1.15, 2.0, 0.8, 1.45]
    max_iters = 0
    for ar in aspect_ratios:
        if time.time() > deadline - 25:
            break
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        for seed in range(4):
            if time.time() > deadline - 20:
                break
            order = list(range(N))
            np.random.default_rng(seed).shuffle(order)
            x, y, r = greedy_place(w, h, order, seed)
            x, y, r = local_search(x, y, r, w, h, 150, seed)
            score = float(np.sum(r))
            if feasible(x, y, r, w, h, tol=1e-7) and score > best_score:
                best_score = score
                best = (x.copy(), y.copy(), r.copy(), w, h)
            if time.time() > deadline - 12:
                break
            pol = polish(x, y, r, w, h, maxiter=500, ftol=1e-9)
            if pol is not None:
                xi, yi, ri, wi, hi = pol
                s = float(np.sum(ri))
                if s > best_score:
                    best_score = s
                    best = (xi, yi, ri, wi, hi)
        max_iters += 1

    # ---------- final high-precision polish on the best packing ----------
    if best is not None:
        x, y, r, w, h = best
        for maxiter, ftol in [(1500, 1e-11), (2500, 1e-12)]:
            if time.time() > deadline - 5:
                break
            pol = polish(x, y, r, w, h, maxiter=maxiter, ftol=ftol)
            if pol is not None:
                xi, yi, ri, wi, hi = pol
                s = float(np.sum(ri))
                if s > best_score:
                    best_score = s
                    best = (xi, yi, ri, wi, hi)

    # ---------- fallback (should not be needed) ----------
    if best is None:
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    x, y, r, w, h = best
    # final clip for safety (keeps perimeter <= 4 and validity)
    x = np.clip(x, 0.0, w); y = np.clip(y, 0.0, h); r = np.clip(r, 0.0, None)
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
