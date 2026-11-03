import numpy as np
from scipy.optimize import minimize, linprog
import time
import random

N = 21
IDX_I, IDX_J = np.triu_indices(N, k=1)

def construct():
    start_time = time.time()
    deadline = start_time + 143.0

    def check_time():
        return time.time() > deadline

    def get_rect_dims(t):
        t = float(np.clip(t, 0.12, 0.88))
        return 2.0 * t, 2.0 * (1.0 - t)

    def unpack(state):
        t = state[0]
        w, h = get_rect_dims(t)
        x = state[1:1 + N]
        y = state[1 + N:1 + 2 * N]
        r = state[1 + 2 * N:1 + 3 * N]
        return w, h, x, y, r

    def is_valid(w, h, x, y, r, eps=1e-6):
        if np.any(r < 1e-6) or w <= 0 or h <= 0:
            return False
        if np.any(x - r < -eps) or np.any(x + r > w + eps) or \
           np.any(y - r < -eps) or np.any(y + r > h + eps):
            return False
        dx = x[IDX_I] - x[IDX_J]
        dy = y[IDX_I] - y[IDX_J]
        d2 = dx * dx + dy * dy
        md = r[IDX_I] + r[IDX_J]
        if np.any(d2 < (md - eps) ** 2):
            return False
        return True

    def sum_r(w, h, x, y, r):
        return float(np.sum(r))

    # ---------- initial configurations ----------
    def make_hex(t, spacing=1.0, jitter=0.0):
        w, h = get_rect_dims(t)
        # Estimate radius based on area and hex density (~0.9069)
        # Area = w*h. N * pi * r^2 * (1/0.9069) ~ w*h -> r ~ sqrt(w*h / (N*pi*1.1))
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        r = 0.95 * r_est # Start slightly smaller to allow expansion
        
        pos = []
        row = 0
        y = r
        # Hex packing in a rectangle
        # Row height is sqrt(3)*r
        # Offset is r
        while y <= h - r + 1e-9 and len(pos) < N:
            off = (r if row % 2 else 0.0)
            x = r + off
            while x <= w - r + 1e-9 and len(pos) < N:
                pos.append((x, y))
                x += 2.0 * r
            y += np.sqrt(3.0) * r
            row += 1
        
        # If we didn't get enough circles, fill remaining with a grid or random
        if len(pos) < N:
            # Simple fallback: place remaining in a grid of the remaining space or just spread them out
            # For robustness, just add them in a grid pattern in the center
            side = int(np.ceil(np.sqrt(N - len(pos))))
            for i in range(N - len(pos)):
                # This is a rough fallback, usually hex works for 21
                pos.append((w/2, h/2)) 

        # Ensure we have exactly N positions
        while len(pos) < N:
            pos.append((w/2, h/2))
            
        state = np.zeros(1 + 3 * N)
        state[0] = t
        for i in range(N):
            px, py = pos[i]
            if jitter > 0:
                px += random.uniform(-jitter, jitter)
                py += random.uniform(-jitter, jitter)
            state[1 + i] = np.clip(px, r, w - r)
            state[1 + N + i] = np.clip(py, r, h - r)
            state[1 + 2 * N + i] = r
        return state

    def make_grid(t, rows, cols, jitter=0.0):
        w, h = get_rect_dims(t)
        state = np.zeros(1 + 3 * N)
        state[0] = t
        for i in range(N):
            rr = i // cols
            cc = i % cols
            px = (cc + 0.5) / cols * w
            py = (rr + 0.5) / rows * h
            if jitter > 0:
                px += random.uniform(-jitter, jitter)
                py += random.uniform(-jitter, jitter)
            state[1 + i] = np.clip(px, 0.02, w - 0.02)
            state[1 + N + i] = np.clip(py, 0.02, h - 0.02)
            state[1 + 2 * N + i] = 0.5 * min(w / cols, h / rows) * 0.9
        return state

    def make_random(t):
        w, h = get_rect_dims(t)
        state = np.zeros(1 + 3 * N)
        state[0] = t
        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        placed = []
        for i in range(N):
            ok = False
            for _ in range(400):
                r = r_est * random.uniform(0.7, 1.3)
                x = random.uniform(r, w - r)
                y = random.uniform(r, h - r)
                good = True
                for (px, py, pr) in placed:
                    if (x - px) ** 2 + (y - py) ** 2 < (r + pr) ** 2:
                        good = False
                        break
                if good:
                    placed.append((x, y, r))
                    state[1 + i] = x
                    state[1 + N + i] = y
                    state[1 + 2 * N + i] = r
                    ok = True
                    break
            if not ok:
                state[1 + i] = w / 2
                state[1 + N + i] = h / 2
                state[1 + 2 * N + i] = 0.02
        return state

    # ---------- physics relaxation (fast pre-relaxer) ----------
    def physics(w, h, x, y, r, max_steps=80, lr=0.02):
        if check_time():
            return x, y, r
        x = x.copy(); y = y.copy(); r = r.copy()
        for _ in range(max_steps):
            if check_time():
                break
            fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]; dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    md = r[i] + r[j]
                    if d2 < md * md:
                        d = np.sqrt(max(d2, 1e-12))
                        ov = md - d
                        f = 8.0 * ov
                        ux = dx / d; uy = dy / d
                        fx[i] += ux * f; fy[i] += uy * f
                        fx[j] -= ux * f; fy[j] -= uy * f
                        fr[i] += 0.3 * ov; fr[j] += 0.3 * ov
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += 8.0 * (r[i] - x[i]); fr[i] -= 0.5 * (r[i] - x[i])
                if x[i] > w - r[i]:
                    fx[i] -= 8.0 * (x[i] - (w - r[i])); fr[i] -= 0.5 * (x[i] - (w - r[i]))
                if y[i] < r[i]:
                    fy[i] += 8.0 * (r[i] - y[i]); fr[i] -= 0.5 * (r[i] - y[i])
                if y[i] > h - r[i]:
                    fy[i] -= 8.0 * (y[i] - (h - y[i])); fr[i] -= 0.5 * (y[i] - (h - y[i]))
            x = np.clip(x + lr * fx, r, w - r)
            y = np.clip(y + lr * fy, r, h - r)
            r = np.clip(r + lr * fr, 1e-5, None)
            for i in range(N):
                r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
        return x, y, r

    # ---------- softplus L-BFGS over full state ----------
    def softplus(z, beta):
        return np.where(z > 1.0 / beta, z, np.log1p(np.exp(beta * z)) / beta)

    def d_softplus(z, beta):
        zz = beta * z
        sig = np.where(zz > 0, 1.0 / (1.0 + np.exp(-zz)), np.exp(zz) / (1.0 + np.exp(zz)))
        return sig * beta

    def refine_smooth(state, is_polish=False):
        if check_time():
            return state
        w, h, x, y, r = unpack(state)
        x, y, r = physics(w, h, x, y, r, max_steps=60)
        v0 = np.zeros(1 + 3 * N)
        v0[0] = state[0]
        v0[1:1 + N] = x
        v0[1 + N:1 + 2 * N] = y
        v0[1 + 2 * N:] = r

        beta = 120.0 if is_polish else 60.0
        pw = 60.0 if is_polish else 25.0
        # Weight for equal radius penalty
        # Higher weight forces radii to be closer
        # The objective is -sum(r) + pw*penalty + pw_eq * sum((r_i - r_mean)^2)
        # Gradient of sum((r_i - r_mean)^2) wrt r_i is 2(r_i - r_mean)
        pw_eq = 10.0 if is_polish else 5.0 

        def obj(v):
            t = v[0]; wv, hv = get_rect_dims(t)
            xv = v[1:1 + N]; yv = v[1 + N:1 + 2 * N]; rv = v[1 + 2 * N:]
            if np.any(rv < 1e-5):
                return 1e6
            pen = 0.0
            pen += np.sum(softplus(rv - xv, beta))
            pen += np.sum(softplus(xv + rv - wv, beta))
            pen += np.sum(softplus(rv - yv, beta))
            pen += np.sum(softplus(yv + rv - hv, beta))
            dx = xv[IDX_I] - xv[IDX_J]
            dy = yv[IDX_I] - yv[IDX_J]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            pen += np.sum(softplus(rv[IDX_I] + rv[IDX_J] - dist, beta))
            
            # Equal radius penalty
            r_mean = np.mean(rv)
            eq_pen = np.sum((rv - r_mean) ** 2)
            
            return -np.sum(rv) + pw * pen + pw_eq * eq_pen

        def grad(v):
            t = v[0]; wv, hv = get_rect_dims(t)
            xv = v[1:1 + N]; yv = v[1 + N:1 + 2 * N]; rv = v[1 + 2 * N:]
            g = np.zeros_like(v)
            g[1 + 2 * N:] = -1.0
            vl = rv - xv; vr = xv + rv - wv; vb = rv - yv; vt = yv + rv - hv
            dsl = d_softplus(vl, beta); dsr = d_softplus(vr, beta)
            dsb = d_softplus(vb, beta); dst = d_softplus(vt, beta)
            g[1:1 + N] += -pw * dsl + pw * dsr
            g[1 + N:1 + 2 * N] += -pw * dsb + pw * dst
            g[1 + 2 * N:] += pw * (dsl + dsr + dsb + dst)
            # t gradient: dw/dt=2, dh/dt=-2
            g[0] += -2.0 * pw * np.sum(dsr) + 2.0 * pw * np.sum(dst)
            dx = xv[IDX_I] - xv[IDX_J]
            dy = yv[IDX_I] - yv[IDX_J]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            viol = rv[IDX_I] + rv[IDX_J] - dist
            ds = d_softplus(viol, beta)
            gdx = pw * ds * (dx / dist)
            gdy = pw * ds * (dy / dist)
            np.add.at(g[1:1 + N], IDX_I, -gdx)
            np.add.at(g[1:1 + N], IDX_J, gdx)
            np.add.at(g[1 + N:1 + 2 * N], IDX_I, -gdy)
            np.add.at(g[1 + N:1 + 2 * N], IDX_J, gdy)
            np.add.at(g[1 + 2 * N:], IDX_I, -pw * ds)
            np.add.at(g[1 + 2 * N:], IDX_J, -pw * ds)
            
            # Equal radius gradient
            r_mean = np.mean(rv)
            g[1 + 2 * N:] += 2.0 * pw_eq * (rv - r_mean)
            
            return g

        bounds = [(0.12, 0.88)] + [(0.0, 2.0)] * (3 * N)
        try:
            if is_polish:
                opts = {'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-13, 'disp': False, 'maxls': 40}
            else:
                opts = {'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False, 'maxls': 40}
            res = minimize(obj, v0, method='L-BFGS-B', jac=grad, bounds=bounds, options=opts)
            if res.fun < obj(v0) - 1e-9:
                return res.x
        except Exception:
            pass
        return v0

    # ---------- exact LP for variable radii given centers ----------
    def solve_radii_lp(w, h, x, y):
        if check_time():
            return None
        c = -np.ones(N)
        A = []; b = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                row = np.zeros(N); row[i] = 1.0; row[j] = 1.0
                A.append(row); b.append(d)
        for i in range(N):
            for bv in (x[i], w - x[i], y[i], h - y[i]):
                if bv > 0:
                    row = np.zeros(N); row[i] = 1.0
                    A.append(row); b.append(bv)
        try:
            res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                          bounds=[(1e-6, None)] * N, method='highs')
            if res.success:
                return res.x
        except Exception:
            pass
        return None

    def refine_variable_radius(state, is_polish=False):
        if check_time():
            return state
        w, h, x, y, r = unpack(state)
        x, y, r = physics(w, h, x, y, r, max_steps=80)
        # Optimize centers to maximize min-distance (creates room), then LP for radii
        v0c = np.zeros(1 + 2 * N)
        v0c[0] = state[0]
        v0c[1:1 + N] = x
        v0c[1 + N:] = y
        beta_c = 90.0

        def cobj(v):
            t = v[0]; wv, hv = get_rect_dims(t)
            xv = v[1:1 + N]; yv = v[1 + N:]
            dx = xv[IDX_I] - xv[IDX_J]; dy = yv[IDX_I] - yv[IDX_J]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            ad = np.concatenate([dist, xv, wv - xv, yv, hv - yv])
            m = np.max(-beta_c * ad)
            lse = m + np.log(np.sum(np.exp(-beta_c * ad - m)))
            return lse / beta_c

        def cgrad(v):
            t = v[0]; wv, hv = get_rect_dims(t)
            xv = v[1:1 + N]; yv = v[1 + N:]
            g = np.zeros_like(v)
            dx = xv[IDX_I] - xv[IDX_J]; dy = yv[IDX_I] - yv[IDX_J]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            ad = np.concatenate([dist, xv, wv - xv, yv, hv - yv])
            nb = -beta_c * ad
            mx = np.max(nb)
            ew = np.exp(nb - mx)
            wt = ew / np.sum(ew)
            wpair = wt[:len(dist)]
            gdx = -beta_c * wpair * (dx / dist)
            gdy = -beta_c * wpair * (dy / dist)
            np.add.at(g[1:1 + N], IDX_I, -gdx)
            np.add.at(g[1:1 + N], IDX_J, gdx)
            np.add.at(g[1 + N:], IDX_I, -gdy)
            np.add.at(g[1 + N:], IDX_J, gdy)
            wp = len(dist)
            wL = wt[wp:wp + N]; wR = wt[wp + N:wp + 2 * N]
            wB = wt[wp + 2 * N:wp + 3 * N]; wT = wt[wp + 3 * N:]
            g[1:1 + N] += -beta_c * wL + beta_c * wR
            g[1 + N:] += -beta_c * wB + beta_c * wT
            g[0] += -beta_c * np.sum(wR) + beta_c * np.sum(wT)
            return g

        try:
            opts = {'maxiter': 300 if is_polish else 150, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            res = minimize(cobj, v0c, method='L-BFGS-B', jac=cgrad,
                           bounds=[(0.12, 0.88)] + [(0.0, 2.0)] * (2 * N), options=opts)
            if res.fun < cobj(v0c) - 1e-9:
                t = res.x[0]; w, h = get_rect_dims(t)
                x = res.x[1:1 + N]; y = res.x[1 + N:]
        except Exception:
            pass
        rvec = solve_radii_lp(w, h, x, y)
        if rvec is None:
            return state
        newstate = np.zeros(1 + 3 * N)
        newstate[0] = state[0]
        newstate[1:1 + N] = x
        newstate[1 + N:1 + 2 * N] = y
        newstate[1 + 2 * N:] = rvec
        return newstate

    # ---------- main search ----------
    best = None
    best_val = -np.inf

    def consider(state):
        nonlocal best, best_val
        w, h, x, y, r = unpack(state)
        if is_valid(w, h, x, y, r, eps=1e-6):
            val = sum_r(w, h, x, y, r)
            if val > best_val:
                best_val = val
                best = state.copy()

    inits = []
    # Focus on square and near-square aspect ratios
    for t in (0.5, 0.48, 0.52, 0.45, 0.55, 0.42, 0.58):
        inits.append(make_hex(t))
    for (rows, cols) in ((5, 5), (4, 6), (6, 4), (3, 7), (7, 3)):
        inits.append(make_grid(0.5, rows, cols))
    for _ in range(6):
        if check_time():
            break
        inits.append(make_random(random.uniform(0.45, 0.55)))

    # Pass 1: physics + softplus smooth over all initial configs
    for st in inits:
        if check_time():
            break
        w, h, x, y, r = unpack(st)
        x, y, r = physics(w, h, x, y, r, max_steps=60)
        st2 = st.copy(); st2[1:1 + N] = x; st2[1 + N:1 + 2 * N] = y; st2[1 + 2 * N:] = r
        consider(st2)
        st3 = refine_smooth(st2, is_polish=False)
        consider(st3)

    # Pass 2: polish the best with softplus (high beta)
    if best is not None and not check_time():
        consider(refine_smooth(best, is_polish=True))

    # Pass 3: variable-radius LP on top candidates
    if best is not None and not check_time():
        consider(refine_variable_radius(best, is_polish=False))

    # Pass 4: final tight polish if time remains
    if best is not None and (deadline - time.time()) > 18.0:
        consider(refine_smooth(best, is_polish=True))
        if (deadline - time.time()) > 10.0:
            consider(refine_variable_radius(best, is_polish=True))

    if best is None:
        # guaranteed valid fallback
        st = make_grid(0.5, 5, 5)
        w, h, x, y, r = unpack(st)
        x, y, r = physics(w, h, x, y, r, max_steps=80)
        rvec = solve_radii_lp(w, h, x, y)
        if rvec is not None:
            st[1 + 2 * N:] = rvec
        best = st

    w, h, x, y, r = unpack(best)
    # safety margin
    r = r * (1 - 1e-7)
    for i in range(N):
        r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
    out = np.column_stack([x, y, r])
    return out
