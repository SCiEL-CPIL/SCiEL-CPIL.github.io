"""Circle packing in perimeter-4 rectangle: Contact-Graph Jiggle + Expansion Local Search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
_TIME_LIMIT = 145.0


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _build_contact_graph(x, y, r, w, h, tol=1e-7):
    """Build contact graph: edges where circles are nearly touching (d ≈ r_i + r_j)."""
    n = len(r)
    contacts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_sum = r[i] + r[j]
            if d < r_sum + tol:
                contacts[i].add(j)
                contacts[j].add(i)
    # Wall contacts
    wall_contacts = [set() for _ in range(n)]  # 0=left,1=right,2=bottom,3=top
    for i in range(n):
        if x[i] < r[i] + tol:
            wall_contacts[i].add(0)
        if x[i] + r[i] > w - tol:
            wall_contacts[i].add(1)
        if y[i] < r[i] + tol:
            wall_contacts[i].add(2)
        if y[i] + r[i] > h - tol:
            wall_contacts[i].add(3)
    return contacts, wall_contacts


def _max_expansion_i(x, y, r, w, h, i, tol=1e-9):
    """Compute maximum radius increase for circle i without violating constraints."""
    r_i = r[i]
    # Wall limits
    max_r_wall = min(x[i], w - x[i], y[i], h - y[i])
    # Neighbor limits
    max_r = max_r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        if d > 1e-12:
            # r_i_new + r_j <= d  =>  r_i_new <= d - r_j
            limit = d - r[j]
            if limit < max_r:
                max_r = limit
    return max(0.0, max_r - r_i)


def _jiggle_step(x, y, r, w, h, dt=0.003, max_iter=200, start_time=None, max_time=10.0):
    """
    Force-directed jiggle: push overlapping circles apart, pull toward center of free space.
    Only moves positions, keeps radii fixed. Returns improved positions.
    """
    if start_time is None:
        start_time = time.time()
    
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    
    for it in range(max_iter):
        if time.time() - start_time > max_time:
            break
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        max_force = 0.0
        
        # Pairwise repulsion for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-12
                min_d = r[i] + r[j]
                if d < min_d:
                    force = (min_d - d) / d * 2.0
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
                    max_force = max(max_force, abs(force))
        
        # Wall repulsion
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 5.0
                max_force = max(max_force, r[i] - x[i])
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 5.0
                max_force = max(max_force, x[i] + r[i] - w)
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 5.0
                max_force = max(max_force, r[i] - y[i])
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 5.0
                max_force = max(max_force, y[i] + r[i] - h)
        
        if max_force < 1e-10:
            break
        
        # Adaptive step
        step = dt if max_force < 0.1 else dt * 0.3
        x += step * fx
        y += step * fy
        
        # Clamp to walls (allow slight penetration to be resolved next iter)
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
    
    return x, y


def _expand_and_jiggle(x, y, r, w, h, start_time=None, max_time=20.0, max_rounds=100):
    """
    Iteratively: find circle with most expansion room, expand it, jiggle to resolve overlaps.
    This is the core local search move.
    """
    if start_time is None:
        start_time = time.time()
    
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    for rnd in range(max_rounds):
        if time.time() - start_time > max_time:
            break
        
        # Find circle with maximum expansion potential
        exps = np.array([_max_expansion_i(x, y, r, w, h, i) for i in range(N)])
        
        if np.max(exps) < 1e-8:
            break
        
        # Pick circle with most room, but prefer those with fewer contacts (easier to expand)
        contacts, wall_contacts = _build_contact_graph(x, y, r, w, h)
        contact_counts = np.array([len(contacts[i]) + len(wall_contacts[i]) for i in range(N)])
        
        # Score: expansion_room / (1 + contact_count)
        scores = exps / (1.0 + contact_counts)
        i_best = int(np.argmax(scores))
        
        # Expand by a fraction of available room
        expand_amount = exps[i_best] * 0.5
        if expand_amount < 1e-10:
            break
        
        r[i_best] += expand_amount
        
        # Jiggle to resolve any new overlaps
        x, y = _jiggle_step(x, y, r, w, h, dt=0.005, max_iter=100,
                           start_time=start_time, max_time=max_time - (time.time() - start_time))
        
        # Verify validity
        if not _is_valid(x, y, r, w, h):
            # Revert
            r[i_best] -= expand_amount
            x, y = _jiggle_step(x, y, r, w, h, dt=0.005, max_iter=50,
                               start_time=start_time, max_time=max_time - (time.time() - start_time))
            if not _is_valid(x, y, r, w, h):
                # Try smaller expansion
                r[i_best] += expand_amount * 0.3
                x, y = _jiggle_step(x, y, r, w, h, dt=0.003, max_iter=50,
                                   start_time=start_time, max_time=max_time - (time.time() - start_time))
                if not _is_valid(x, y, r, w, h):
                    r[i_best] -= expand_amount * 0.3
                    break
    
    return x, y, r


def _slsqp_refine(x, y, r, w, start_time=None, max_time=15.0):
    """
    SLSQP refinement on a valid packing to squeeze out more sum of radii.
    Uses a hard-constraint formulation via penalty but starts from valid config.
    """
    if start_time is None:
        start_time = time.time()
    
    n = N
    z0 = np.zeros(3 * n + 1)
    z0[0::3][:n] = x
    z0[1::3][:n] = y
    z0[2::3][:n] = r
    z0[-1] = w
    
    def obj(z_):
        xx = z_[0::3][:n]
        yy = z_[1::3][:n]
        rr = z_[2::3][:n]
        ww = z_[-1]
        hh = 2.0 - ww
        
        if ww < 0.3 or hh < 0.3 or np.any(rr < 0.001):
            return 1e6
        
        val = -np.sum(rr)
        
        # Pairwise overlap penalty (hard)
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xx[i] - xx[j]
                dy = yy[i] - yy[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = (rr[i] + rr[j]) - d
                if overlap > 0:
                    penalty += overlap * overlap
                # Wall violations
                for viol in [rr[i] - xx[i], xx[i] + rr[i] - ww, rr[i] - yy[i], yy[i] + rr[i] - hh]:
                    if viol > 0:
                        penalty += viol * viol
        return val + 500.0 * penalty
    
    bounds = [(0, 2.0)] * (3 * n) + [(0.4, 1.6)]
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x
        xo = z_opt[0::3][:n]
        yo = z_opt[1::3][:n]
        ro = z_opt[2::3][:n]
        wo = z_opt[-1]
        ho = 2.0 - wo
        if _is_valid(xo, yo, ro, wo, ho):
            return xo, yo, ro, wo
    except Exception:
        pass
    return x, y, r, w


def _cellular_init(w, h, n_circles):
    """Grid-based initial placement guaranteeing disjoint interiors."""
    best = None
    best_score = -1
    for cols in range(1, n_circles + 1):
        rows = int(np.ceil(n_circles / cols))
        if rows * cols < n_circles:
            continue
        cw = w / cols
        ch = h / rows
        r_cell = min(cw, ch) / 2.0 - 1e-6
        score = np.pi * r_cell * r_cell * n_circles / (w * h)
        if score > best_score:
            best_score = score
            best = (cols, rows, cw, ch, r_cell)
    
    cols, rows, cw, ch, r_cell = best
    x = np.zeros(n_circles)
    y = np.zeros(n_circles)
    r = np.zeros(n_circles)
    for i in range(n_circles):
        row, col = divmod(i, cols)
        x[i] = (col + 0.5) * cw
        y[i] = (row + 0.5) * ch
        r[i] = r_cell
    return x, y, r


def _random_valid_init(w, h, seed=42):
    """Generate a random valid packing using sequential placement."""
    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Start with a reasonable base radius
    base_r = min(w, h) / 8.0
    
    for i in range(N):
        placed = False
        for attempt in range(500):
            r_i = base_r * rng.uniform(0.7, 1.3)
            if r_i > w / 2 or r_i > h / 2:
                r_i = min(w, h) / 2 * 0.9
            cx = rng.uniform(r_i, w - r_i)
            cy = rng.uniform(r_i, h - r_i)
            
            # Check against all placed circles
            valid = True
            for j in range(i):
                dx = cx - x[j]
                dy = cy - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r_i + r[j] - 1e-9:
                    valid = False
                    break
            if valid:
                x[i] = cx
                y[i] = cy
                r[i] = r_i
                placed = True
                break
        
        if not placed:
            # Fallback: tiny circle
            r_i = 0.01
            x[i] = r_i + 0.01
            y[i] = r_i + 0.01
            r[i] = r_i
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_x, best_y, best_r, best_w = None, None, None, None
    
    # Phase 1: Generate diverse valid initial packings
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.1, 0.9, 1.4, 0.7, 1.6, 0.65]
    
    packings = []
    
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x, y, r = _cellular_init(w0, h0, N)
        if _is_valid(x, y, r, w0, h0):
            packings.append((x, y, r, w0))
    
    # Add random inits
    for seed in range(10):
        w0 = 0.7 + 0.6 * seed / 9.0
        h0 = 2.0 - w0
        x, y, r = _random_valid_init(w0, h0, seed=100 + seed)
        if _is_valid(x, y, r, w0, h0):
            packings.append((x, y, r, w0))
    
    # Phase 2: Local search on each packing
    for idx, (x, y, r, w) in enumerate(packings):
        if time.time() > deadline - 35:
            break
        
        h = 2.0 - w
        val = float(np.sum(r))
        
        # Jiggle first to resolve any marginal issues
        x, y = _jiggle_step(x, y, r, w, h, dt=0.003, max_iter=100,
                           start_time=start_time, max_time=5.0)
        if not _is_valid(x, y, r, w, h):
            continue
        val = float(np.sum(r))
        
        # Expansion local search
        x_e, y_e, r_e = _expand_and_jiggle(x, y, r, w, h,
                                           start_time=start_time,
                                           max_time=min(15.0, deadline - time.time() - 20),
                                           max_rounds=80)
        if _is_valid(x_e, y_e, r_e, w, h):
            val_e = float(np.sum(r_e))
            if val_e > val:
                x, y, r, val = x_e, y_e, r_e, val_e
        
        # SLSQP refinement
        remaining = deadline - time.time()
        if remaining > 15:
            x_s, y_s, r_s, w_s = _slsqp_refine(x, y, r, w,
                                               start_time=start_time,
                                               max_time=min(12.0, remaining - 5))
            h_s = 2.0 - w_s
            if _is_valid(x_s, y_s, r_s, w_s, h_s):
                val_s = float(np.sum(r_s))
                if val_s > val:
                    x, y, r, w, val = x_s, y_s, r_s, w_s, val_s
        
        # Final jiggle + expansion squeeze
        if time.time() < deadline - 10:
            h_f = 2.0 - w
            x, y = _jiggle_step(x, y, r, w, h_f, dt=0.002, max_iter=50,
                               start_time=start_time, max_time=3.0)
            if _is_valid(x, y, r, w, h_f):
                x_f, y_f, r_f = _expand_and_jiggle(x, y, r, w, h_f,
                                                   start_time=start_time,
                                                   max_time=min(5.0, deadline - time.time() - 5),
                                                   max_rounds=30)
                if _is_valid(x_f, y_f, r_f, w, h_f):
                    val_f = float(np.sum(r_f))
                    if val_f > val:
                        x, y, r, val = x_f, y_f, r_f, val_f
        
        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w
    
    # Phase 3: Final polish on best
    if best_x is not None and time.time() < deadline - 10:
        h_b = 2.0 - best_w
        # One more SLSQP
        x_p, y_p, r_p, w_p = _slsqp_refine(best_x, best_y, best_r, best_w,
                                           start_time=start_time,
                                           max_time=min(8.0, deadline - time.time() - 3))
        h_p = 2.0 - w_p
        if _is_valid(x_p, y_p, r_p, w_p, h_p):
            val_p = float(np.sum(r_p))
            if val_p > best_value:
                best_value = val_p
                best_x, best_y, best_r, best_w = x_p, y_p, r_p, w_p
                h_b = h_p
        
        # Final expansion
        if time.time() < deadline - 3:
            x_f2, y_f2, r_f2 = _expand_and_jiggle(best_x, best_y, best_r, best_w, h_b,
                                                  start_time=start_time,
                                                  max_time=min(3.0, deadline - time.time() - 1),
                                                  max_rounds=20)
            if _is_valid(x_f2, y_f2, r_f2, best_w, h_b):
                val_f2 = float(np.sum(r_f2))
                if val_f2 > best_value:
                    best_x, best_y, best_r = x_f2, y_f2, r_f2
    
    if best_x is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Safety shrink
    for _ in range(20):
        r_trial = best_r * (1 - 1e-9)
        if _is_valid(best_x, best_y, r_trial, best_w, 2.0 - best_w):
            best_r = r_trial
        else:
            break
    
    return np.column_stack((best_x, best_y, best_r))
