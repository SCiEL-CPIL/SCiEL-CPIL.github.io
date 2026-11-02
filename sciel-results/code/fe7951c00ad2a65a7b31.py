"""Pack 300 equal circles in the unit disk via soft-min gradient optimization."""
import time
import numpy as np

N = 300
DEADLINE = 140.0
SEED = 12345

# ---------- helpers ----------

def _softmin(d, beta):
    """Smooth approximation of min(d) via log-sum-exp (vectorized)."""
    m = d.min()
    z = d - m
    a = np.exp(-beta * z)
    return m - np.log(a.sum()) / beta


def _softmin_grad(d, beta):
    """Gradient of softmin(d) w.r.t. d."""
    m = d.min()
    z = d - m
    a = np.exp(-beta * z)
    w = a / a.sum()
    return w


def _init_hexagonal():
    """Hexagonal lattice trimmed to the unit disk, ordered by radius."""
    rng = np.random.default_rng(SEED)
    for spacing in np.linspace(0.030, 0.055, 60):
        k = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                pts.append((x, y))
        pts = np.array(pts)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        mask = rad <= 1.0 - spacing / 2.0
        pts = pts[mask]
        if len(pts) >= N:
            idx = np.argsort(rad, kind="stable")[:N]
            p = pts[idx].copy()
            # tiny random jitter to break symmetry
            p += rng.normal(0, spacing * 0.02, size=p.shape)
            return p
    # fallback: concentric rings
    p = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt((i + 1) / N) * 0.99
        th = i * 2.399963229728653
        p[i] = [r * np.cos(th), r * np.sin(th)]
    return p


def _score(pts):
    """Hard score = min(pairwise half-dist, boundary margin)."""
    n = len(pts)
    d = np.hypot(pts[:, 0][None, :] - pts[:, 0][None, :],
                 pts[:, 1][None, :] - pts[:, 1][None, :])
    np.fill_diagonal(d, np.inf)
    r_pairs = d.min() / 2.0
    r_bound = 1.0 - np.hypot(pts[:, 0], pts[:, 1]).max()
    return min(r_pairs, r_bound)


# ---------- main ----------

def construct():
    t0 = time.time()
    pts = _init_hexagonal()

    beta = 200.0
    lr = 0.02
    best = pts.copy()
    best_score = _score(pts)

    def _elapsed():
        return time.time() - t0

    while _elapsed() < DEADLINE:
        # --- compute softmin objective and gradient ---
        n = len(pts)
        # pairwise distances
        dx = pts[:, 0][None, :] - pts[:, 0][None, :]
        dy = pts[:, 1][None, :] - pts[:, 1][None, :]
        d = np.hypot(dx, dy)
        np.fill_diagonal(d, np.inf)

        # soft-min of pairwise distances / 2  ->  r_pairs_smooth
        d_pair = d / 2.0
        m = d_pair.min()
        z = d_pair - m
        a = np.exp(-beta * z)
        a[~np.isfinite(a)] = 0.0
        a_sum = a.sum()
        if a_sum <= 0:
            a_sum = 1.0
        w = a / a_sum
        r_pairs = m - np.log(a_sum) / beta

        # boundary margin
        rad = np.hypot(pts[:, 0], pts[:, 1])
        margin = 1.0 - rad
        r_bound = margin.min()

        # combined soft objective: softmin of [r_pairs, r_bound]
        # treat r_pairs and r_bound as two "values"; use softmin over the two
        # plus the pairwise contribution
        obj = r_pairs
        # gradient of r_pairs w.r.t. pts
        # d(r_pairs)/d(d_pair) = w
        # d(d_pair)/d(pts_i) = (pts_i - pts_j)/|pts_i - pts_j|  (for the pair)
        # accumulate
        grad = np.zeros_like(pts)
        # w is n x n, symmetric; only upper triangle needed but use full
        # d_pair[i,j] = |p_i - p_j|/2
        # d d_pair[i,j] / d p_i = (p_i - p_j) / (2 * |p_i - p_j|)
        # d d_pair[i,j] / d p_j = (p_j - p_i) / (2 * |p_i - p_j|)
        # d r_pairs / d p_i = sum_j w[i,j] * (p_i - p_j) / (2 * d[i,j])
        #                     - sum_j w[j,i] * (p_i - p_j) / (2 * d[j,i])
        # Since w is symmetric and d is symmetric, this simplifies:
        # d r_pairs / d p_i = sum_j w[i,j] * (p_i - p_j) / d[i,j]
        # (the factor of 2 cancels with the 1/2 from d_pair = d/2)
        # Let's just compute directly.
        # d_pair[i,j] = 0.5 * d[i,j]
        # d r_pairs/d p_i = sum_j w[i,j] * d(d_pair[i,j])/d p_i
        # d(d_pair[i,j])/d p_i = (p_i - p_j) / (2 * d[i,j])
        # So d r_pairs/d p_i = sum_j w[i,j] * (p_i - p_j) / (2 * d[i,j])
        # We want to MAXIMIZE r_pairs, so gradient ascent.
        # But we also want to maximize r_bound = min_i margin_i.
        # Use a combined approach: gradient ascent on a smooth objective that
        # is large when both r_pairs and r_bound are large.
        # Simple: objective = min(r_pairs, r_bound) approximated smoothly.
        # For gradient, use: if r_pairs < r_bound, push pairs apart; else push toward center.
        # Smooth version:
        # Let f = r_pairs, g = r_bound.  softmin(f,g) with beta2.
        beta2 = 50.0
        fg = np.array([r_pairs, r_bound])
        mfg = fg.min()
        zfg = fg - mfg
        afg = np.exp(-beta2 * zfg)
        afg_sum = afg.sum()
        wfg = afg / afg_sum  # wfg[0] for r_pairs, wfg[1] for r_bound

        # gradient of r_pairs w.r.t. pts
        # d r_pairs / d p_i = sum_j w[i,j] * (p_i - p_j) / (2 * d[i,j])
        # Build vectorized: for each i, sum over j of w[i,j] * (p_i - p_j) / (2*d[i,j])
        # = p_i * sum_j w[i,j]/(2*d[i,j])  -  sum_j w[i,j]*(p_j)/(2*d[i,j])
        inv_d = np.zeros_like(d)
        mask = d > 1e-12
        inv_d[mask] = 1.0 / (2.0 * d[mask])
        # grad_pairs[i] = p_i * (w @ inv_d)[i]  -  (w * inv_d) @ pts  ... careful
        # Let S[i] = sum_j w[i,j] * inv_d[i,j]
        S = w @ inv_d
        # T[i] = sum_j w[i,j] * inv_d[i,j] * p_j
        W_inv = w * inv_d
        T = W_inv @ pts
        grad_pairs = pts * S[:, None] - T

        # gradient of r_bound w.r.t. pts
        # r_bound = min_i (1 - |p_i|), smooth approx:
        # r_bound_smooth = m_bound - log(sum exp(-beta3*(margin - m_bound)))/beta3
        # For simplicity use the same softmin with a moderate beta3
        beta3 = 50.0
        mb = margin.min()
        zb = margin - mb
        ab = np.exp(-beta3 * zb)
        ab_sum = ab.sum()
        wb = ab / ab_sum
        # d r_bound / d p_i = -wb[i] * p_i / |p_i|
        grad_bound = -wb[:, None] * pts / np.maximum(rad[:, None], 1e-12)

        # combined gradient (ascent direction)
        grad = wfg[0] * grad_pairs + wfg[1] * grad_bound

        # --- adaptive step ---
        # normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break
        step = lr * grad / gnorm

        # --- update ---
        new_pts = pts + step
        # clip to stay inside disk (with small margin)
        new_rad = np.hypot(new_pts[:, 0], new_pts[:, 1])
        too_far = new_rad > 0.999
        if np.any(too_far):
            new_pts[too_far] *= 0.999 / new_rad[too_far, None]

        new_score = _score(new_pts)
        if new_score > best_score:
            best = new_pts.copy()
            best_score = new_score
            pts = new_pts
            # increase beta slowly to sharpen
            beta = min(beta * 1.05, 5000.0)
            lr = min(lr * 1.01, 0.05)
        else:
            # reject, shrink step
            lr *= 0.95
            if lr < 1e-6:
                lr = 0.01
                # small random kick
                pts += np.random.default_rng(int(time.time() * 1e6) % 100000).normal(0, 0.001, size=pts.shape)

    # --- final polish: local pairwise refinement on best ---
    # A few passes of pushing close pairs apart and pulling boundary points in
    for _pass in range(3):
        if _elapsed() > DEADLINE - 2:
            break
        for _it in range(50):
            if _elapsed() > DEADLINE - 2:
                break
            b = best.copy()
            n = len(b)
            dx = b[:, 0][None, :] - b[:, 0][None, :]
            dy = b[:, 1][None, :] - b[:, 1][None, :]
            d = np.hypot(dx, dy)
            np.fill_diagonal(d, np.inf)
            # for each pair closer than current best radius*2, push apart
            cur_r = _score(b)
            min_d = d.min()
            # push all pairs that are close
            push = np.zeros_like(b)
            for i in range(n):
                for j in range(i + 1, n):
                    dij = d[i, j]
                    if dij < min_d * 1.2:
                        f = 0.5 * (min_d - dij) / max(dij, 1e-12)
                        push[i, 0] += f * dx[i, j]
                        push[i, 1] += f * dy[i, j]
                        push[j, 0] -= f * dx[i, j]
                        push[j, 1] -= f * dy[i, j]
            # boundary: pull points too close to edge inward
            rad = np.hypot(b[:, 0], b[:, 1])
            for i in range(n):
                if rad[i] > 1.0 - cur_r:
                    f = 0.5 * (rad[i] - (1.0 - cur_r)) / max(rad[i], 1e-12)
                    push[i, 0] -= f * b[i, 0]
                    push[i, 1] -= f * b[i, 1]
            b += 0.1 * push
            # clip
            rad2 = np.hypot(b[:, 0], b[:, 1])
            tf = rad2 > 0.9999
            if np.any(tf):
                b[tf] *= 0.9999 / rad2[tf, None]
            if _score(b) > cur_r:
                best = b
            else:
                break
        # recompute best_score
        best_score = _score(best)

    # final validity check
    rad = np.hypot(best[:, 0], best[:, 1])
    if np.any(rad > 1.0) or np.any(~np.isfinite(best)):
        # fall back to hexagonal
        best = _init_hexagonal()

    return best
