"""
Sutton-Chen transition-metal cluster global minima via multi-start basin-hopping
with parallel relaxation, diverse structural templates, and adaptive step control.
"""

import time
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_GLOBAL_START = time.time()
_HARD_LIMIT = 325.0  # Safety margin for 330s kill

def _remaining():
    return max(0.0, _HARD_LIMIT - (time.time() - _GLOBAL_START))

def _fcc_template(n, scale=1.0):
    """Generate FCC-like template with n atoms."""
    a = scale * np.sqrt(2.0)
    r_est = (3.0 * n * a**3 / (16.0 * np.pi)) ** (1.0/3.0) * 1.3
    nside = int(np.ceil(r_est / a)) + 4
    cands = []
    for i in range(-nside, nside + 1):
        for j in range(-nside, nside + 1):
            for k in range(-nside, nside + 1):
                base = np.array([i, j, k], dtype=float) * a
                for off in [(0.0,0.0,0.0),(0.5,0.5,0.0),(0.5,0.0,0.5),(0.0,0.5,0.5)]:
                    pt = base + np.array(off, dtype=float) * a
                    if np.dot(pt, pt) <= r_est * r_est:
                        cands.append(pt)
    if len(cands) < n:
        r_est *= 1.5
        nside = int(np.ceil(r_est / a)) + 4
        cands = []
        for i in range(-nside, nside + 1):
            for j in range(-nside, nside + 1):
                for k in range(-nside, nside + 1):
                    base = np.array([i, j, k], dtype=float) * a
                    for off in [(0.0,0.0,0.0),(0.5,0.5,0.0),(0.5,0.0,0.5),(0.0,0.5,0.5)]:
                        pt = base + np.array(off, dtype=float) * a
                        if np.dot(pt, pt) <= r_est * r_est:
                            cands.append(pt)
    cands = np.array(cands)
    d = np.linalg.norm(cands, axis=1)
    order = np.argsort(d)
    cands = cands[order[:n]]
    return cands - cands.mean(axis=0)

def _hcp_template(n, scale=1.0):
    """Generate HCP-like template with n atoms."""
    a = scale
    c = a * np.sqrt(8.0 / 3.0)
    nlay = int(np.ceil(np.sqrt(n))) + 5
    nside = int(np.ceil(np.sqrt(n))) + 5
    r2 = (n / 4.0) ** (2.0/3.0) * a**2 * 4.0
    cands = []
    for layer in range(-nlay, nlay + 1):
        z = layer * c
        sx = 0.0 if layer % 2 == 0 else a / 2.0
        sy = 0.0 if layer % 2 == 0 else a * np.sqrt(3.0) / 6.0
        for i in range(-nside, nside + 1):
            for j in range(-nside, nside + 1):
                x = i * a + j * a * 0.5 + sx
                y = j * a * np.sqrt(3.0) / 2.0 + sy
                if x*x + y*y + z*z <= r2:
                    cands.append([x, y, z])
    cands = np.array(cands)
    if len(cands) < n:
        rng = np.random.default_rng(7)
        while len(cands) < n:
            cands = np.vstack([cands, rng.uniform(-2*a, 2*a, 3)])
    d = np.linalg.norm(cands, axis=1)
    order = np.argsort(d)
    cands = cands[order[:n]]
    return cands - cands.mean(axis=0)

def _bcc_template(n, scale=1.0):
    """Generate BCC-like template with n atoms."""
    a = scale * 2.0 / np.sqrt(3.0)
    r_est = (3.0 * n * a**3 / (8.0 * np.pi)) ** (1.0/3.0) * 1.3
    nside = int(np.ceil(r_est / a)) + 4
    cands = []
    for i in range(-nside, nside + 1):
        for j in range(-nside, nside + 1):
            for k in range(-nside, nside + 1):
                base = np.array([i, j, k], dtype=float) * a
                for off in [(0.0,0.0,0.0),(0.5,0.5,0.5)]:
                    pt = base + np.array(off, dtype=float) * a
                    if np.dot(pt, pt) <= r_est * r_est:
                        cands.append(pt)
    if len(cands) < n:
        r_est *= 1.5
        nside = int(np.ceil(r_est / a)) + 4
        cands = []
        for i in range(-nside, nside + 1):
            for j in range(-nside, nside + 1):
                for k in range(-nside, nside + 1):
                    base = np.array([i, j, k], dtype=float) * a
                    for off in [(0.0,0.0,0.0),(0.5,0.5,0.5)]:
                        pt = base + np.array(off, dtype=float) * a
                        if np.dot(pt, pt) <= r_est * r_est:
                            cands.append(pt)
    cands = np.array(cands)
    d = np.linalg.norm(cands, axis=1)
    order = np.argsort(d)
    cands = cands[order[:n]]
    return cands - cands.mean(axis=0)

def _icosahedral_template(n, scale=1.0):
    """Generate icosahedral-like template with n atoms."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
        [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
        [phi,0,1],[phi,0,-1],[-phi,0,1],[-phi,0,-1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    pts = [np.zeros(3)]
    if n > 1:
        for v in verts:
            if len(pts) < n:
                pts.append(v * scale)
    a = scale * np.sqrt(2.0)
    if len(pts) < n:
        for i in range(-6, 7):
            for j in range(-6, 7):
                for k in range(-6, 7):
                    base = np.array([i, j, k], dtype=float) * a
                    for off in [(0.0,0.0,0.0),(0.5,0.5,0.0),(0.5,0.0,0.5),(0.0,0.5,0.5)]:
                        p = base + np.array(off, dtype=float) * a
                        r = np.linalg.norm(p)
                        if 1.3 * scale < r < 3.5 * scale and len(pts) < n:
                            pts.append(p / r * 2.2 * scale)
    rng = np.random.default_rng(3)
    while len(pts) < n:
        p = rng.uniform(-4*scale, 4*scale, 3)
        r = np.linalg.norm(p)
        if r > 0.1:
            pts.append(p / r * (1.8 + rng.random() * 1.5) * scale)
    pts = np.array(pts[:n])
    return pts - pts.mean(axis=0)

def _random_packed(n, scale=1.0, seed=0):
    """Generate randomly packed template with n atoms."""
    rng = np.random.default_rng(seed)
    radius = scale * (n / 4.0) ** (1.0/3.0) * 3.0
    pts = []
    min_d = 0.55 * scale
    for _ in range(n * 300):
        if len(pts) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.dot(p, p) > radius * radius:
            continue
        if not pts:
            pts.append(p)
            continue
        arr = np.array(pts)
        diffs = arr - p
        if np.min(np.linalg.norm(diffs, axis=1)) >= min_d:
            pts.append(p)
    while len(pts) < n:
        pts.append(rng.uniform(-radius, radius, 3))
    pts = np.array(pts[:n])
    return pts - pts.mean(axis=0)

def _enforce_min_dist(pos, thresh=0.36):
    """Ensure all pairwise distances are above threshold."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(100):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < thresh:
                    moved = True
                    if d > 1e-12:
                        dirn = (pos[i] - pos[j]) / d
                    else:
                        dirn = np.array([1.0, 0.0, 0.0])
                    shift = (thresh - d) / 2.0 + 0.01
                    pos[i] += dirn * shift
                    pos[j] -= dirn * shift
        if not moved:
            break
    return pos

def _polish(pos, kind):
    """High-precision L-BFGS-B polish."""
    def f(x):
        pp = x.reshape(-1, 3)
        e, _ = sutton_chen_energy_gradient(pp, kind)
        return e if np.isfinite(e) else 1e12

    def g(x):
        pp = x.reshape(-1, 3)
        _, gg = sutton_chen_energy_gradient(pp, kind)
        if not np.all(np.isfinite(gg)):
            return np.zeros_like(gg).ravel()
        return gg.ravel()

    x0 = pos.ravel()
    try:
        res = minimize(f, x0, jac=g, method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-25, 'gtol': 1e-20, 'maxls': 2000})
        if np.isfinite(res.fun) and np.all(np.isfinite(res.x)):
            fp = res.x.reshape(-1, 3)
            # Check min distance
            if np.min(np.linalg.norm(fp[:, None, :] - fp[None, :, :], axis=2)[np.triu_indices(len(fp), 1)]) > 0.35:
                return res.fun, fp
    except Exception:
        pass
    return relax(pos, 'sutton_chen', kind)

def _bh_single(start_pos, potential, kind, rng, n_iter, scale0, scale_min, temp, grad_prob=0.5):
    """Single basin-hopping trajectory."""
    best_e, best_p = relax(start_pos, potential, kind)
    cur_e, cur_p = best_e, best_p.copy()
    cur_scale = scale0
    stagnant = 0
    
    for it in range(n_iter):
        if _remaining() < 10.0:
            break
        
        # Adaptive scale
        if stagnant > 5:
            cur_scale = min(scale0, cur_scale * 1.25)
        else:
            cur_scale = max(scale_min, cur_scale * 0.92)
        
        # Generate perturbation
        if rng.random() < grad_prob:
            try:
                _, grad = sutton_chen_energy_gradient(cur_p, kind)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-8:
                    grad_dir = grad / grad_norm
                    step = cur_scale * 0.6
                    noise = rng.normal(0.0, cur_scale * 0.4, cur_p.shape)
                    new_p = cur_p - grad_dir * step + noise
                else:
                    noise = rng.normal(0.0, cur_scale, cur_p.shape)
                    new_p = cur_p + noise
            except Exception:
                noise = rng.normal(0.0, cur_scale, cur_p.shape)
                new_p = cur_p + noise
        else:
            noise = rng.normal(0.0, cur_scale, cur_p.shape)
            new_p = cur_p + noise

        try:
            e, p = relax(new_p, potential, kind)
        except Exception:
            continue
        
        if not np.isfinite(e):
            continue
        
        accept = False
        if e < cur_e:
            accept = True
            stagnant = 0
        elif temp > 0:
            T = temp * (1.0 + it / 80.0)
            if rng.random() < np.exp(-(e - cur_e) / T):
                accept = True
                stagnant += 1
        else:
            stagnant += 1

        if accept:
            cur_e, cur_p = e, p.copy()
        
        if e < best_e:
            best_e, best_p = e, p.copy()
            stagnant = 0
            
        # Restart from best if stuck too long
        if stagnant > 18:
            cur_p = best_p.copy()
            cur_e = best_e
            stagnant = 0
            cur_scale = scale0

    return best_e, best_p

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(42 + n * 7)

    best_e = np.inf
    best_p = None

    # Generate diverse templates
    templates = []
    try:
        templates.append(_fcc_template(n))
    except Exception:
        pass
    try:
        templates.append(_hcp_template(n))
    except Exception:
        pass
    try:
        templates.append(_bcc_template(n))
    except Exception:
        pass
    try:
        templates.append(_icosahedral_template(n))
    except Exception:
        pass
    # Add multiple random templates with different seeds
    for s in range(20):
        try:
            templates.append(_random_packed(n, seed=s))
        except Exception:
            pass

    # Relax all templates in parallel (up to 4 threads)
    relaxed = []
    if templates:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for idx, t in enumerate(templates):
                if _remaining() < 20.0:
                    break
                fut = executor.submit(relax, t, potential, kind)
                futures[fut] = idx
            
            for fut in as_completed(futures):
                if _remaining() < 15.0:
                    break
                try:
                    e, p = fut.result()
                    if np.isfinite(e):
                        relaxed.append((e, p))
                        if e < best_e:
                            best_e, best_p = e, p.copy()
                except Exception:
                    pass

    if best_p is None:
        best_p = _random_packed(n, seed=999)
        best_e, best_p = relax(best_p, potential, kind)

    # Sort relaxed by energy
    relaxed.sort(key=lambda x: x[0])

    # Main basin hopping from best start
    if _remaining() > 40.0:
        n_iters = max(60, int((_remaining() - 30.0) * 3.5))
        try:
            e, p = _bh_single(best_p, potential, kind, rng, n_iters, 0.7, 0.1, 0.05, grad_prob=0.6)
            if e < best_e:
                best_e, best_p = e, p.copy()
        except Exception:
            pass

    # Basin hopping from other good starts
    for e_t, p_t in relaxed[:12]:
        if _remaining() < 25.0:
            break
        if e_t > best_e + 15.0:
            continue
        try:
            e, p = _bh_single(p_t, potential, kind, rng, 50, 0.6, 0.1, 0.04, grad_prob=0.5)
            if e < best_e:
                best_e, best_p = e, p.copy()
        except Exception:
            pass

    # High-precision polish
    if _remaining() > 20.0:
        try:
            e_pol, p_pol = _polish(best_p, kind)
            if np.isfinite(e_pol) and e_pol < best_e:
                best_e, best_p = e_pol, p_pol.copy()
        except Exception:
            pass

    # Second round of basin hopping from polished structure
    if _remaining() > 50.0:
        try:
            e, p = _bh_single(best_p, potential, kind, rng, 60, 0.5, 0.05, 0.03, grad_prob=0.7)
            if e < best_e:
                best_e, best_p = e, p.copy()
        except Exception:
            pass

        # Final polish
        if _remaining() > 18.0:
            try:
                e_pol, p_pol = _polish(best_p, kind)
                if np.isfinite(e_pol) and e_pol < best_e:
                    best_e, best_p = e_pol, p_pol.copy()
            except Exception:
                pass

    # Third round if time permits
    if _remaining() > 60.0:
        try:
            e, p = _bh_single(best_p, potential, kind, rng, 40, 0.4, 0.03, 0.02, grad_prob=0.8)
            if e < best_e:
                best_e, best_p = e, p.copy()
        except Exception:
            pass

        if _remaining() > 15.0:
            try:
                e_pol, p_pol = _polish(best_p, kind)
                if np.isfinite(e_pol) and e_pol < best_e:
                    best_e, best_p = e_pol, p_pol.copy()
            except Exception:
                pass

    # Enforce minimum distance
    if best_p is not None:
        best_p = _enforce_min_dist(best_p, 0.36)

    return best_p
