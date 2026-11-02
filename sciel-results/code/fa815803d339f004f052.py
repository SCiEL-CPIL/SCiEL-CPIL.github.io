import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0   # safety margin vs 150 s

    # ---------- geometry helpers ----------
    def hex_vertices(cx, cy, side, ang):
        a = ang + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + side*np.cos(a), cy + side*np.sin(a)])

    def sat_min_overlap(V1, V2):
        """Minimum translation depth of V1 into V2 (0 if separated). 6x2 each."""
        edges = np.roll(V1, -1, 0) - V1
        edges = np.vstack([edges, np.roll(V2, -1, 0) - V2])
        n = np.column_stack([edges[:,1], -edges[:,0]])
        nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-15
        n = n / nn
        p1 = V1 @ n.T; p2 = V2 @ n.T
        ov = np.minimum(p1.max(0), p2.max(0)) - np.maximum(p1.min(0), p2.min(0))
        return float(ov.min())

    def sat_penetration(V_inner, V_outer):
        """How far V_inner pokes outside V_outer (0 if fully inside)."""
        edges = np.roll(V_outer, -1, 0) - V_outer
        n = np.column_stack([edges[:,1], -edges[:,0]])
        nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-15
        n = n / nn
        p_in = V_inner @ n.T
        p_out = (V_outer @ n.T).max(0)
        return float(np.maximum(0.0, np.max(p_in.max(0) - p_out)))

    def smooth_max(x, k=50.0):
        """Soft max using log-sum-exp."""
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(k*(x - x_max)))) / k

    # ---------- objective ----------
    def make_objective(alpha, theta):
        def f(p):
            if time.time() > DEADLINE:
                return 1e8
            pos = p[:2*N].reshape(N, 2)
            s = p[2*N]
            if s < 2.0:
                return 1e6

            c = np.cos(alpha); sn = np.sin(alpha)
            # unit hexagon vertices (flat-top at alpha=0)
            base = np.array([[1.0,0.0],[0.5,np.sqrt(3)/2],[-0.5,np.sqrt(3)/2],
                             [-1.0,0.0],[-0.5,-np.sqrt(3)/2],[0.5,-np.sqrt(3)/2]])
            R = np.array([[c,-sn],[sn,c]])
            verts = pos[:,None,:] + (base @ R.T)[None,:,:]   # N,6,2

            outer = hex_vertices(0.0, 0.0, s, theta)

            # containment: for each inner, max over 6 outer normals
            oedges = np.roll(outer,-1,0)-outer
            on = np.column_stack([oedges[:,1],-oedges[:,0]])
            on /= (np.linalg.norm(on,axis=1,keepdims=True)+1e-15)
            p_in = np.einsum('nij,jk->nik', verts, on.T)      # N,6,6
            p_out = (outer @ on.T).max(0)                     # 6
            # Use smooth max for better gradient behavior
            contain = smooth_max(p_in.max(1) - p_out[None,:], k=50.0)
            contain = np.maximum(0.0, contain)

            # pairwise: 12 SAT axes per pair
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    A = verts[i]; B = verts[j]
                    eA = np.roll(A,-1,0)-A
                    eB = np.roll(B,-1,0)-B
                    ed = np.vstack([eA,eB])
                    nn = np.column_stack([ed[:,1],-ed[:,0]])
                    nn /= (np.linalg.norm(nn,axis=1,keepdims=True)+1e-15)
                    pA = A@nn.T; pB = B@nn.T
                    ov = np.minimum(pA.max(0),pB.max(0))-np.maximum(pA.min(0),pB.min(0))
                    pen += max(0.0, ov.min())

            return s + 50.0*contain.sum() + 50.0*pen
        return f

    def is_valid(pos, s, alpha, theta, tol=1e-4):
        c = np.cos(alpha); sn = np.sin(alpha)
        base = np.array([[1.0,0.0],[0.5,np.sqrt(3)/2],[-0.5,np.sqrt(3)/2],
                         [-1.0,0.0],[-0.5,-np.sqrt(3)/2],[0.5,-np.sqrt(3)/2]])
        R = np.array([[c,-sn],[sn,c]])
        verts = pos[:,None,:] + (base @ R.T)[None,:,:]
        outer = hex_vertices(0.0,0.0,s,theta)
        for i in range(N):
            if sat_penetration(verts[i], outer) > tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                if sat_min_overlap(verts[i], verts[j]) > tol:
                    return False
        return True

    # ---------- tight initial layout ----------
    def build_init():
        # 1 center, 6 inner ring, 5 outer ring
        d1 = np.sqrt(3)
        d2 = 2*np.sqrt(3)
        pos = [[0.0,0.0]]
        for k in range(6):
            a = np.deg2rad(60*k)
            pos.append([d1*np.cos(a), d1*np.sin(a)])
        for k in range(5):
            a = np.deg2rad(60*k)
            pos.append([d2*np.cos(a), d2*np.sin(a)])
        return np.array(pos)

    init_pos = build_init()
    init_s = 3.7

    best = None
    best_s = np.inf
    best_alpha = 0.0
    best_theta = 0.0

    alphas = [0.0, np.deg2rad(30)]
    thetas = [0.0, np.deg2rad(30)]

    for alpha in alphas:
        if time.time() > DEADLINE - 25:
            break
        for theta in thetas:
            if time.time() > DEADLINE - 25:
                break
            obj = make_objective(alpha, theta)
            x0 = np.concatenate([init_pos.ravel(), [init_s]])
            try:
                res = minimize(obj, x0, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8,
                                        'maxls': 50})
                pos = res.x[:2*N].reshape(N,2)
                s = res.x[2*N]
                if is_valid(pos, s, alpha, theta):
                    if s < best_s:
                        best_s = s
                        best = pos.copy()
                        best_alpha = alpha
                        best_theta = theta
            except Exception:
                pass

        # simulated-annealing polish on positions (fix s, alpha, theta)
        if time.time() > DEADLINE - 30:
            continue
        # re-run SA to improve the best-so-far for this (alpha,theta)
        if best is not None and np.isclose(best_alpha, alpha) and np.isclose(best_theta, theta):
            pass
        # generic SA polish
        def sa_polish(pos, s, alpha, theta):
            obj = make_objective(alpha, theta)
            cur = np.concatenate([pos.ravel(), [s]])
            cur_f = obj(cur)
            cur_pos = cur[:2*N].reshape(N,2)
            cur_s = cur[2*N]
            cur_best = (cur_s if is_valid(cur_pos, cur_s, alpha, theta) else np.inf)
            cur_best_pos = cur_pos.copy() if np.isfinite(cur_best) else None
            T = 0.03
            rng = np.random.default_rng(12345)
            steps = 0
            while time.time() < DEADLINE - 28 and steps < 6000:
                steps += 1
                i = rng.integers(0, N)
                d = rng.normal(0, T, size=2)
                new_pos = cur_pos.copy()
                new_pos[i] += d
                new_s = cur_s
                if not is_valid(new_pos, new_s, alpha, theta, tol=1e-3):
                    # penalize heavily
                    f_new = obj(np.concatenate([new_pos.ravel(), [new_s]]))
                    accept = (f_new < cur_f) or (rng.random() < np.exp(-(f_new-cur_f)/max(T,1e-6)))
                else:
                    f_new = new_s
                    accept = True
                if accept:
                    cur_pos = new_pos
                    cur_f = f_new
                    if is_valid(new_pos, new_s, alpha, theta) and new_s < cur_best:
                        cur_best = new_s
                        cur_best_pos = new_pos.copy()
                T = max(0.001, T*0.998)
            return cur_best_pos, cur_best
        # only polish if we have a candidate for this orientation
        cand_pos = None
        cand_s = np.inf
        try:
            res2 = minimize(obj, x0, method='L-BFGS-B',
                            options={'maxiter': 400, 'ftol': 1e-9, 'gtol': 1e-7})
            p2 = res2.x[:2*N].reshape(N,2); s2 = res2.x[2*N]
            if is_valid(p2, s2, alpha, theta):
                cand_pos, cand_s = p2, s2
        except Exception:
            pass
        if cand_pos is not None:
            try:
                pp, ps = sa_polish(cand_pos, cand_s, alpha, theta)
                if pp is not None and ps < best_s:
                    best_s = ps
                    best = pp.copy()
                    best_alpha = alpha
                    best_theta = theta
            except Exception:
                pass

    # ---------- fallback ----------
    if best is None:
        # return a known-valid flower layout with a safe side
        s = 4.0
        pos = init_pos
        # verify, else enlarge
        for a in [0.0, np.deg2rad(30)]:
            for th in [0.0, np.deg2rad(30)]:
                if is_valid(pos, s, a, th):
                    best = pos; best_s = s; best_alpha = a; best_theta = th
                    break
            if best is not None:
                break
    if best is None:
        # last resort: enlarge until valid
        s = 3.8
        while s < 6.0:
            for a in [0.0, np.deg2rad(30)]:
                for th in [0.0, np.deg2rad(30)]:
                    if is_valid(init_pos, s, a, th):
                        best = init_pos; best_s = s; best_alpha = a; best_theta = th
                        break
                if best is not None:
                    break
            if best is not None:
                break
            s += 0.1
    if best is None:
        best = init_pos; best_s = 5.0; best_alpha = 0.0; best_theta = 0.0

    inner = [[float(best[i,0]), float(best[i,1]), float(np.rad2deg(best_alpha))] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = float(best_s)
    outer_angle_deg = float(np.rad2deg(best_theta))
    return inner, outer_center, outer_side, outer_angle_deg
