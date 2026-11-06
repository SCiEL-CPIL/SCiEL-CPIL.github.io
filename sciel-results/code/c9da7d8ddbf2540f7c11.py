"""
Universal 8-node positive quadrature for dimensionless luminosity distance.
Strategy: minimax LP for weights + Nelder-Mead node optimization.
"""
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog, minimize
from scipy.special import roots_legendre

_START = time.time()
_DEADLINE = _START + 86.0

def _E(z, w0, wa, Om=0.3):
    Ode = 1.0 - Om
    opz = 1.0 + z
    ex = 3.0 + 3.0*w0 + 3.0*wa*z
    return np.sqrt(Om * opz**3 + Ode * opz**ex)

def _ref(z, w0, wa, Om=0.3):
    if z < 1e-12:
        return 1.0
    def f(t):
        return 1.0 / _E(z * t, w0, wa, Om)
    v, _ = quad(f, 0, 1, limit=300, epsabs=1e-13, epsrel=1e-13)
    return v

def _lp_weights(A, refs, n):
    N = A.shape[0]
    c = np.zeros(n + 1)
    c[n] = 1.0
    A_ub = np.zeros((2*N, n + 1))
    b_ub = np.zeros(2*N)
    for i in range(N):
        A_ub[2*i, :n] = A[i]
        A_ub[2*i, n] = -refs[i]
        b_ub[2*i] = refs[i]
        A_ub[2*i+1, :n] = -A[i]
        A_ub[2*i+1, n] = -refs[i]
        b_ub[2*i+1] = -refs[i]
    A_eq = np.ones((1, n + 1))
    A_eq[0, n] = 0.0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * (n + 1)
    try:
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method='highs')
        if r.success:
            w = r.x[:n]
            s = w.sum()
            if s > 1e-15:
                w /= s
            return w, r.x[n]
    except:
        pass
    return None, np.inf

def _build_points():
    z_vals = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3,
                       0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    w0_vals = np.array([-1.5, -1.0, -0.8, -0.5, -0.3, 0.0, 0.2, 0.5, 0.8, 1.0, 1.5])
    wa_vals = np.array([-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0])
    pts = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                pts.append((z, w0, wa))
    # Subsample to ~350
    if len(pts) > 350:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pts), 350, replace=False)
        pts = [pts[i] for i in idx]
    return pts

def _eval(nodes, pts, n=8):
    N = len(pts)
    A = np.zeros((N, n))
    refs = np.zeros(N)
    for i, (z, w0, wa) in enumerate(pts):
        if z < 1e-12:
            A[i] = 1.0
            refs[i] = 1.0
        else:
            A[i] = 1.0 / _E(z * nodes, w0, wa)
            refs[i] = _ref(z, w0, wa)
    w, t = _lp_weights(A, refs, n)
    return w, t

def _logit(x):
    x = np.clip(x, 1e-8, 1-1e-8)
    return np.log(x / (1-x))

def _expit(u):
    return 1.0 / (1.0 + np.exp(-u))

def construct():
    if time.time() > _DEADLINE - 3:
        gn, gw = roots_legendre(8)
        return {"nodes": (gn+1)/2.0, "weights": gw/2.0}
    
    n = 8
    pts = _build_points()
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    # Initial candidates
    cands = []
    gn, gw = roots_legendre(n)
    cands.append((gn+1)/2.0)
    
    # Gauss-Jacobi variants
    from scipy.special import roots_jacobi
    for a, b in [(0.5,0.5),(1.0,0.0),(0.0,1.0),(2.0,0.0),(0.0,2.0),(1.5,1.5),(2.5,0.0)]:
        try:
            jn, _ = roots_jacobi(n, a, b)
            jm = (jn+1)/2.0
            if np.all(np.diff(jm) > 1e-8):
                cands.append(jm)
        except:
            pass
    
    # Perturbed GL
    rng = np.random.RandomState(99)
    base = (gn+1)/2.0
    for _ in range(30):
        p = rng.uniform(-0.05, 0.05, n)
        c = np.sort(base + p)
        c = np.clip(c, 1e-4, 1-1e-4)
        if np.all(np.diff(c) > 1e-6):
            cands.append(c)
    
    # Power laws
    for pw in [0.7, 0.8, 1.2, 1.5, 2.0, 2.5]:
        t = np.linspace(0, 1, n)
        c = np.clip(t**pw, 1e-4, 1-1e-4)
        if np.all(np.diff(c) > 1e-6):
            cands.append(c)
    for pw in [0.7, 0.8, 1.2, 1.5, 2.0]:
        t = np.linspace(0, 1, n)
        c = np.clip(1-(1-t)**pw, 1e-4, 1-1e-4)
        if np.all(np.diff(c) > 1e-6):
            cands.append(c)
    
    # Evaluate candidates
    for c in cands:
        if time.time() > _DEADLINE - 10:
            break
        w, e = _eval(c, pts, n)
        if w is not None and e < best_err:
            best_err = e
            best_nodes = c.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        gn, gw = roots_legendre(n)
        return {"nodes": (gn+1)/2.0, "weights": gw/2.0}
    
    # Nelder-Mead refinement in logit space
    if time.time() < _DEADLINE - 12:
        def obj(u):
            nd = _expit(u)
            if np.any(nd <= 1e-5) or np.any(nd >= 1-1e-5):
                return 1e6
            if np.any(np.diff(nd) <= 1e-8):
                return 1e6
            w, e = _eval(nd, pts, n)
            if w is None:
                return 1e6
            return e
        
        u0 = _logit(best_nodes)
        try:
            res = minimize(obj, u0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            nd = _expit(res.x)
            if np.all(nd > 1e-4) and np.all(nd < 1-1e-4) and np.all(np.diff(nd) > 1e-7):
                w, e = _eval(nd, pts, n)
                if w is not None and e < best_err:
                    best_err = e
                    best_nodes = nd.copy()
                    best_weights = w.copy()
        except:
            pass
    
    # Multiple restarts
    if time.time() < _DEADLINE - 10:
        rng2 = np.random.RandomState(555)
        for _ in range(15):
            if time.time() > _DEADLINE - 8:
                break
            pert = rng2.normal(0, 0.04, n)
            u2 = _logit(best_nodes) + pert
            try:
                res2 = minimize(obj, u2, method='Nelder-Mead',
                               options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                nd = _expit(res2.x)
                if np.all(nd > 1e-4) and np.all(nd < 1-1e-4) and np.all(np.diff(nd) > 1e-7):
                    w, e = _eval(nd, pts, n)
                    if w is not None and e < best_err:
                        best_err = e
                        best_nodes = nd.copy()
                        best_weights = w.copy()
            except:
                pass
    
    # Final polish with larger test set
    if time.time() < _DEADLINE - 5:
        z_fine = np.array([0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0,3.0,4.0,5.0,7.0,10.0])
        w0_fine = np.array([-1.5,-1.0,-0.8,-0.5,-0.3,0.0,0.2,0.5,0.8,1.0,1.5])
        wa_fine = np.array([-1.0,-0.7,-0.5,-0.3,0.0,0.3,0.5,0.7,1.0])
        fine_pts = []
        for z in z_fine:
            for w0 in w0_fine:
                for wa in wa_fine:
                    fine_pts.append((z, w0, wa))
        if len(fine_pts) > 400:
            rng3 = np.random.RandomState(77)
            idx3 = rng3.choice(len(fine_pts), 400, replace=False)
            fine_pts = [fine_pts[i] for i in idx3]
        wf, ef = _eval(best_nodes, fine_pts, n)
        if wf is not None:
            best_weights = wf
            best_err = ef
    
    # Validate and return
    best_nodes = np.sort(best_nodes)
    if np.any(best_weights <= 0) or not np.all(np.diff(best_nodes) > 0):
        gn, gw = roots_legendre(n)
        return {"nodes": (gn+1)/2.0, "weights": gw/2.0}
    
    s = best_weights.sum()
    if abs(s - 1.0) > 1e-12:
        best_weights /= s
    
    return {"nodes": best_nodes, "weights": best_weights}
