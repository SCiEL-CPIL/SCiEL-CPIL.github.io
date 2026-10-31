import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 142  # safety margin before 150s hard limit

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def fast_metric(h):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)

    def circulant_peak(h):
        """Peak of circular autocorrelation, normalized. Lower bound on linear peak."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        H = np.fft.rfft(h, n=N)
        return np.max(np.abs(H)**2) / (s * s)

    # Precompute DFT basis (real) for the N-point DFT
    # H(k) = sum_n h(n) exp(-2πi kn/N), k = 0..N/2
    # We'll build a matrix B where H = B @ h
    kmax = N // 2
    kk = np.arange(kmax + 1)
    nn = np.arange(N)
    # B[k, n] = exp(-2πi kn/N); H(k) = B[k,:] @ h
    phase = np.exp(-2j * np.pi * np.outer(kk, nn) / N)
    B = phase  # shape (kmax+1, N)
    # DC row
    B[0, :] = 1.0

    # Phase 1: Solve circulant peak minimization via log-convex bisection on t.
    # Minimize t s.t. |H(k)|^2 <= t * s^2 for all k, s = sum(h) >= 1, h >= 0.
    # By homogeneity, fix s = 1 (scale invariant). Then: minimize t s.t. |H(k)|^2 <= t, sum h = 1, h >= 0.
    # This is convex in (h, t). Use CVX-style: minimize t + 0*||h||^2 s.t. |H(k)|^2 <= t, 1^T h = 1, h >= 0.
    # We solve via bisection on t: for given t, check feasibility of |H(k)|^2 <= t, sum h = 1, h>=0.
    # Feasibility: minimize max_k (|H(k)|^2 - t) subject to sum h = 1, h >= 0.
    # Actually simpler: for given t, the feasible set is {h: |H(k)|^2 <= t ∀k, 1^T h = 1, h >= 0}.
    # Nonempty iff min_{1^T h=1, h>=0} max_k |H(k)|^2 <= t.
    # The inner min is our objective; bisection on t with a feasibility check.

    # Feasibility check for given t: does there exist h >= 0, sum h = 1, with |H(k)|^2 <= t ∀k?
    # This is: min over h of max_k |H(k)|^2 s.t. 1^T h = 1, h >= 0 <= t.
    # Equivalently: solve min ||h||_1 s.t. |H(k)|^2 <= t, h >= 0, and check min ||h||_1 <= 1.
    # (If we can achieve sum h <= 1 with the peak constraint, then by scaling up we get sum h = 1 with same peak.)

    def feasible(t):
        """Check if exists h >= 0 with |H(k)|^2 <= t and sum(h) <= 1."""
        # Minimize sum(h) subject to |H(k)|^2 <= t, h >= 0.
        # |H(k)|^2 = (Re H(k))^2 + (Im H(k))^2 <= t  <=>  ||[Re H(k), Im H(k)]||_2 <= sqrt(t)
        # Re H(k) = Re(B[k,:]) @ h, Im H(k) = -Im(B[k,:]) @ h
        # So constraints: ||A_k @ h||_2 <= sqrt(t) for each k, where A_k = [Re B[k]; -Im B[k]] (2 x N).
        # Minimize 1^T h s.t. ||A_k h||_2 <= sqrt(t) ∀k, h >= 0.
        # This is a second-order program; solve via scipy minimize (SLSQP) or via dual.
        # Use SLSQP on h.
        h0 = np.full(N, 1.0 / N)
        sqrt_t = np.sqrt(t)
        def obj(h):
            return np.sum(h)
        def obj_grad(h):
            return np.ones(N)
        cons = []
        for k in range(kmax + 1):
            Ak = np.array([np.real(B[k, :]), -np.imag(B[k, :])])  # 2 x N
            def con_k(h, Ak=Ak):
                return sqrt_t**2 - np.dot(Ak @ h, Ak @ h)
            cons.append({'type': 'ineq', 'fun': con_k})
        bounds = [(0, None)] * N
        try:
            res = minimize(obj, h0, jac=obj_grad, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 80, 'ftol': 1e-10})
            if res.success or res.fun <= 1.0 + 1e-8:
                return True
            return res.fun <= 1.0 + 1e-8
        except Exception:
            return False

    # Bisection on t
    t_lo = 0.0
    # Upper bound: constant h=1/N gives peak = N^2 * (1/N)^2 ... let's compute
    h_const = np.full(N, 1.0/N)
    t_hi = circulant_peak(h_const)  # should be ~1
    # Actually for constant h, H(0)=1, H(k)=0 for k>0, so peak = 1. t_hi = 1.
    t_hi = max(t_hi, 1.0)
    # Lower bound: theoretical min is 1 (by Parseval, sum |H(k)|^2 = N*sum h(n)^2 >= (sum h)^2/N,
    # so max |H(k)|^2 >= (sum h)^2/N, and normalized peak >= 1/N * N = 1... wait let me recompute)
    # Parseval: sum_k |H(k)|^2 = N * sum_n h(n)^2. With sum h = 1, sum h^2 >= 1/N (by Cauchy).
    # So sum_k |H(k)|^2 >= N/N = 1. Thus max_k |H(k)|^2 >= 1/(kmax+1) = 1/(N/2+1).
    # Normalized: peak >= 1/(N/2+1) ≈ 1/301 ≈ 0.0033.
    t_lo = 1.0 / (kmax + 1)
    # Bisection
    t_star = t_hi
    n_bisect = 30
    for it in range(n_bisect):
        if elapsed() > DEADLINE - 60:
            break
        t_mid = 0.5 * (t_lo + t_hi)
        if feasible(t_mid):
            t_star = t_mid
            t_hi = t_mid
        else:
            t_lo = t_mid

    # Now solve the actual optimization: min t s.t. |H(k)|^2 <= t, sum h = 1, h >= 0
    # Use the t_star as a target and solve for h that achieves peak close to t_star.
    # Minimize max_k (|H(k)|^2 - t_star) + penalty, or just minimize max |H(k)|^2 with sum h = 1.
    # Use SLSQP: minimize t + mu*0 s.t. |H(k)|^2 <= t, sum h = 1, h >= 0.
    # Variables: (h, t). 
    def circ_opt():
        # Minimize t s.t. |H(k)|^2 <= t ∀k, sum h = 1, h >= 0
        # Use a smooth approach: minimize t + sum_k softplus(|H(k)|^2 - t)
        # Or SLSQP directly.
        h0 = np.full(N, 1.0/N)
        t0 = circulant_peak(h0)
        x0 = np.concatenate([h0, [t0]])
        def obj(x):
            h = x[:N]
            t = x[N]
            H = B @ h
            return t
        def obj_grad(x):
            h = x[:N]
            t = x[N]
            H = B @ h
            # grad_t = 1
            # grad_h: 0 (t doesn't depend on h in objective, only in constraints)
            g = np.zeros(N + 1)
            g[N] = 1.0
            return g
        cons = []
        for k in range(kmax + 1):
            Bk = B[k, :]
            def con_k(x, Bk=Bk):
                h = x[:N]
                t = x[N]
                Hk = Bk @ h
                return t - np.abs(Hk)**2
            cons.append({'type': 'ineq', 'fun': con_k})
        cons.append({'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - 1.0})
        bounds = [(0, None)] * N + [(0, None)]
        try:
            res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < 1e9:
                h_opt = np.clip(res.x[:N], 0, None)
                s = np.sum(h_opt)
                if s > 1e-10:
                    h_opt = h_opt / s  # normalize to sum 1 (scale invariant)
                return h_opt, res.x[N]
        except Exception:
            pass
        return None, None

    h_circ, t_circ = circ_opt()
    if h_circ is not None and np.sum(h_circ) > 1e-10:
        # Normalize: the metric is scale-invariant, so scale to convenient magnitude
        h_circ = h_circ / np.sum(h_circ) * N  # scale so mean ~ 1
        v_circ = fast_metric(h_circ)
        best_h = h_circ.copy()
        best_val = v_circ
    else:
        # Fallback: constant
        best_h = np.ones(N)
        best_val = fast_metric(best_h)

    # Phase 2: Local polish on linear peak via coordinate descent
    # The circulant solution may not be optimal for linear peak, but it's a great starting point.
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(42)

    no_improve = 0
    max_no_improve = 8000
    mean_h = np.mean(current_h)

    while elapsed() < DEADLINE - 40 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        mean_h = np.mean(current_h)
        step_factors = np.array([0.02, 0.05, 0.1, 0.2, 0.5])
        step_sizes = step_factors * mean_h
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])

        best_local_val = current_val
        best_local_v = orig
        for cv in candidates_val:
            if abs(cv - orig) < 1e-13:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv

        if best_local_val < current_val - 1e-13:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1

        if no_improve % 500 == 0:
            # Perturbation to escape local minima
            pert = current_h.copy()
            pert += 0.03 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = fast_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()

    # Phase 3: Fine polish with tiny moves
    if elapsed() < DEADLINE - 10:
        step_frac = 0.001
        for _ in range(3000):
            if elapsed() > DEADLINE - 5:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h

    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return best_h.tolist()
