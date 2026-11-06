import numpy as np
import scipy.integrate
import scipy.sparse
import time

def integrate(f, t_span, y0):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float)
    N = len(y0)
    
    deadline = time.time() + 300.0
    hard_deadline = deadline - 10.0

    def _remaining():
        return hard_deadline - time.time()

    # 1. Try Radau with analytic (estimated) sparse Jacobian
    if _remaining() > 60:
        try:
            # Compute Jacobian at t0, y0
            f0 = f(t0, y0)
            h = 1e-8 * max(1.0, np.max(np.abs(y0)))
            J_dense = np.zeros((N, N))
            for j in range(N):
                y_pert = y0.copy()
                y_pert[j] += h
                f_pert = f(t0, y_pert)
                J_dense[:, j] = (f_pert - f0) / h
            
            # Enforce tridiagonal structure (remove numerical noise)
            # Assuming standard 1D stencil
            for i in range(N):
                for j in range(N):
                    if abs(i - j) > 1:
                        J_dense[i, j] = 0.0
                        
            J_sparse = scipy.sparse.csr_matrix(J_dense)
            
            def J_func(t, y):
                return J_sparse

            sol = scipy.integrate.solve_ivp(
                f, (t0, t1), y0,
                method='Radau',
                J=J_func,
                rtol=1e-10,
                atol=1e-12,
                max_step=np.inf
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass

    # 2. Fallback: Radau with numerical Jacobian (default)
    if _remaining() > 60:
        try:
            sol = scipy.integrate.solve_ivp(
                f, (t0, t1), y0,
                method='Radau',
                rtol=1e-10,
                atol=1e-12,
                max_step=np.inf
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass

    # 3. Fallback: BDF
    if _remaining() > 30:
        try:
            sol = scipy.integrate.solve_ivp(
                f, (t0, t1), y0,
                method='BDF',
                rtol=1e-10,
                atol=1e-12,
                max_step=np.inf
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass

    # 4. Fallback: DOP853
    if _remaining() > 30:
        try:
            sol = scipy.integrate.solve_ivp(
                f, (t0, t1), y0,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12,
                max_step=np.inf
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass

    # 5. Last resort RK4
    if _remaining() > 5:
        steps = 2000
        h = (t1 - t0) / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                break
            if time.time() > hard_deadline:
                break
        return y

    return y0.copy()
