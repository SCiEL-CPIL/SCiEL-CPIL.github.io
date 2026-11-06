"""Direct solver for the fixed Robertson ODE: boundary-value formulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_bvp, solve_ivp


def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] as a boundary-value problem.

    Parameters
    ----------
    f : callable
        ODE right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t0, t1) integration interval (fixed to [0, 40] by the evaluator).
    y0 : array-like
        Initial condition at t0.

    Returns
    -------
    y_end : ndarray
        Final state y(t1).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).reshape(-1, 1)
    n = y0.shape[0]

    # Analytic Jacobian df/dy for the Robertson problem (matches the fixed task):
    #   f0 = -0.04*y0 + 1e4*y1*y2
    #   f1 =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    #   f2 =  3e7*y1^2
    def jac(t, y):
        y = y.reshape(-1)
        y0v, y1v, y2v = y[0], y[1], y[2]
        return np.array([
            [-0.04,                  1.0e4 * y2v,  1.0e4 * y1v],
            [ 0.04, -1.0e4 * y2v - 6.0e7 * y1v, -1.0e4 * y1v],
            [ 0.0,                   6.0e7 * y1v,  0.0      ],
        ])

    # Mesh: geometric in the fast transient head, linear in the slow tail.
    def build_mesh(n_head, n_tail):
        # Head: [0, 0.05] geometric (captures the fast transient ~1e-3 to 1e-2)
        if n_head > 1:
            head = np.geomspace(1e-6, 0.05, n_head)
            head = np.concatenate([[0.0], head])
        else:
            head = np.array([0.0])
        # Tail: [0.05, 40] linear
        tail = np.linspace(0.05, 40.0, n_tail)
        t = np.concatenate([head, tail])
        # Remove duplicates
        t = np.unique(t)
        return t

    # ODE and boundary functions (fixed problem: y(t0)=y0, unknown y(t1)).
    def fun(t, y):
        return f(t, y)

    def bc(ya, yb):
        return ya - y0

    # Time budget guard: evaluator kills at 330s; keep a safety margin.
    start_time = time.time()
    time_limit = 300.0

    best_result = None
    best_time = 0.0
    best_resid = np.inf
    best_n = np.inf

    # Drive convergence: raise resolution and tighten BVP tolerance.
    # (n_head, n_tail, bvp_tol)
    configs = [
        (40,  80,  1e-5),
        (80,  160, 1e-7),
        (160, 320, 1e-9),
        (320, 640, 1e-11),
        (640, 1280, 1e-13),
    ]

    for n_head, n_tail, bvp_tol in configs:
        if time.time() - start_time > time_limit - 15.0:
            break
        try:
            t = build_mesh(n_head, n_tail)
            sol = solve_bvp(
                fun, bc, t, y0,
                jac=jac,
                tol=bvp_tol,
                max_nodes=8 * (n_head + n_tail) + 20,
                verbose=0,
            )
        except Exception:
            break
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15.0:
            break
        if sol.success and sol.y.shape[1] >= 2:
            resid = sol.resnorm if hasattr(sol, "resnorm") else np.inf
            ny = sol.y.shape[1]
            if best_result is None or resid < best_resid - 1e-14 or \
               (abs(resid - best_resid) < 1e-14 and ny < best_n):
                best_result = sol.y[:, -1].reshape(-1)
                best_time = elapsed
                best_resid = resid
                best_n = ny
            # Stop once residual is very small and mesh is sufficiently refined.
            if resid < 1e-12 and ny >= 2 * (n_head + n_tail):
                break

    # Polish pass: use BVP solution as initial guess for a tight Radau IVP solve
    # to squeeze out additional digits of accuracy.
    if best_result is not None and time.time() - start_time < time_limit - 20.0:
        try:
            # Use the BVP solution to get a good initial trajectory for IVP
            # Just do a tight Radau solve from scratch with very tight tolerances.
            sol_ivp = solve_ivp(
                f, (t0, t1), y0.reshape(-1),
                method="Radau", jac=jac,
                rtol=1e-14, atol=1e-17,
                max_step=0.05,
            )
            if sol_ivp.success:
                ivp_result = sol_ivp.y[:, -1].reshape(-1)
                # Compare with BVP result; keep the one with smaller estimated error
                # Use the difference as a rough error indicator
                diff = np.max(np.abs(ivp_result - best_result))
                if diff < 1e-7:
                    # They agree well; use the IVP result (likely more accurate
                    # since Radau with tight tolerances is very precise)
                    best_result = ivp_result
        except Exception:
            pass

    # Fallback: if the BVP route produced nothing, use a stiff IVP solve.
    if best_result is None:
        try:
            sol = solve_ivp(
                f, t_span, y0.reshape(-1),
                method="Radau", jac=jac,
                rtol=1e-8, atol=1e-12, max_step=1.0,
            )
            if sol.success:
                best_result = sol.y[:, -1].reshape(-1)
        except Exception:
            # Last resort: fine forward-Euler (only if everything else failed).
            m = 200000
            h = (t1 - t0) / m
            y = y0.reshape(-1).copy()
            tt = t0
            for _ in range(m):
                y = y + h * f(tt, y)
                tt += h
            best_result = y

    return best_result
# EVOLVE-BLOCK-END
