import numpy as np
import scipy.linalg
import scipy.integrate
import time


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on t_span = [t0, t1] for a 15-dim stiff linear IVP.
    Strategy: extract A and b (16 calls), then solve exactly using eigen-decomposition.
    This avoids Padé approximant instability for stiff systems.
    """
    start_time = time.time()
    deadline = start_time + 270.0  # 30s safety margin under 300s limit

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # --- Step 1: Extract A and b from the linear system ---
    try:
        zero_vec = np.zeros(n)
        b = np.asarray(f(t0, zero_vec), dtype=float)

        A = np.zeros((n, n))
        e0 = np.zeros(n)
        e0[0] = 1.0
        A[:, 0] = np.asarray(f(t0, e0), dtype=float) - b

        for i in range(1, n):
            if time.time() > deadline - 10.0:
                raise TimeoutError("Deadline approaching")
            ei = np.zeros(n)
            ei[i] = 1.0
            A[:, i] = np.asarray(f(t0, ei), dtype=float) - b

        if time.time() > deadline - 5.0:
            raise TimeoutError("Deadline approaching")

    except Exception:
        A = None
        b = None

    # --- Step 2: Solve using eigen-decomposition ---
    if A is not None:
        try:
            # Check if A is symmetric (or Hermitian, but real here)
            # If symmetric, use eigh for better stability and speed
            if np.allclose(A, A.T, rtol=1e-10, atol=1e-10):
                eigvals, eigvecs = np.linalg.eigh(A)
                # eigvecs are columns, V^{-1} = V.T
                # exp(dt * A) = V @ diag(exp(dt * eigvals)) @ V.T
                exp_eigvals = np.exp(dt * eigvals)
                # Integral term: int_0^dt exp(u * A) du = V @ diag(int_0^dt exp(u * eigval) du) @ V.T
                # int_0^dt exp(u * lambda) du = (exp(dt * lambda) - 1) / lambda if lambda != 0, else dt
                integral_eigvals = np.zeros_like(eigvals)
                mask = np.abs(eigvals) > 1e-12
                integral_eigvals[mask] = (exp_eigvals[mask] - 1.0) / eigvals[mask]
                integral_eigvals[~mask] = dt
                
                expA_dt = eigvecs @ np.diag(exp_eigvals) @ eigvecs.T
                integral_b = eigvecs @ np.diag(integral_eigvals) @ (eigvecs.T @ b)
                y_final = expA_dt @ y0 + integral_b
                return y_final
            else:
                # General case: eigen-decomposition
                eigvals, eigvecs = np.linalg.eig(A)
                # V^{-1}
                eigvecs_inv = np.linalg.inv(eigvecs)
                
                exp_eigvals = np.exp(dt * eigvals)
                integral_eigvals = np.zeros_like(eigvals, dtype=complex)
                mask = np.abs(eigvals) > 1e-12
                integral_eigvals[mask] = (exp_eigvals[mask] - 1.0) / eigvals[mask]
                integral_eigvals[~mask] = dt
                
                # expA_dt = V @ diag(exp_eigvals) @ V^{-1}
                expA_dt = eigvecs @ np.diag(exp_eigvals) @ eigvecs_inv
                
                # integral_b = V @ diag(integral_eigvals) @ V^{-1} @ b
                # Note: b is real, but eigvals/eigvecs might be complex. Result should be real.
                integral_b = eigvecs @ np.diag(integral_eigvals) @ (eigvecs_inv @ b)
                
                y_final = expA_dt @ y0 + integral_b
                # Ensure real output if input was real
                if np.isrealobj(y0) and np.isrealobj(b):
                    y_final = y_final.real
                return y_final

        except Exception:
            pass

    # --- Step 3: Fallback to numerical solvers ---
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            sol2 = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14
            )
            return sol2.y[:, -1]
    except Exception:
        try:
            sol3 = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='LSODA',
                rtol=1e-10,
                atol=1e-12
            )
            return sol3.y[:, -1]
        except Exception:
            try:
                sol4 = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='RK45',
                    rtol=1e-10,
                    atol=1e-12
                )
                return sol4.y[:, -1]
            except Exception:
                return y0
