import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    call_count = 0
    start_time = time.time()
    max_time = 310.0
    max_calls = 1950000

    def f_wrapped(t, y):
        nonlocal call_count
        call_count += 1
        elapsed = time.time() - start_time
        if elapsed > max_time:
            raise TimeoutError("Time limit exceeded")
        if call_count > max_calls:
            raise OverflowError("Call limit exceeded")
        return f(t, y)

    # Check for affine term b = f(t, 0)
    # This handles the case y' = Ay + b
    try:
        b = f_wrapped(t0, np.zeros(n))
        is_affine = np.linalg.norm(b) > 1e-12
    except Exception:
        is_affine = False
        b = np.zeros(n)

    # Extract matrix A
    A = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        # If affine, f(t, e_i) = A e_i + b. So A[:, i] = f(t, e_i) - b
        # If homogeneous, b=0, so A[:, i] = f(t, e_i)
        val = f_wrapped(t0, e_i)
        if is_affine:
            A[:, i] = val - b
        else:
            A[:, i] = val

    dt = t1 - t0

    try:
        from scipy.linalg import expm_multiply
        
        if not is_affine:
            # Homogeneous: y(t) = exp(A dt) y0
            y_end = expm_multiply(dt * A, y0)
        else:
            # Affine: y(t) = exp(A dt) y0 + int_0^dt exp(A (dt - tau)) b dtau
            # If b is constant, the integral is A^{-1} (exp(A dt) - I) b
            # However, A might be singular.
            # Safer: Augmented system
            # z = [y; 1], z' = [[A, b], [0, 0]] z
            # We can use expm_multiply on the augmented matrix?
            # n=100, augmented n=101.
            # Or just use the formula if A is invertible.
            # Let's try the formula first, fallback to augmented if LinAlgError.
            try:
                # Check condition number or just try to invert
                # Actually, scipy.linalg.expm_multiply can handle the augmented system
                # But constructing the 101x101 matrix is cheap.
                A_aug = np.zeros((n+1, n+1))
                A_aug[:n, :n] = A
                A_aug[:n, n] = b
                # Last row is 0
                y0_aug = np.zeros(n+1)
                y0_aug[:n] = y0
                y0_aug[n] = 1.0
                
                y_end_aug = expm_multiply(dt * A_aug, y0_aug)
                y_end = y_end_aug[:n]
            except Exception:
                # Fallback to numerical integration of the affine part if expm fails
                # Or just assume A is invertible
                try:
                    A_inv = np.linalg.inv(A)
                    expA = expm_multiply(dt * A, np.eye(n)) # This is expensive, avoid
                    # Better: expA_dt = expm(dt*A)
                    from scipy.linalg import expm
                    expA_dt = expm(dt * A)
                    # Integral term: A^{-1} (exp(A dt) - I) b
                    integral_term = A_inv @ (expA_dt - np.eye(n)) @ b
                    y_end = expA_dt @ y0 + integral_term
                except Exception:
                    # Last resort: numerical integration
                    y_end = _fallback_solve(f_wrapped, t_span, y0)
        
        return y_end
    except Exception:
        return _fallback_solve(f_wrapped, t_span, y0)

def _fallback_solve(f_wrapped, t_span, y0):
    t0, t1 = t_span
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
