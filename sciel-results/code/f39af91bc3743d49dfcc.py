"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1].
    Strategy:
    1. Detect if the system is linear time-invariant (y' = A*y).
    2. If so, compute the matrix exponential expm((t1-t0)*A) and apply it to y0.
       This requires N+1 calls to f (to probe A) and provides near-machine-precision accuracy.
    3. If not linear or detection fails, fall back to a high-order adaptive method (RK45/RK23 hybrid or just high-step RK4).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin

    # --- Step 1: Probe for Linearity ---
    # We assume y' = A * y + b.
    # For heat equation, b=0.
    # Probe f(t, 0) to check for b.
    y_zero = np.zeros_like(y0)
    try:
        f_zero = f(t0, y_zero)
    except Exception:
        f_zero = None

    # If f_zero is not close to zero, it might be affine or non-linear.
    # Let's check linearity by probing basis vectors.
    # We need to be careful: if f is non-linear, this will fail or produce wrong results.
    # A robust check: Compute A from basis vectors, then verify f(t, y_test) ≈ A*y_test + b for a random y_test.
    
    is_linear = True
    A = None
    b = None
    
    if f_zero is not None:
        b = f_zero
        # Probe columns of A
        # Use a small perturbation to avoid numerical issues if f is sensitive, 
        # but for linear systems, exact basis probing is fine.
        A = np.zeros((n, n), dtype=float)
        
        # To save time, we can try to infer A from the structure if n is large, 
        # but n=100 is small enough to probe directly.
        # However, probing 100 columns takes 100 calls.
        # Total calls: 1 (for b) + 100 (for A) + 1 (verification) = 102 calls.
        
        for j in range(n):
            if time.time() > deadline:
                is_linear = False
                break
            
            e_j = np.zeros_like(y0)
            e_j[j] = 1.0
            try:
                f_j = f(t0, e_j)
            except Exception:
                is_linear = False
                break
            
            A[:, j] = f_j - b
        
        # Verify linearity
        if is_linear:
            # Pick a random test vector
            rng = np.random.default_rng(42)
            y_test = rng.standard_normal(n)
            try:
                f_test = f(t0, y_test)
            except Exception:
                is_linear = False
                f_test = None
            
            if f_test is not None:
                predicted = A @ y_test + b
                # Check relative error
                norm_pred = np.linalg.norm(predicted)
                norm_diff = np.linalg.norm(f_test - predicted)
                if norm_pred < 1e-15:
                    if norm_diff > 1e-10:
                        is_linear = False
                else:
                    if norm_diff / norm_pred > 1e-10:
                        is_linear = False
            else:
                is_linear = False

    # --- Step 2: Solve ---
    if is_linear:
        # Use scipy.linalg.expm for high accuracy
        # Note: expm is accurate for moderate matrix sizes. 100x100 is fine.
        try:
            from scipy.linalg import expm
            dt = t1 - t0
            if dt == 0:
                return y0.copy()
            
            # Compute expm(dt * A)
            # For large dt or stiff A, expm might be slow or inaccurate?
            # expm uses Padé approximation + scaling/squaring. It's generally very accurate.
            expA = expm(dt * A)
            
            # y(t1) = exp(dt*A) * y0 + integral of exp((t1-s)A) b ds
            # If b is constant (time-invariant affine), the integral is (exp(dt*A) - I) * A^{-1} * b
            # But for heat equation, b=0 usually.
            # Let's handle b=0 first. If b is not zero, we can integrate the affine term.
            
            if np.allclose(b, 0, atol=1e-15):
                y_end = expA @ y0
            else:
                # Affine case: y' = A y + b
                # Solution: y(t) = exp((t-t0)A) y0 + ∫_{t0}^{t} exp((t-s)A) b ds
                # ∫_{0}^{dt} exp(τ A) b dτ = (exp(dt A) - I) A^{-1} b  (if A invertible)
                # Or more robustly: solve (exp(dt A) - I) v = dt * A^{-1} ... no.
                # Let's just use a high-order ODE solver for the affine case if b != 0, 
                # or assume b=0 for this specific task (Heat Equation).
                # The task says "Heat2D 10x10", which implies y' = A y with Dirichlet/Neumann BCs handled in A.
                # So b should be 0.
                # If b is not zero, we'll fall back to numerical integration to be safe.
                y_end = None 
                # Fallback for affine: use RK4 with many steps? 
                # Or just use scipy solve_ivp with method='Radau' or 'BDF'?
                # But we want to avoid extra calls if possible.
                # Let's assume b=0 for the heat equation. If this fails, the verification step would have caught non-linearity, 
                # but affine is linear in y. 
                # Actually, affine systems are solvable analytically.
                # Let's implement the affine solution properly.
                try:
                    # Compute integral term: I = ∫_0^dt exp(τ A) dτ
                    # I = (exp(dt A) - I) A^{-1}
                    # A might be singular. 
                    # Alternative: I = (exp(dt A) - I) / A element-wise? No.
                    # Use: I = (exp(dt A) - I) @ inv(A)
                    # If A is singular, use pseudo-inverse or series expansion.
                    # For heat equation, A is negative definite (or non-positive), so invertible unless 0 eigenvalue.
                    # Let's try:
                    I_mat = np.linalg.solve(A, expA - np.eye(n))
                    # Wait, (exp(dt A) - I) A^{-1} b = A^{-1} (exp(dt A) - I) b
                    # So I_b = A^{-1} (expA - I) b
                    I_b = np.linalg.solve(A, (expA - np.eye(n)) @ b)
                    y_end = expA @ y0 + I_b
                except np.linalg.LinAlgError:
                    # A is singular. Use scipy solve_ivp?
                    # For safety, fallback to numerical integration
                    y_end = None

            if y_end is not None:
                return y_end

        except Exception as e:
            # expm failed or other error
            pass

    # --- Step 3: Fallback to Numerical Integration ---
    # If linear detection failed or expm failed, use a robust ODE solver.
    # Since we don't know if it's stiff, we can try RK45 (adaptive) or Radau.
    # scipy.integrate.solve_ivp is available.
    # We need to count calls? No, the evaluator counts calls to f.
    # solve_ivp will call f.
    
    try:
        from scipy.integrate import solve_ivp
        
        # Use Radau for stiff problems, or BDF.
        # Heat equation is stiff. Radau is a good choice.
        # rtol and atol should be tight to maximize score.
        # Score: -log10(err) - 0.5*log10(calls)
        # We want low err and low calls.
        
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14, 
            max_step=0.01 # Prevent too large steps
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Try Radau with larger tolerance? Or BDF?
            sol2 = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12
            )
            if sol2.success:
                return sol2.y[:, -1]
            
    except Exception as e:
        pass

    # --- Step 4: Last Resort: Manual RK4 ---
    # If scipy fails or is not available (though it is), do manual RK4.
    steps = 2000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
            
    return y

# EVOLVE-BLOCK-END
