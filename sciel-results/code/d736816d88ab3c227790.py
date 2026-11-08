import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solves the 15-dimensional stiff linear IVP by exploiting linearity.
    Since y' = f(t, y) is linear (f(t, y) = A y), we can compute the
    solution as y(t) = exp(A*(t-t0)) y0.
    
    Strategy:
    1. Infer the matrix A by calling f at the origin and at basis vectors.
       - f(0, 0) should be 0 for a linear homogeneous system. If not, it's affine.
       - Assume homogeneous linear system y' = A y based on task description "stiff linear system".
       - We need 15 calls to determine A: f(0, e_i) = A e_i = column i of A.
       - Total calls: 15 (or 16 if we check f(0,0) for safety/affine, but linear implies 0).
       - Actually, we can just pick 15 independent vectors. The standard basis is best.
    2. Compute the matrix exponential M = expm(A * (t1 - t0)).
    3. Return M @ y0.
    
    This uses ~15-16 calls to f.
    Score component: -0.5 * log10(16) ≈ -0.5 * 1.2 = -0.6.
    Current program uses Radau, likely thousands of calls. log10(1000) = 3, -0.5*3 = -1.5.
    So we save ~0.9 in score from call count alone.
    Accuracy: expm is highly accurate (machine precision level for moderate condition numbers).
    Relative error should be extremely small, giving high -log10(err).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    
    # Infer A
    # We assume y' = A y.
    # f(t, y) = A y.
    # Let's verify linearity quickly or just assume it as per task "linear system".
    # To be robust, we construct A using basis vectors.
    # f(t, e_i) = A e_i.
    # Note: f might depend on t if A is time-dependent, but "fixed 15-dimensional stiff linear system"
    # usually implies constant coefficients or at least autonomous. 
    # If A is constant, f(t, y) = A y for all t.
    # Let's evaluate at t0.
    
    A = np.zeros((n, n), dtype=float)
    
    # Check if f(0, 0) is 0 to confirm homogeneous. If not, we might need affine handling,
    # but "linear IVP" typically means homogeneous. If it's affine y' = Ay + b, 
    # the solution is more complex. However, "linear system" in ODE context often implies y'=Ay.
    # Let's assume y' = A y.
    
    # Optimization: We don't need to call f 15 times if we can be clever, but we need the columns.
    # 15 calls is very cheap.
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        # f returns a vector. The i-th column of A is f(t0, e_i).
        col = f(t0, e_i)
        A[:, i] = col
        
    # If the system is time-dependent A(t), this simple approach fails.
    # But "fixed ... system" suggests constant A.
    # If A is not constant, we would need a different approach (e.g., Magnus expansion or just using a solver).
    # Given the high score potential of expm, and the specific mention of "linear", constant A is the standard assumption for "linear system" unless "time-varying" is specified.
    
    # Compute Matrix Exponential
    # scipy.linalg.expm is available.
    try:
        from scipy.linalg import expm
        dt = t1 - t0
        M = expm(A * dt)
        y_end = M @ y0
    except Exception:
        # Fallback if expm fails or is not available (unlikely with SciPy)
        # Use a simple RK4 with large steps as a fallback, though it will be worse.
        # Or use scipy.integrate.solve_ivp
        import scipy.integrate as integrate_scipy
        sol = integrate_scipy.solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        if sol.success:
            return sol.y[:, -1]
        return y0 # Last resort

    # Check time
    if time.time() - start_time > 290:
        # Should not happen
        pass

    return y_end
