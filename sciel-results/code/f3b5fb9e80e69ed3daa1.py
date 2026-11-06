"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# We can import scipy if available, but we must handle its absence.
try:
    from scipy.sparse import csc_matrix, lil_matrix, eye as speye
    from scipy.sparse.linalg import spsolve, aslinearoperator, gmres, bicgstab, cg
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False

try:
    from scipy.linalg import expm, eigh, expm_frechet
    HAS_SCIPY_LINALG = True
except ImportError:
    HAS_SCIPY_LINALG = False

import math

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # 320s to leave 10s margin before 330s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Heuristic for 10x10 grid
    # If n == 100, it's likely a 2D heat equation on a 10x10 grid.
    # The RHS f(t, y) likely computes the Laplacian with some BCs.
    # For the 2D heat equation, the system is linear: y' = A y.
    # We can exploit this by estimating A and using exponential integrators or
    # implicit methods that solve linear systems efficiently.
    
    # Strategy:
    # 1. Detect linearity. If f is linear, estimate A.
    # 2. Use a high-order implicit Runge-Kutta method or exponential integrator.
    #    Given the stiffness of heat equations, implicit methods are best.
    #    Since it's linear, we can pre-factor the matrix (A - (1/dt)*I) once if coefficients are constant.
    #    However, the problem might have time-dependent coefficients or non-linear BCs, but "fixed 10x10 two-dimensional heat equation" usually implies constant coefficients.
    
    # Let's try to estimate A.
    # f(t, y) = A y + b(t) ? Or just A y?
    # Let's probe with zero vector and unit vectors.
    # But calling f 100 times just to estimate A is cheap compared to the integration steps.
    
    # However, if f is not linear, this fails.
    # The task says "fixed 10x10 two-dimensional heat equation", which is linear.
    # y' = D_xx y + D_yy y.
    
    # Let's assume f(t, y) = A y.
    # We can estimate A by calling f(t0, e_i) for basis vectors? No, that's 100 calls.
    # Or call f(t0, y0) and f(t0, 0)?
    # f(t0, 0) gives the bias term b. If b is 0, it's homogeneous.
    
    # Let's try a more robust approach:
    # Use an adaptive step size implicit method.
    # Since it's 10x10, n=100.
    # A direct LU decomposition is fast.
    # We can use BDF2 or similar.
    
    # But wait, the score penalizes calls to f.
    # If we use a linear multistep method or RK, we use multiple f calls per step.
    # If we use a method that only requires solving linear systems (implicit), we still need f calls to evaluate the RHS at new points.
    
    # Key Insight: If the system is linear and constant (A is constant), we can use the matrix exponential or a rational approximation (like Chebyshev or Padé) to integrate exactly or with high accuracy in few steps.
    # Or, we can use a Krylov subspace method with an exponential integrator.
    
    # Let's estimate A.
    # y' = A y
    # Let's pick a random vector v.
    # f(t0, v) approximates A v.
    # If we assume f is linear, f(t, y) = A y.
    
    # Let's test linearity.
    # y1 = y0
    # y2 = 2*y0
    # f1 = f(t0, y1)
    # f2 = f(t0, y2)
    # If f2 ≈ 2*f1, it's likely linear.
    
    f0 = f(t0, y0)
    f2 = f(t0, 2*y0)
    
    is_linear = np.allclose(f2, 2*f0, rtol=1e-10, atol=1e-12)
    
    if is_linear and n == 100:
        # We are in the linear 10x10 heat case.
        # Let's estimate A.
        # We can use a few random probes to build A, or just use a method that doesn't require explicit A if we can solve (I - dt*A)x = y efficiently.
        # But we don't have A explicitly.
        
        # Alternative: Use a high-order Runge-Kutta method with adaptive step size?
        # Heat equation is stiff. Explicit RK needs tiny steps.
        # Implicit methods are better.
        
        # Since it's linear, we can use the "Method of Lines" with a solver.
        # But we don't have the matrix A.
        
        # Can we estimate A efficiently?
        # If we call f(t, e_i) for all i, we get columns of A. That's 100 calls.
        # Then we can factorize (I - dt*A) once.
        # Then each step is a back-substitution.
        # Total calls: 100 (for A) + 1 (initial) + steps * 1 (for implicit Euler) or steps * k (for higher order).
        # If we use Implicit Euler (1st order), we need many steps for accuracy.
        # If we use a high-order A-stable method like BDF4 or Radau, we need fewer steps.
        
        # Let's estimate A.
        A = np.zeros((n, n))
        e = np.zeros(n)
        for i in range(n):
            e[i] = 1.0
            A[:, i] = f(t0, e)
            e[i] = 0.0
        
        # Check if A is correct by verifying A@y0 ≈ f0
        if not np.allclose(A @ y0, f0, rtol=1e-5, atol=1e-8):
            # Fallback to non-linear assumption
            return _integrate_nonlinear(f, t_span, y0)
        
        # Now we have A.
        # We want to solve y' = A y from t0 to t1.
        # The exact solution is y(t) = exp(A*(t-t0)) y0.
        # We can compute this using scipy.linalg.expm or expm_frechet.
        
        # Let's use the matrix exponential directly if scipy is available.
        # y_end = expm(A * (t1 - t0)) @ y0
        
        # But is `expm` accurate enough? Yes, it's very accurate.
        # How many f calls did we use?
        # 100 (for A) + 1 (initial check) + 1 (2*y0 check) = 102 calls.
        # The cost of expm is internal, doesn't count as f calls.
        # This should be extremely accurate (close to machine precision relative error).
        # Score = -log10(rel_err) - 0.5*log10(calls).
        # rel_err ~ 1e-15 -> -log10(1e-15) = 15.
        # calls = 102 -> -0.5*log10(102) ≈ -1.
        # Score ≈ 14.
        
        # Compare to seed:
        # Seed uses RK4 with 2000 steps. Calls = 2000*4 = 8000.
        # Accuracy of RK4 for stiff problem? RK4 is explicit, so it's unstable for stiff problems unless h is very small.
        # Heat equation with 10x10 grid: eigenvalues of A are around -40 (max).
        # Stability limit for RK4: h < 2.78 / lambda_max ≈ 2.78/40 ≈ 0.07.
        # t_span is 0.05. So h=0.05 is within stability limit?
        # Wait, lambda_max for 10x10 2D Laplacian (Dirichlet BC):
        # lambda_k,l = -4 (sin^2(k pi/11) + sin^2(l pi/11)).
        # Max lambda is for k=10, l=10: -4 (sin^2(10 pi/11) + sin^2(10 pi/11)) = -8 sin^2(10 pi/11).
        # sin(10 pi/11) = sin(pi/11) ≈ 0.28.
        # lambda_max ≈ -8 * 0.08 = -0.64?
        # Wait, grid spacing h_grid. If domain is [0,1]x[0,1], h_grid = 1/10.
        # Laplacian eigenvalues: -4/h_grid^2 (sin^2... + sin^2...).
        # h_grid^2 = 0.01. 4/0.01 = 400.
        # lambda_max ≈ -400 * (2 * 0.08) = -64.
        # Stability limit h < 2.78 / 64 ≈ 0.043.
        # t_span = 0.05.
        # Seed uses 2000 steps -> h = 0.05/2000 = 2.5e-5. Very stable.
        # Accuracy of RK4 with h=2.5e-5 is extremely high.
        # So seed is likely very accurate.
        # Calls = 8000.
        
        # My approach: 102 calls.
        # Accuracy: expm is ~1e-15.
        # Score: 15 - 1 = 14.
        # Seed Score: 15 - 0.5*log10(8000) = 15 - 1.75 = 13.25.
        # So my approach should be better.
        
        # Implementation:
        dt = t1 - t0
        
        # Check time remaining
        if _time_remaining() < 10.0:
            # Not enough time, use a quick fallback
            return y0 + f0 * dt  # Euler step as last resort
        
        try:
            from scipy.linalg import expm_frechet
            # Use expm_frechet for better accuracy: computes expm(A*dt) @ y0 directly
            y_end, _ = expm_frechet(A * dt, y0)
            return y_end
        except (ImportError, Exception):
            try:
                from scipy.linalg import expm
                M = expm(A * dt)
                y_end = M @ y0
                return y_end
            except (ImportError, Exception):
                # Fallback: Use Taylor series for expm
                I = np.eye(n)
                M = np.zeros((n, n))
                term = I.copy()
                A_dt = A * dt
                for k in range(1, 50):
                    term = term @ A_dt / k
                    M = M + term
                    if np.linalg.norm(term) < 1e-16 * np.linalg.norm(M):
                        break
                y_end = M @ y0
                return y_end

    else:
        return _integrate_nonlinear(f, t_span, y0)


def _integrate_nonlinear(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    dt_total = t1 - t0
    
    # Try scipy Radau5 for stiff problems
    if HAS_SCIPY_LINALG:
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=dt_total/100)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: Simple RK4 with fixed steps
    n_steps = 2000
    h = dt_total / n_steps
    t = t0
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y

# EVOLVE-BLOCK-END
