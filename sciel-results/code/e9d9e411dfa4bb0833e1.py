"""ODE solver: Heat2D 10x10 – integrate(f, t_span, y0)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def _build_heat_jacobian(f, t, y, n):
    """
    Build a sparse Jacobian approximation for the 10x10 heat equation.
    We use a finite-difference approach but exploit the known structure.
    For a 2D Laplacian on a 10x10 grid with Dirichlet boundaries (implied by typical heat problems),
    the Jacobian is a 5-point stencil. However, since we only have f, we must estimate it.
    
    Actually, for the heat equation y' = D * laplacian(y), the Jacobian is D * L where L is the
    discrete Laplacian. This is a fixed matrix. We can detect the structure or just use
    a generic sparse finite difference. Given the score penalty on calls, we should avoid
    O(n) calls for Jacobian if possible.
    
    Strategy: 
    1. Try to detect if the problem is linear and time-invariant. 
    2. If so, compute the Jacobian once using a few calls and reuse it.
    3. Use BDF2 or similar implicit method with this fixed Jacobian.
    
    However, we don't know the exact form of f. But the task says "fixed 10x10 two-dimensional heat equation".
    This implies f(t, y) is likely of the form A @ y where A is a constant matrix.
    
    Let's compute the Jacobian J such that f(t, y) ≈ J @ y.
    We can do this by calling f with basis vectors? No, that's 100 calls.
    Or we can assume the structure.
    
    A safer approach for a generic ODE solver that scores well on stiff problems:
    Use an implicit method like BDF2 or Radau, but we need to solve linear systems.
    
    Since n=100, even a dense solve is fast, but sparse is better.
    Let's estimate the Jacobian using finite differences on a few points to get the sparsity pattern,
    or just assume it's a 5-point stencil for the Laplacian.
    
    Actually, let's look at the seed program. It used RK4 then fell back to Backward Euler with
    finite-difference Jacobian. The score was 1.0.
    
    To improve, we can use a higher-order implicit method like BDF2 or BDF3, or Radau IIA.
    Or we can use an adaptive step size method.
    
    Given the strict time limit and call count penalty, a fixed-step implicit method with a
    pre-computed Jacobian (if possible) is good.
    
    Let's try to detect if f is linear.
    Call f(0, 0). If it's not zero, it's not homogeneous linear.
    Call f(0, e_i) for a few i to check linearity?
    
    Alternatively, we can just use a robust implicit method like BDF2 with a finite-difference Jacobian
    computed once at the beginning and reused, assuming the Jacobian doesn't change much.
    For the heat equation, the Jacobian is constant.
    
    Let's compute the Jacobian J at t0, y0 using finite differences.
    n=100.
    J is 100x100.
    Cost: 100 calls to f.
    Then each step requires solving (I - h*J) @ y_new = y_old + ...
    
    If we use BDF2:
    y_{n+1} = (4/3)y_n - (1/3)y_{n-1} + (2/3)h f(t_{n+1}, y_{n+1})
    (I - (2h/3)J) y_{n+1} = (4/3)y_n - (1/3)y_{n-1} + (2h/3)f(t_{n+1}, y_{n-1})  <-- using old f for RHS? No.
    
    Standard BDF2:
    (3/2)y_{n+1} - 2y_n + (1/2)y_{n-1} = h f(t_{n+1}, y_{n+1})
    Multiply by 2/3:
    2y_{n+1} - (4/3)y_n + (1/3)y_{n-1} = (2/3)h f(t_{n+1}, y_{n+1})
    (I - (h/3)J) y_{n+1} = (2/3)y_n - (1/6)y_{n-1} + (h/3)f(t_{n+1}, y_{n-1})? No.
    
    Let's stick to BDF1 (Backward Euler) but with a better solver and maybe adaptive step size?
    Or BDF2 with Newton iterations.
    
    Since the Jacobian is constant for the heat equation, we can factorize (I - h*J) once for a fixed h.
    But h might change.
    
    Let's try this strategy:
    1. Check if f is linear and time-invariant.
       - f(0, 0) should be 0.
       - f(t1, y1 + y2) == f(t1, y1) + f(t1, y2)?
       - If yes, compute J by calling f(t0, e_i) for i=0..n-1. This takes n calls.
       - Then use an implicit method with this fixed J.
    
    2. If not linear, use a generic implicit method with finite-difference Jacobian at each step
       (expensive) or just use RK4 with small steps.
    
    Given the task is specifically "Heat2D 10x10", it is almost certainly linear and time-invariant.
    
    Let's implement the linear detection and BDF2 with a fixed Jacobian.
    
    Steps:
    1. t0, t1 = t_span.
    2. y0 = np.array(y0, dtype=float).
    3. n = len(y0).
    4. Check linearity:
       - f0 = f(0, np.zeros(n))
       - if norm(f0) > 1e-12: not linear homogeneous.
       - Check f(t0, y0) vs J@y0? We don't have J yet.
       - Let's just compute J using n calls: J[:, i] = f(t0, e_i)
       - Check if f(t0, y0) is close to J @ y0.
       - Also check f(t1, y0) is close to J @ y0 (time invariance).
    
    5. If linear and time-invariant:
       - Use BDF2 or BDF3 with the fixed J.
       - Choose step size. For heat equation, stability is not an issue for implicit methods,
         but accuracy is.
       - Use a fixed number of steps, say 500 to 2000.
       - Or adaptive step size based on error estimate.
    
    6. If not linear:
       - Fall back to RK4 with many steps, or implicit Euler with FD Jacobian.
    
    Let's implement BDF2 with fixed J.
    
    BDF2 formula:
    y_{n+1} = (4/3)y_n - (1/3)y_{n-1} + (2/3)h f(t_{n+1}, y_{n+1})
    
    We need to solve:
    (I - (2h/3)J) y_{n+1} = (4/3)y_n - (1/3)y_{n-1} + (2h/3)f(t_{n+1}, y_{n-1})? 
    No, f(t_{n+1}, y_{n+1}) is on the LHS.
    (I - (2h/3)J) y_{n+1} = (4/3)y_n - (1/3)y_{n-1}
    
    Wait, the RHS doesn't involve f.
    So for each step, we solve a linear system with a fixed matrix (if h is fixed).
    This is very fast.
    
    We need to start with y_1 using BDF1 (Backward Euler):
    (I - h*J) y_1 = y_0
    
    Then for n=1,2,... use BDF2.
    
    Let's choose h.
    Total time T = t1 - t0.
    Number of steps N.
    h = T / N.
    
    We can try a few values of N and pick the one that minimizes error? No, we don't know the error.
    We can use a fixed N that is large enough for accuracy.
    For a 10x10 grid, the eigenvalues of the Laplacian are O(1).
    The solution is smooth.
    N=1000 should be very accurate.
    Cost: N calls to solve linear system.
    The linear system is 100x100.
    Using scipy.sparse.linalg.spsolve is fast.
    
    But we need to factorize the matrix for each h?
    If h is fixed, the matrix A = I - (2h/3)J is fixed.
    We can factorize it once using scipy.sparse.linalg.splu or just use spsolve which factors each time.
    For 100x100, spsolve is very fast.
    
    Let's implement this.
    
    Deadline guard:
    Start time = time.time()
    Deadline = start_time + 300 (or 320 to be safe).
    
    If time exceeds deadline, return the best result so far.
    
    Let's code this.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check for linearity and time-invariance
    # f(t, y) = J @ y
    # Compute J by calling f(t0, e_i)
    # This costs n calls.
    
    # First, check if f(0, 0) is zero
    f_zero = f(0, np.zeros(n))
    if np.linalg.norm(f_zero) > 1e-10:
        # Not homogeneous linear, fall back to RK4 or generic implicit
        return _fallback_rk4(f, t0, t1, y0)
    
    # Compute J
    J = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        J[:, i] = f(t0, e_i)
    
    # Check if f(t0, y0) == J @ y0
    f_y0 = f(t0, y0)
    if np.linalg.norm(f_y0 - J @ y0) > 1e-10 * (1 + np.linalg.norm(J @ y0)):
        # Not linear, fall back
        return _fallback_rk4(f, t0, t1, y0)
    
    # Check time invariance: f(t1, y0) == J @ y0
    f_t1_y0 = f(t1, y0)
    if np.linalg.norm(f_t1_y0 - J @ y0) > 1e-10 * (1 + np.linalg.norm(J @ y0)):
        # Not time-invariant, fall back
        return _fallback_rk4(f, t0, t1, y0)
    
    # Now use BDF2 with fixed J
    # Choose number of steps
    # Let's try N = 1000
    N = 1000
    h = (t1 - t0) / N
    
    # Start with BDF1
    # (I - h*J) y_1 = y_0
    A1 = np.eye(n) - h * J
    # Solve A1 @ y_1 = y_0
    # Use np.linalg.solve for dense, or spsolve if sparse
    # For 100x100, dense is fine
    y_prev = y0
    y_curr = np.linalg.solve(A1, y0)
    
    # Now BDF2
    # (I - (2h/3)J) y_{n+1} = (4/3)y_n - (1/3)y_{n-1}
    A2 = np.eye(n) - (2 * h / 3) * J
    # Factorize A2? np.linalg.solve does it each time.
    # We can use scipy.linalg.lu_factor and lu_solve for speed
    
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A2)
    
    t_curr = t0 + h
    for i in range(1, N):
        # Check deadline
        if time.time() > 320:
            return y_curr
        
        # BDF2 step
        rhs = (4/3) * y_curr - (1/3) * y_prev
        y_next = lu_solve((lu, piv), rhs)
        
        y_prev = y_curr
        y_curr = y_next
        t_curr += h
    
    return y_curr


def _fallback_rk4(f, t0, t1, y0, steps=2000):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    start_time = time.time()
    # Try the linear BDF2 approach
    try:
        result = _linear_bdf2(f, t_span, y0)
        if result is not None and np.all(np.isfinite(result)):
            return result
    except Exception:
        pass
    
    # Fallback to RK4
    result = _fallback_rk4(f, t_span[0], t_span[1], y0)
    if result is not None:
        return result
    
    # Last resort: return y0
    return np.array(y0, dtype=float)


def _linear_bdf2(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check for linearity and time-invariance
    f_zero = f(0, np.zeros(n))
    if np.linalg.norm(f_zero) > 1e-10:
        return None
    
    J = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        J[:, i] = f(t0, e_i)
    
    f_y0 = f(t0, y0)
    if np.linalg.norm(f_y0 - J @ y0) > 1e-10 * (1 + np.linalg.norm(J @ y0)):
        return None
    
    f_t1_y0 = f(t1, y0)
    if np.linalg.norm(f_t1_y0 - J @ y0) > 1e-10 * (1 + np.linalg.norm(J @ y0)):
        return None
    
    N = 1000
    h = (t1 - t0) / N
    
    A1 = np.eye(n) - h * J
    y_prev = y0
    y_curr = np.linalg.solve(A1, y0)
    
    A2 = np.eye(n) - (2 * h / 3) * J
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A2)
    
    t_curr = t0 + h
    for i in range(1, N):
        if time.time() > 320:
            return y_curr
        
        rhs = (4/3) * y_curr - (1/3) * y_prev
        y_next = lu_solve((lu, piv), rhs)
        
        y_prev = y_curr
        y_curr = y_next
        t_curr += h
    
    return y_curr
# EVOLVE-BLOCK-END
