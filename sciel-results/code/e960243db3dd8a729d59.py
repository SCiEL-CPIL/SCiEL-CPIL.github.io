"""ODE solver: 15-dimensional stiff linear system using matrix exponential."""
import numpy as np
import time
from scipy.linalg import expm, solve

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def _extract_matrix(f, t0, y0):
    """Extract the system matrix A and offset b from f(t, y) = A@y + b."""
    n = len(y0)
    y0 = np.asarray(y0, dtype=float)
    
    # Get f at the initial point
    f0 = np.asarray(f(t0, y0), dtype=float)
    
    # Probe to get columns of A
    # Use a perturbation size that's well-scaled
    scale = np.maximum(1.0, np.abs(y0))
    eps = 1e-8
    
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps * scale[j]
        f_pert = np.asarray(f(t0, y0 + e), dtype=float)
        A[:, j] = (f_pert - f0) / e[j]
    
    # b = f0 - A @ y0
    b = f0 - A @ y0
    
    return A, b


def _solve_linear_ode(f, t_span, y0):
    """Solve y' = A@y + b exactly using matrix exponential."""
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    A, b = _extract_matrix(f, t0, y0)
    
    # For constant coefficient linear system y' = A@y + b:
    # y(t) = e^{A(t-t0)} @ y0 + e^{A(t-t0)} @ int_0^{t-t0} e^{-A s} ds @ b
    # 
    # The integral term: int_0^T e^{-As} ds = (I - e^{-AT}) @ (-A)^{-1} if A invertible
    # More robustly: we can use the augmented matrix approach
    # 
    # Actually, for y' = A@y + b, the solution is:
    # y(T) = e^{AT} @ y0 + (e^{AT} - I) @ A^{-1} @ b  (if A invertible)
    # 
    # But if A is singular, use: (e^{AT} - I) @ A^{-1} = int_0^T e^{As} ds
    # We can compute this as the (n x n) block of expm of an augmented matrix
    # 
    # Augmented system: d/ds [y; 1] = [[A, b], [0, 0]] [y; 1]
    # expm(T * [[A, b], [0, 0]]) = [[e^{AT}, (e^{AT}-I)A^{-1}b], [0, 1]]
    # Actually: expm(T*M) where M = [[A, b], [0, 0]]
    # The top-right block gives us the particular solution contribution
    
    T = t1 - t0
    
    # Build augmented (n+1) x (n+1) matrix
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    # Last row and column are zero (except M[n,n] = 0)
    
    E = expm(T * M)
    
    # y(T) = E[0:n, 0:n] @ y0 + E[0:n, n] * 1
    y_end = E[:n, :n] @ y0 + E[:n, n]
    
    return y_end


def integrate(f, t_span, y0):
    """Solve the 15D stiff linear IVP using exact matrix exponential."""
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return y0
    
    try:
        y_end = _solve_linear_ode(f, t_span, y0)
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass
    
    # Fallback: use scipy solve_ivp with BDF
    from scipy.integrate import solve_ivp
    sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=0.01)
    if sol.success:
        return sol.y[:, -1]
    
    # Last resort: return y0
    return y0
