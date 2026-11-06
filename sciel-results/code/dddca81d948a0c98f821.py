"""
Strategy: Structural change - instead of reconstructing the full N×N matrix A and
computing expm(A*dt) via SVD/padé (O(N^3) with scipy), use Krylov subspace methods
to compute expm(A*dt) @ y0 directly in O(N * k) operations where k is the Krylov
dimension (much smaller than N for smooth convection-diffusion operators).

Key insight: For a 100-point convection-diffusion PDE discretized on a grid,
the operator A has a specific tridiagonal/pentadiagonal structure. Rather than
reconstructing A fully (N calls) and doing O(N^3) matrix exponential, we can:

1. Use a small number of basis probes to verify linearity and get A implicitly
2. Apply Arnoldi/Krylov methods to compute expm(A*dt) @ y0 without forming A explicitly
3. This reduces both the call count (still ~N for reconstruction) but more importantly
   reduces numerical error from the O(N^3) expm computation

Actually, let me reconsider. The score is -log10(rel_err) - 0.5*log10(calls).
The current approach uses N=100 calls + 1 check = 101 calls total.
The relative error from expm is essentially machine precision (~1e-15).

The bottleneck is the number of calls (101), not the accuracy. To improve the score,
we need to either:
- Reduce the number of calls (hard, since we need to identify A)
- Improve accuracy (already at machine precision)

Alternative structural change: Use the fact that convection-diffusion operators
are sparse and structured. If we can identify the stencil pattern (which for 1D
convection-diffusion is tridiagonal with 3 non-zero entries per row), we only need
3N calls to verify the structure, but actually we can do better.

Wait - for a 1D convection-diffusion on 100 points with standard discretization,
A is tridiagonal. We can exploit this:
- Only need to probe f on vectors that isolate the diagonal and off-diagonal entries
- A tridiagonal 100×100 matrix has 297 non-zero entries
- But we still need at least N calls to get all the information

Let me think differently. The real structural change: instead of expm(A*dt),
use a high-order implicit method with very few steps. For a stable problem,
Crank-Nicolson with 1 step gives O(dt^2) accuracy. With dt=0.5, that's ~0.25 error
which is terrible. But if we use a 4th-order Runge-Kutta in the implicit form...

Actually, the best structural change: use scipy's solve_ivp with the Jacobian
provided (since we can compute A @ v for free once we have A). This avoids
adaptive stepping overhead and uses the linear structure directly.

Or better: since we have A explicitly (after N calls), use scipy.linalg.expm
but with the scaling-and-squaring algorithm implemented to higher precision, or
use the eigenvalue decomposition which is more accurate for well-conditioned matrices.

Let me try: diagonalize A = V D V^{-1}, then expm(A*dt) = V exp(D*dt) V^{-1}.
For a 100×100 matrix, this is O(N^3) but with better constants than expm's
scaling-and-squaring. And for convection-diffusion, A should be diagonalizable
with real eigenvalues (it's a symmetric-ish operator).

Actually the cleanest structural change: use the eigenvalue approach AND verify
the result against a few time-stepped solutions to ensure correctness.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg
import scipy.integrate

def integrate(f, t_span, y0):
    """
    Solve linear autonomous ODE y' = A*y on [t0, t1] using eigenvalue decomposition.
    
    Structural change: Instead of scipy.linalg.expm (which uses scaling-and-squaring
    with Padé approximation), use explicit eigenvalue decomposition:
        y(t1) = V @ diag(exp(eigvals * dt)) @ V^{-1} @ y0
    
    This is more accurate for well-conditioned diagonalizable matrices and avoids
    the cumulative error in the Padé approximation chain.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    if N == 0:
        return y0
        
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0

    # Reconstruct A from N basis vector probes
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
    
    # Quick autonomy check
    e_0 = np.zeros(N)
    e_0[0] = 1.0
    v_t1 = f(t1, e_0)
    if not np.allclose(v_t1, A[:, 0], rtol=1e-10, atol=1e-12):
        # Fallback for non-autonomous case
        if time.time() > deadline - 5.0:
            return y0
        result = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='DOP853', rtol=1e-13, atol=1e-15
        )
        return result.y[:, -1] if result.success else y0

    # Try eigenvalue decomposition approach
    try:
        eigvals, eigvecs = scipy.linalg.eig(A)
        
        # Check conditioning of eigenvector matrix
        V = eigvecs
        cond_V = scipy.linalg.cond(V)
        
        if cond_V < 1e12:
            # Well-conditioned: use eigenvalue decomposition
            # y_final = V @ diag(exp(eigvals * dt)) @ inv(V) @ y0
            exp_eigvals = np.exp(eigvals * dt)
            
            # Compute inv(V) @ y0 first for efficiency
            try:
                V_inv_y0 = scipy.linalg.solve(V, y0)
                y_final = V @ (exp_eigvals * V_inv_y0)
            except np.linalg.LinAlgError:
                # V is singular, fall back to expm
                M = scipy.linalg.expm(A * dt)
                y_final = M @ y0
        else:
            # Ill-conditioned: use expm with higher precision scaling
            M = scipy.linalg.expm(A * dt)
            y_final = M @ y0
            
        return y_final
        
    except Exception:
        # Final fallback
        if time.time() > deadline - 5.0:
            return y0
        try:
            M = scipy.linalg.expm(A * dt)
            return M @ y0
        except Exception:
            result = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='Radau', rtol=1e-12, atol=1e-14
            )
            return result.y[:, -1] if result.success else y0

# EVOLVE-BLOCK-END
