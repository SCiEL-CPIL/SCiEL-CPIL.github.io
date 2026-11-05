"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # 1. Probe for linearity and time-invariance
    # We assume the problem is y' = A*y (common for heat equation).
    # We verify this by checking if f(t, y) is linear in y and independent of t.
    
    # Check time invariance: f(t0, y0) vs f(t1, y0)
    # Also check linearity: f(t, y0 + y1) == f(t, y0) + f(t, y1)
    
    # To save calls, we'll do a few probes.
    # Probe 1: f(t0, y0)
    y0 = np.array(y0, dtype=float)
    f_t0_y0 = f(t0, y0)
    
    # Probe 2: f(t1, y0)
    f_t1_y0 = f(t1, y0)
    
    # Check time invariance
    if not np.allclose(f_t0_y0, f_t1_y0, rtol=1e-9, atol=1e-12):
        # Not time invariant, fall back to RK45 or similar
        return _rk45(f, t_span, y0)
        
    # Check linearity: f(t, y0 + y0) == 2 * f(t, y0)?
    # f(t, 2*y0) should be 2 * f(t, y0) if linear
    f_t_2y0 = f(t0, 2.0 * y0)
    if not np.allclose(f_t_2y0, 2.0 * f_t0_y0, rtol=1e-9, atol=1e-12):
        # Not linear, fall back
        return _rk45(f, t_span, y0)
        
    # Check additivity: f(t, y0 + y1) == f(t, y0) + f(t, y1)
    # We need a y1. We can use a small perturbation or just check homogeneity more.
    # Actually, homogeneity f(t, c*y) = c*f(t, y) for c=2 and c=1 (trivial) is a good sign.
    # Let's check f(t, 0.5 * y0) == 0.5 * f(t, y0)
    f_t_half_y0 = f(t0, 0.5 * y0)
    if not np.allclose(f_t_half_y0, 0.5 * f_t0_y0, rtol=1e-9, atol=1e-12):
        return _rk45(f, t_span, y0)
        
    # It seems linear and time invariant.
    # Now we need to reconstruct A.
    # f(t, y) = A @ y
    # We have A @ y0 = f_t0_y0. This is one equation for n unknowns in A (n*n).
    # We need n linearly independent vectors y_i such that we can form Y = [y1, ..., yn]
    # and F = [f(y1), ..., f(yn)]
    # Then A = F @ inv(Y)
    
    # Strategy: Use the standard basis vectors e_i.
    # But we already used y0.
    # Let's try to build A using standard basis vectors.
    # However, calling f n times for basis vectors is n calls.
    # Total calls so far: 4.
    # If n=100, 100 calls is fine.
    
    # But wait, we can be smarter.
    # If we assume the structure of the heat equation (5-point Laplacian), we can infer A directly?
    # The prompt mentions "Analytic structure assumption (5-point Laplacian) eliminates the need for N probe calls".
    # Let's try to detect if A matches the 5-point Laplacian structure.
    
    # First, let's just reconstruct A generally using basis vectors.
    # We need to be careful with the number of calls.
    # n calls for basis vectors.
    
    # Let's check if y0 is close to a basis vector? No.
    # Let's just use basis vectors.
    
    A = np.zeros((n, n))
    
    # We already have f(t0, y0).
    # Let's use the fact that we can call f(t, x) for any x.
    
    # To minimize calls, we can use a random matrix V and solve A V = F V.
    # But basis vectors are safest for numerical stability if we do it right.
    
    # Let's use the standard basis vectors e_i.
    # For i in range(n):
    #    e_i = zeros(n); e_i[i] = 1
    #    f_i = f(t0, e_i)
    #    A[:, i] = f_i
    
    # This takes n calls.
    # Total calls: 4 + n.
    
    # Is there a way to do better?
    # If we assume the 5-point Laplacian structure, we only need to verify the coefficients.
    # The heat equation on a 2D grid with Dirichlet or Neumann BCs has a specific sparse structure.
    # However, the problem statement says "fixed 10x10 two-dimensional heat equation".
    # This implies a specific grid and likely standard discretization.
    # But we shouldn't hardcode the grid size unless we know it. n=100 implies 10x10.
    
    # Let's try to detect the structure.
    # For a 5-point Laplacian on a 10x10 grid (row-major),
    # A[i, i] = -4 (or -2 for Neumann corners? No, usually -4 for interior, less for boundaries)
    # A[i, i-10] = 1 (up)
    # A[i, i+10] = 1 (down)
    # A[i, i-1] = 1 (left)
    # A[i, i+1] = 1 (right)
    
    # Let's probe a few points to see if this structure holds.
    # We already have f(t0, y0).
    
    # Let's try to reconstruct A using a small number of probes that cover the structure.
    # Actually, reconstructing A fully with n calls is robust.
    # n=100, so 104 calls total. This is very small compared to 2000000.
    # And expm is exact (up to numerical precision).
    
    # Let's proceed with full reconstruction.
    
    # We need to be careful: the first 4 calls were for y0, 2y0, 0.5y0.
    # We can reuse f_t0_y0 if y0 happens to be a basis vector, but it's unlikely.
    
    A = np.zeros((n, n))
    
    # We will compute columns of A.
    # col_i = f(t0, e_i)
    
    # Optimization: If we suspect the matrix is sparse, we can verify sparsity.
    # But let's just compute it.
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_ei = f(t0, e_i)
        A[:, i] = f_ei
        
    # Now we have A.
    # Verify A @ y0 == f_t0_y0
    if not np.allclose(A @ y0, f_t0_y0, rtol=1e-9, atol=1e-12):
        # Something went wrong, fall back
        return _rk45(f, t_span, y0)
        
    # Compute expm(A * dt)
    # This is the exact solution for y' = A y.
    # y(t) = expm(A * (t - t0)) @ y0
    
    # Check if we have time.
    # expm for 100x100 is fast.
    
    E = expm(A * dt)
    y_end = E @ y0
    
    return y_end


def _rk45(f, t_span, y0):
    """Fallback RK45 integrator."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Simple RK4 with fixed step size, or adaptive?
    # Let's use a fixed step RK4 with a reasonable number of steps.
    # The original seed used 2000 steps.
    # Let's use 1000 steps for RK4.
    steps = 1000
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y

# EVOLVE-BLOCK-END
