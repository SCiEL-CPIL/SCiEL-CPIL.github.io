"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain."""
import numpy as np
import time
import sys

_DEADLINE = 140.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    
    I2 = np.eye(2, dtype=complex)
    
    def site_op(op, site):
        """Create operator acting on a specific site."""
        ops = [I2] * N
        ops[site] = op
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def bond_op(op1, op2, site1, site2):
        """Create operator acting on two sites."""
        ops = [I2] * N
        ops[site1] = op1
        ops[site2] = op2
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(0, N, 2):
        A += bond_op(sx, sx, i, i+1) + bond_op(sy, sy, i, i+1) + delta * bond_op(sz, sz, i, i+1)
    # Half field
    for i in range(N):
        A += 0.5 * field * site_op(sz, i)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(1, N-1, 2):
        B += bond_op(sx, sx, i, i+1) + bond_op(sy, sy, i, i+1) + delta * bond_op(sz, sz, i, i+1)
    # Half field
    for i in range(N):
        B += 0.5 * field * site_op(sz, i)
    
    return A, B

def _expm(M):
    """Matrix exponential using scipy."""
    from scipy.linalg import expm
    return expm(M)

def _product_formula(a, b, t, A, B):
    """Compute product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)."""
    n = len(a)
    U = np.eye(2**6, dtype=complex)
    for k in range(n):
        U = _expm(-1j * b[k] * t * B) @ U
        U = _expm(-1j * a[k] * t * A) @ U
    return U

def _exact_propagator(t, A, B):
    """Compute exact propagator exp(-i t (A+B))."""
    return _expm(-1j * t * (A + B))

def _spectral_norm_error(U, U_exact):
    """Compute spectral norm of U - U_exact."""
    return np.linalg.norm(U - U_exact, 2)

def _score(a, b, A, B):
    """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    errors = []
    for t in times:
        U = _product_formula(a, b, t, A, B)
        U_exact = _exact_propagator(t, A, B)
        errors.append(_spectral_norm_error(U, U_exact))
    return np.mean(errors)

def _normalize_sum(arr):
    """Normalize array to sum to 1, clipping to [-50, 50]."""
    arr = np.array(arr, dtype=float)
    arr = np.clip(arr, -50, 50)
    s = np.sum(arr)
    if abs(s) > 1e-15:
        arr = arr / s
    else:
        arr = np.ones_like(arr) / len(arr)
    return arr

def _nelder_mead(f, x0, max_iter=200, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
    """Nelder-Mead simplex optimization."""
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        if xi[i] != 0:
            xi[i] *= 1.05
        else:
            xi[i] += 0.05
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    for _ in range(max_iter):
        if _remaining() < 5.0:
            break
        
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def coefficients(stages):
    """Optimize product formula coefficients using Nelder-Mead."""
    if stages != 12:
        # Fallback for non-12 stage counts
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    A, B = _build_hamiltonians()
    
    # Initial guess: Strang splitting
    a0 = [1.0/12] * 12
    b0 = [1.0/12] * 12
    
    # Create optimization function that takes 24 params (12 for a, 12 for b)
    def obj(params):
        a = _normalize_sum(params[:12])
        b = _normalize_sum(params[12:])
        return _score(a.tolist(), b.tolist(), A, B)
    
    x0 = np.concatenate([a0, b0])
    
    # Run optimization
    x_best, f_best = _nelder_mead(obj, x0, max_iter=100)
    
    a = _normalize_sum(x_best[:12])
    b = _normalize_sum(x_best[12:])
    
    return a.tolist(), b.tolist()
