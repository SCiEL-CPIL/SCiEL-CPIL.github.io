"""Nelder-Mead optimization of 12-stage product formula coefficients for Heisenberg chain."""
import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    
    The full Hamiltonian H = A + B where:
    - A = sum over odd bonds (1-2, 3-4, 5-6) of XXZ interaction + half uniform field
    - B = sum over even bonds (2-3, 4-5) of XXZ interaction + half uniform field
    
    XXZ interaction on bond (i, i+1): Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + lambda Sz_i Sz_{i+1}
    Uniform field: h * sum_i Sz_i, split as h/2 in A and h/2 in B
    """
    N = 6
    lam = 0.8
    h = 0.3
    
    # Pauli matrices / spin-1/2 operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_interaction(i, j):
        """XXZ interaction between sites i and j (0-indexed)."""
        ops = [eye] * N
        ops[i] = sx
        ops[j] = sx
        term1 = kron_all(ops)
        
        ops = [eye] * N
        ops[i] = sy
        ops[j] = sy
        term2 = kron_all(ops)
        
        ops = [eye] * N
        ops[i] = sz
        ops[j] = sz
        term3 = kron_all(ops)
        
        return term1 + term2 + lam * term3
    
    # Odd bonds: (0,1), (2,3), (4,5)
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in [(0,1), (2,3), (4,5)]:
        A += bond_interaction(i, j)
    # Half field in A
    for i in range(N):
        ops = [eye] * N
        ops[i] = sz
        A += (h/2) * kron_all(ops)
    
    # Even bonds: (1,2), (3,4)
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in [(1,2), (3,4)]:
        B += bond_interaction(i, j)
    # Half field in B
    for i in range(N):
        ops = [eye] * N
        ops[i] = sz
        B += (h/2) * kron_all(ops)
    
    return A, B

def _compute_error(a, b, A, B, A_eig, B_eig, times):
    """
    Compute mean spectral-norm error over the given times.
    U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) for k=1..12
    Exact: exp(-i t (A+B))
    """
    total_err = 0.0
    for t in times:
        U = np.eye(2**6, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B)
            if abs(b[k]) > 1e-15:
                expB = B_eig[0] @ (np.diag(np.exp(-1j * b[k] * t * B_eig[1])) @ B_eig[2])
            else:
                expB = np.eye(2**6, dtype=complex)
            U = expB @ U
            
            # exp(-i a_k t A)
            if abs(a[k]) > 1e-15:
                expA = A_eig[0] @ (np.diag(np.exp(-1j * a[k] * t * A_eig[1])) @ A_eig[2])
            else:
                expA = np.eye(2**6, dtype=complex)
            U = expA @ U
        
        # Exact propagator
        H = A + B
        # Use eigendecomposition of H (symmetric/Hermitian)
        # We'll compute this once outside; but for simplicity compute here
        # Actually let's precompute exact propagator outside
        
        # For now, compute exact using eig
        eigh_H, evec_H = np.linalg.eigh(H)
        exact = evec_H @ np.diag(np.exp(-1j * t * eigh_H)) @ evec_H.conj().T
        
        diff = U - exact
        # Spectral norm = largest singular value
        err = np.linalg.norm(diff, 2)
        total_err += err
    return total_err / len(times)

def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-10):
    """Simple Nelder-Mead implementation with bounds handling."""
    n = len(x0)
    # Create initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 if x0[i] == 0 else abs(x0[i]) * 0.1
        xi[i] += step
        # Clip to bounds
        for j in range(n):
            xi[j] = max(bounds[j][0], min(bounds[j][1], xi[j]))
        simplex.append(xi)
    
    fvals = [f(x) for x in simplex]
    best_idx = np.argmin(fvals)
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    for it in range(max_iter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i].copy() for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients using Nelder-Mead.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Precompute eigen-decompositions
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    
    # Precompute exact propagators at target times
    times = [0.25, 0.5, 1.0]
    H = A + B
    H_eigvals, H_eigvecs = np.linalg.eigh(H)
    exact_props = []
    for t in times:
        exact = H_eigvecs @ np.diag(np.exp(-1j * t * H_eigvals)) @ H_eigvecs.conj().T
        exact_props.append(exact)
    
    dim = 2**6
    identity = np.eye(dim, dtype=complex)
    
    def objective(params):
        """
        params: 24-vector [a_1,...,a_12, b_1,...,b_12]
        Enforce sum(a)=1, sum(b)=1 by projecting.
        """
        a_raw = params[:12]
        b_raw = params[12:]
        
        # Project to sum=1
        a = a_raw - np.mean(a_raw) + 1.0/12.0
        b = b_raw - np.mean(b_raw) + 1.0/12.0
        
        # Clip to [-50, 50]
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        
        # Re-project after clipping (simple: renormalize)
        a = a - np.mean(a) + 1.0/12.0
        b = b - np.mean(b) + 1.0/12.0
        
        total_err = 0.0
        for idx, t in enumerate(times):
            U = identity.copy()
            for k in range(12):
                if abs(b[k]) > 1e-15:
                    expB = B_eigvecs @ (np.diag(np.exp(-1j * b[k] * t * B_eigvals)) @ B_eigvecs.conj().T)
                else:
                    expB = identity
                U = expB @ U
                
                if abs(a[k]) > 1e-15:
                    expA = A_eigvecs @ (np.diag(np.exp(-1j * a[k] * t * A_eigvals)) @ A_eigvecs.conj().T)
                else:
                    expA = identity
                U = expA @ U
            
            diff = U - exact_props[idx]
            err = np.linalg.norm(diff, 2)
            total_err += err
        
        return total_err / len(times)
    
    # Initial guess: uniform splitting
    x0 = np.ones(24) / 12.0
    
    # Bounds: each coefficient in [-50, 50]
    bounds = [(-50.0, 50.0)] * 24
    
    # Run Nelder-Mead
    best_x, best_f = _nelder_mead(objective, x0, bounds, max_iter=300, tol=1e-12)
    
    elapsed = time.time() - start_time
    
    # If we have time, try a few more restarts from perturbed points
    if elapsed < deadline - 30:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            x0_r = best_x + np.random.randn(24) * 0.1
            x0_r = np.clip(x0_r, -50, 50)
            xr, fr = _nelder_mead(objective, x0_r, bounds, max_iter=150, tol=1e-12)
            if fr < best_f:
                best_x = xr
                best_f = fr
    
    # Extract a and b
    a_raw = best_x[:12]
    b_raw = best_x[12:]
    a = a_raw - np.mean(a_raw) + 1.0/12.0
    b = b_raw - np.mean(b_raw) + 1.0/12.0
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Final projection
    a = a - np.mean(a) + 1.0/12.0
    b = b - np.mean(b) + 1.0/12.0
    
    return list(a), list(b)
