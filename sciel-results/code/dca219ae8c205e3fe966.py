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
    
    # Identity for 2^N dimensional space
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply operator op on given site (0-indexed) to full 2^N space."""
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((2**N, 2**N), dtype=complex)
    # Odd bonds: (0,1), (2,3), (4,5)
    for i in range(0, N, 2):
        j = i + 1
        A += op_on_site(sx, i) @ op_on_site(sx, j)
        A += op_on_site(sy, i) @ op_on_site(sy, j)
        A += lam * op_on_site(sz, i) @ op_on_site(sz, j)
    # Half field
    for i in range(N):
        A += (h/2) * op_on_site(sz, i)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((2**N, 2**N), dtype=complex)
    # Even bonds: (1,2), (3,4)
    for i in range(1, N-1, 2):
        j = i + 1
        B += op_on_site(sx, i) @ op_on_site(sx, j)
        B += op_on_site(sy, i) @ op_on_site(sy, j)
        B += lam * op_on_site(sz, i) @ op_on_site(sz, j)
    # Half field
    for i in range(N):
        B += (h/2) * op_on_site(sz, i)
    
    return A, B

# Cache for eigen-decompositions
_eig_cache = {}

def _get_eig(A, B):
    """Get eigen-decompositions of A, B, and A+B, cached."""
    key = (id(A), id(B))
    if key not in _eig_cache:
        wA, vA = np.linalg.eigh(A)
        wB, vB = np.linalg.eigh(B)
        H = A + B
        wH, vH = np.linalg.eigh(H)
        _eig_cache[key] = (wA, vA, wB, vB, wH, vH)
    return _eig_cache[key]

def _compute_error(a, b, A, B):
    """Compute mean spectral-norm error at t = 0.25, 0.5, 1.0."""
    wA, vA, wB, vB, wH, vH = _get_eig(A, B)
    
    errors = []
    for t in [0.25, 0.5, 1.0]:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Use eigen-decompositions for efficiency
        U = np.eye(2**6, dtype=complex)
        
        for k in range(len(a)):
            # exp(-i a_k t A)
            expA = vA @ np.diag(np.exp(-1j * a[k] * t * wA)) @ vA.conj().T
            U = U @ expA
            # exp(-i b_k t B)
            expB = vB @ np.diag(np.exp(-1j * b[k] * t * wB)) @ vB.conj().T
            U = U @ expB
        
        # Exact: exp(-i t (A+B))
        U_exact = vH @ np.diag(np.exp(-1j * t * wH)) @ vH.conj().T
        
        diff = U - U_exact
        # Spectral norm = largest singular value
        err = np.linalg.norm(diff, 2)
        errors.append(err)
    
    return np.mean(errors)

def _nelder_mead(f, x0, max_iter=200, tol=1e-10):
    """Simple Nelder-Mead optimizer."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        xi[i] += 0.1 if x0[i] == 0 else 0.1 * x0[i]
        simplex.append(xi)
    
    fvals = [f(x) for x in simplex]
    
    for it in range(max_iter):
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
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
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
                    continue
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
                    continue
    
    # Return best point
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def coefficients(stages):
    """Optimize 12-stage product formula coefficients using Nelder-Mead."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    A, B = _build_hamiltonians()
    
    # Initial guess: symmetric Strang-like splitting
    a0 = [1.0/12] * 12
    b0 = [1.0/12] * 12
    
    # Optimize a and b separately, enforcing sum=1
    # Parameterize: a_i = a0_i + delta_a_i, with sum(delta_a)=0
    # Use 11 free parameters for a (last determined by sum constraint)
    
    def make_f(a_init, b_init):
        def f(params):
            # params: first 11 are a deltas, next 11 are b deltas
            a = list(a_init)
            b = list(b_init)
            
            for i in range(11):
                a[i] += params[i]
                b[i] += params[11 + i]
            
            # Enforce sum = 1
            a[11] = 1.0 - sum(a[:11])
            b[11] = 1.0 - sum(b[:11])
            
            # Clamp to [-50, 50]
            a = [max(-50, min(50, x)) for x in a]
            b = [max(-50, min(50, x)) for x in b]
            
            return _compute_error(a, b, A, B)
        return f
    
    f = make_f(a0, b0)
    x0 = [0.0] * 22
    
    # Run Nelder-Mead with time check
    best_x, best_f = x0, f(x0)
    
    # Custom loop to check time
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        xi[i] += 0.05
        simplex.append(xi)
    
    fvals = [f(x) for x in simplex]
    
    max_iter = 300
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < 1e-10:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
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
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
                    continue
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
                    continue
        
        # Track best
        order2 = np.argsort(fvals)
        if fvals[order2[0]] < best_f:
            best_f = fvals[order2[0]]
            best_x = simplex[order2[0]]
    
    # Extract a and b from best_x
    a = list(a0)
    b = list(b0)
    for i in range(11):
        a[i] += best_x[i]
        b[i] += best_x[11 + i]
    a[11] = 1.0 - sum(a[:11])
    b[11] = 1.0 - sum(b[:11])
    
    # Final clamp
    a = [max(-50, min(50, x)) for x in a]
    b = [max(-50, min(50, x)) for x in b]
    
    return a, b
# EVOLVE-BLOCK-END
