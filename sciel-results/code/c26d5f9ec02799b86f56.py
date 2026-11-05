"""Optimized product-formula coefficients for Heisenberg-chain time evolution."""
import numpy as np
import time
import sys

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for a 6-site open XXZ Heisenberg chain.
    
    Anisotropy gamma = 0.8, field h = 0.3.
    The chain has sites 0..5. Bonds: (0,1), (1,2), (2,3), (3,4), (4,5).
    Odd bonds: (0,1), (2,3), (4,5) -> A
    Even bonds: (1,2), (3,4) -> B
    Uniform field h*S_z is split equally: h/2 in A and h/2 in B.
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        """Kronecker product of a list of 2x2 operators."""
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    # Build S_z for each site (full chain)
    # S_z on site j = I x ... x (sz/2) x ... x I
    def site_op(site, op):
        ops = []
        for i in range(N):
            if i == site:
                ops.append(op)
            else:
                ops.append(np.eye(2, dtype=complex))
        return kron_all(ops)

    # Build Hamiltonian for a set of bonds
    def bond_hamiltonian(bonds, field_frac):
        H = np.zeros((dim, dim), dtype=complex)
        # XXZ interaction: for each bond (i, i+1):
        # S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + gamma * S_i^z S_{i+1}^z
        # where S^x = sx/2, S^y = sy/2, S^z = sz/2
        for (i, j) in bonds:
            sx_i = site_op(i, sx / 2.0)
            sx_j = site_op(j, sx / 2.0)
            sy_i = site_op(i, sy / 2.0)
            sy_j = site_op(j, sy / 2.0)
            sz_i = site_op(i, sz / 2.0)
            sz_j = site_op(j, sz / 2.0)
            H += sx_i @ sx_j + sy_i @ sy_j + gamma * (sz_i @ sz_j)
        # Uniform field: field_frac * h * sum_j S_j^z
        if field_frac != 0:
            for j in range(N):
                H += field_frac * h * site_op(j, sz / 2.0)
        return H

    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]

    A = bond_hamiltonian(odd_bonds, 0.5)
    B = bond_hamiltonian(even_bonds, 0.5)

    return A, B


def _compute_error(a, b, A, B, times):
    """Compute mean spectral-norm error over the given times."""
    n = len(a)
    errors = []
    for t in times:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(2**6, dtype=complex)
        for k in range(n):
            if abs(b[k]) > 1e-15:
                U = np.linalg.expm(-1j * b[k] * t * B) @ U
            if abs(a[k]) > 1e-15:
                U = np.linalg.expm(-1j * a[k] * t * A) @ U
        # Exact propagator
        H = A + B
        U_exact = np.linalg.expm(-1j * t * H)
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)


def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, deadline=None):
    """Pure-NumPy Nelder-Mead optimizer with bounds."""
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy() for _ in range(n + 1)]
    for i in range(n):
        step = 0.01 * max(abs(x0[i]), 1.0)
        simplex[i + 1][i] += step

    # Evaluate initial simplex
    fvals = []
    for i in range(n + 1):
        fv = f(simplex[i])
        fvals.append(fv)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i].copy() for i in order]
        fvals = [fvals[i] for i in order]

        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break

        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - simplex[-1])
        f_ref = f(x_ref)

        if fvals[0] <= f_ref < fvals[-2]:
            simplex[-1] = x_ref
            fvals[-1] = f_ref
        elif f_ref < fvals[0]:
            # Expansion
            gamma = 2.0
            x_exp = centroid + gamma * (x_ref - centroid)
            f_exp = f(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                fvals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
        else:
            # Contraction
            rho = 0.5
            x_con = centroid + rho * (simplex[-1] - centroid)
            f_con = f(x_con)
            if f_con < fvals[-1]:
                simplex[-1] = x_con
                fvals[-1] = f_con
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])

    # Return best
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]


def coefficients(stages):
    """Find optimal coefficients for the product formula using Nelder-Mead optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Build Hamiltonians
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]

    # Initial guess: equal distribution
    a0 = [1.0 / stages] * stages
    b0 = [1.0 / stages] * stages

    # Objective: we need sum(a)=1, sum(b)=1
    # We'll optimize 2*stages - 2 free parameters (fix last of each to satisfy sum constraint)
    # Actually, let's optimize all 2*stages and use penalty for sum constraints
    # Or better: parameterize with (stages-1) free params for each, last = 1 - sum(others)

    def objective(params):
        """params: 2*(stages-1) free parameters + 2 fixed (sum constraints handled)."""
        free_a = params[:stages - 1]
        free_b = params[stages - 1:2 * (stages - 1)]
        a = list(free_a) + [1.0 - sum(free_a)]
        b = list(free_b) + [1.0 - sum(free_b)]

        # Check bounds
        for val in a + b:
            if val < -50 or val > 50:
                return 1e10

        return _compute_error(a, b, A, B, times)

    # Initial parameters
    x0 = [1.0 / stages] * (stages - 1) + [1.0 / stages] * (stages - 1)

    # Run Nelder-Mead
    best_x, best_f = x0, objective(x0)

    if time.time() < deadline - 5.0:
        x_opt, f_opt = _nelder_mead(objective, x0, None, max_iter=500, tol=1e-12, deadline=deadline - 5.0)
        if f_opt < best_f:
            best_x = x_opt
            best_f = f_opt

    # Extract final coefficients
    free_a = best_x[:stages - 1]
    free_b = best_x[stages - 1:2 * (stages - 1)]
    a = list(free_a) + [1.0 - sum(free_a)]
    b = list(free_b) + [1.0 - sum(free_b)]

    # Clamp to bounds
    a = [max(-50.0, min(50.0, v)) for v in a]
    b = [max(-50.0, min(50.0, v)) for v in b]

    # Re-normalize to ensure exact sum = 1
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [v / sa for v in a]
    if abs(sb) > 1e-15:
        b = [v / sb for v in b]

    return a, b
