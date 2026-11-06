"""Optimized product formula for Heisenberg chain time evolution."""
import time
import numpy as np

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def op_on_site(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(j, op1, op2):
        ops = [I2] * N
        ops[j] = op1
        ops[j+1] = op2
        return kron_all(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    for j in [0, 2, 4]:
        A += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Even bonds: (1,2), (3,4) -> B
    for j in [1, 3]:
        B += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Uniform field: half in A, half in B
    for j in range(N):
        A += 0.5 * field * op_on_site(j, sz)
        B += 0.5 * field * op_on_site(j, sz)
    
    return A, B


def _precompute_eigs(A, B, H):
    """Precompute eigendecompositions of A, B, and H for fast matrix exponentials."""
    A_eig = np.linalg.eig(A)
    B_eig = np.linalg.eig(B)
    H_eig = np.linalg.eig(H)
    
    # Precompute inverse of eigenvector matrices
    A_inv = np.linalg.inv(A_eig[1])
    B_inv = np.linalg.inv(B_eig[1])
    H_inv = np.linalg.inv(H_eig[1])
    
    return A_eig, B_eig, H_eig, A_inv, B_inv, H_inv


def _expm_from_eig(evals, evecs, inv_evecs, scale):
    """Compute exp(scale * M) from precomputed eigendecomposition."""
    return evecs @ np.diag(np.exp(scale * evals)) @ inv_evecs


def _objective_fast(a, b, A_eig, B_eig, H_eig, A_inv, B_inv, H_inv, dim):
    """Fast objective using precomputed eigendecompositions."""
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        # Exact propagator
        exact = _expm_from_eig(H_eig[0], H_eig[1], H_inv, -1j * t)
        
        # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Applied in order k = 0, 1, ..., 11
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            exp_a = _expm_from_eig(A_eig[0], A_eig[1], A_inv, -1j * a[k] * t)
            exp_b = _expm_from_eig(B_eig[0], B_eig[1], B_inv, -1j * b[k] * t)
            U = exp_b @ exp_a @ U
        
        diff = U - exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0] if len(s) > 0 else 0.0)
    
    return np.mean(errors)


def coefficients(stages):
    """
    12-stage product formula with precomputed eigendecompositions and
    multi-start gradient-free optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Precompute eigendecompositions
    A_eig, B_eig, H_eig, A_inv, B_inv, H_inv = _precompute_eigs(A, B, H)
    
    def eval_score(a, b):
        """Evaluate with bounds and sum checks."""
        if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
            return 1e10
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        return _objective_fast(a, b, A_eig, B_eig, H_eig, A_inv, B_inv, H_inv, dim)
    
    # Generate diverse initial candidates using known high-order splitting patterns
    candidates = []
    
    # 1. Equal weights baseline
    eq = np.ones(stages) / stages
    candidates.append((eq.copy(), eq.copy()))
    
    # 2. Symmetric 4th-order Suzuki-Trotter (Yoshida) coefficients
    # For 4th order: c = 2^(1/3), x1 = 1/(2-c), x2 = -c/(2-c), x3 = 1 - 2*x1
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - 2.0 * x1
    
    # Symmetric 6-step pattern, duplicated for 12 steps
    a_sym = np.array([x3/2, x1/2, x1/2, x3/2, x1/2, x1/2,
                       x3/2, x1/2, x1/2, x3/2, x1/2, x1/2])
    b_sym = np.array([x2/2, x2/2, 0, 0, x2/2, x2/2,
                       x2/2, x2/2, 0, 0, x2/2, x2/2])
    a_sym = a_sym / np.sum(a_sym)
    b_sym = b_sym / np.sum(b_sym)
    candidates.append((a_sym.copy(), b_sym.copy()))
    
    # 3. Alternating pattern
    a_alt = np.zeros(stages)
    b_alt = np.zeros(stages)
    for k in range(stages):
        if k % 2 == 0:
            a_alt[k] = 2.0 / stages
            b_alt[k] = 0.0
        else:
            a_alt[k] = 0.0
            b_alt[k] = 2.0 / stages
    a_alt = a_alt / np.sum(a_alt)
    b_alt = b_alt / np.sum(b_alt)
    candidates.append((a_alt.copy(), b_alt.copy()))
    
    # 4. Linear ramp
    a_ramp = np.linspace(0.5, 1.5, stages)
    b_ramp = np.linspace(1.5, 0.5, stages)
    a_ramp = a_ramp / np.sum(a_ramp)
    b_ramp = b_ramp / np.sum(b_ramp)
    candidates.append((a_ramp.copy(), b_ramp.copy()))
    
    # 5. Center-weighted
    center = np.arange(stages) - stages / 2.0
    a_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    b_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    a_cw = a_cw / np.sum(a_cw)
    b_cw = b_cw / np.sum(b_cw)
    candidates.append((a_cw.copy(), b_cw.copy()))
    
    # Evaluate all candidates
    best_a = candidates[0][0].copy()
    best_b = candidates[0][1].copy()
    best_score = eval_score(best_a, best_b)
    
    for a_c, b_c in candidates[1:]:
        s = eval_score(a_c, b_c)
        if s < best_score:
            best_a = a_c.copy()
            best_b = b_c.copy()
            best_score = s
    
    # Nelder-Mead with reduced parameterization (22 free params)
    def constrained_obj(params):
        a_free = params[:11]
        b_free = params[11:22]
        a_full = np.append(a_free, 1.0 - np.sum(a_free))
        b_full = np.append(b_free, 1.0 - np.sum(b_free))
        return eval_score(a_full, b_full)
    
    # Multi-start Nelder-Mead
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x0 = np.concatenate([best_a[:11], best_b[:11]])
        else:
            x0 = np.concatenate([best_a[:11], best_b[:11]]) + np.random.normal(0, 0.2, 22)
        
        # Initialize simplex
        n = 22
        simplex = [x0.copy()]
        for i in range(n):
            v = x0.copy()
            v[i] += 0.05 * (abs(v[i]) + 1.0)
            simplex.append(v)
        
        fvals = [constrained_obj(v) for v in simplex]
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = 200
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[idx] for idx in order]
            fvals = [fvals[idx] for idx in order]
            
            if abs(fvals[0] - fvals[-1]) < 1e-10:
                break
            
            a_full = np.append(simplex[0][:11], 1.0 - np.sum(simplex[0][:11]))
            b_full = np.append(simplex[0][11:22], 1.0 - np.sum(simplex[0][11:22]))
            if fvals[0] < best_score:
                best_a = a_full.copy()
                best_b = b_full.copy()
                best_score = fvals[0]
            
            centroid = np.mean(simplex[:-1], axis=0)
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = constrained_obj(x_r)
            
            if f_r < fvals[0]:
                x_e = centroid + gamma * (x_r - centroid)
                f_e = constrained_obj(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = constrained_obj(x_c)
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    for j in range(1, n + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        fvals[j] = constrained_obj(simplex[j])
    
    # Final cleanup
    best_a = np.clip(best_a, -50, 50)
    best_b = np.clip(best_b, -50, 50)
    best_a = best_a + (1.0 - np.sum(best_a)) / stages
    best_b = best_b + (1.0 - np.sum(best_b)) / stages
    
    if np.any(best_a < -50) or np.any(best_a > 50) or np.any(best_b < -50) or np.any(best_b > 50):
        best_a = np.ones(stages) / stages
        best_b = np.ones(stages) / stages
    
    return list(best_a), list(best_b)
