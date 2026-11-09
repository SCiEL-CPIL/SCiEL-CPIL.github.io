"""Product-formula coefficient optimizer for 6-site Heisenberg chain.

Structural change: Use cached eigen-decompositions and a direct numerical
optimization with flexible (non-symmetric) parameterization. The formula
U = prod_k exp(-i b_k t B) exp(-i a_k t A) has ordering asymmetry, so we
optimize all 24 coefficients directly (with sum constraints) using fast
Nelder-Mead with precomputed eigendecompositions.
"""
import numpy as np
import time

_DEADLINE = 145.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    N = 6
    delta = 0.8
    field = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def site_op(op, site):
        ops = [I] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(i, j):
        ops_sz = [I] * N
        ops_sz[i] = sz
        ops_sz[j] = sz
        ops_sx = [I] * N
        ops_sx[i] = sx
        ops_sx[j] = sx
        ops_sy = [I] * N
        ops_sy[i] = sy
        ops_sy[j] = sy
        return delta * (kron_all(ops_sz) + kron_all(ops_sx) + kron_all(ops_sy))
    
    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    A = bond_op(1, 2) + bond_op(3, 4)
    
    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)
    
    A += field_op
    B += field_op
    
    return A, B

# Precompute eigen-decompositions (cached globally)
_eig_cache = None

def _get_eig():
    """Get cached eigen-decompositions of A, B, and H=A+B."""
    global _eig_cache
    if _eig_cache is None:
        A, B = _build_hamiltonians()
        H = A + B
        ev_A, vec_A = np.linalg.eigh(A)
        ev_B, vec_B = np.linalg.eigh(B)
        ev_H, vec_H = np.linalg.eigh(H)
        dim = A.shape[0]
        _eig_cache = {
            'A': A, 'B': B, 'H': H, 'dim': dim,
            'ev_A': ev_A, 'vec_A': vec_A,
            'ev_B': ev_B, 'vec_B': vec_B,
            'ev_H': ev_H, 'vec_H': vec_H,
        }
    return _eig_cache

def _compute_error_fast(a, b):
    """Compute mean spectral norm error using cached eigen-decompositions."""
    cache = _get_eig()
    ev_A, vec_A = cache['ev_A'], cache['vec_A']
    ev_B, vec_B = cache['ev_B'], cache['vec_B']
    ev_H, vec_H = cache['ev_H'], cache['vec_H']
    dim = cache['dim']
    
    times = [0.25, 0.5, 1.0]
    errors = []
    
    for t in times:
        # Exact propagator
        exact = (vec_H * np.exp(-1j * t * ev_H)) @ vec_H.conj().T
        
        # Product formula
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            exp_b = (vec_B * np.exp(-1j * b[k] * t * ev_B)) @ vec_B.conj().T
            exp_a = (vec_A * np.exp(-1j * a[k] * t * ev_A)) @ vec_A.conj().T
            U = exp_b @ exp_a @ U
        
        err = np.linalg.norm(U - exact, ord=2)
        errors.append(err)
    
    return np.mean(errors)

def coefficients(stages):
    """Optimize 12-stage product formula coefficients via direct numerical optimization."""
    if stages != 12:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    # Initialize cache
    cache = _get_eig()
    
    # Start with equal weights
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_err = _compute_error_fast(best_a, best_b)
    
    # --- Phase 1: Multi-start with diverse initial guesses ---
    initial_guesses = []
    
    # Equal weights
    initial_guesses.append(([1.0/12]*12, [1.0/12]*12))
    
    # Strang-like: a and b equal, with slight edge emphasis
    for edge_w in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        a_g = [1.0/12] * 12
        b_g = [1.0/12] * 12
        # Emphasize edges
        for k in range(12):
            dist = min(k, 11-k)
            a_g[k] = 1.0/12 * (1.0 + 0.3*(6-dist)/6.0)
            b_g[k] = 1.0/12 * (1.0 + 0.3*(6-dist)/6.0)
        # Normalize
        sa = sum(a_g); sb = sum(b_g)
        a_g = [x/sa for x in a_g]
        b_g = [x/sb for x in b_g]
        initial_guesses.append((a_g, b_g))
    
    # 4th-order Suzuki-like: negative coefficients at edges
    # For 4th order with 4 sub-steps per "half", scale to 12
    # Standard 4th order: alpha1=alpha3=1/(6-6^(1/3)), alpha2=alpha4=-1/(6-6^(1/3))
    # beta1=beta3=1/(2-6^(1/3)), beta2=beta4=1/(2-6^(1/3))
    c = 6**(1/3)
    alpha1 = 1.0/(6.0 - 6.0*c)
    alpha2 = -1.0/(6.0 - 6.0*c)
    beta1 = 1.0/(2.0 - 6.0*c)
    # Scale to sum to 1 over 12 stages
    # Use pattern: repeat the 4-coeff pattern 3 times
    a_g = [alpha1, alpha2, alpha2, alpha1] * 3
    b_g = [beta1, beta1, beta1, beta1] * 3
    sa = sum(a_g); sb = sum(b_g)
    if abs(sa) > 1e-10:
        a_g = [x/sa for x in a_g]
    if abs(sb) > 1e-10:
        b_g = [x/sb for x in b_g]
    initial_guesses.append((a_g, b_g))
    
    # Try each initial guess
    for a_g, b_g in initial_guesses:
        if _remaining() < 5:
            break
        err = _compute_error_fast(a_g, b_g)
        if err < best_err:
            best_err = err
            best_a = a_g[:]
            best_b = b_g[:]
    
    # --- Phase 2: Coordinate descent refinement (fast, 1D line searches) ---
    def _line_search(a, b, idx, is_a, cur_err):
        """1D line search for one coefficient with sum constraint."""
        # Vary one coefficient, adjust another to maintain sum=1
        # Use a paired adjustment: change param i, adjust param (i+1)%12
        other_idx = (idx + 1) % 12
        
        best_local = cur_err
        best_val = a[idx] if is_a else b[idx]
        
        # Try several step sizes
        for step in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            for direction in [-1, 1]:
                new_val = best_val + direction * step
                if new_val < -50 or new_val > 50:
                    continue
                
                if is_a:
                    a_new = a[:]
                    a_new[idx] = new_val
                    # Adjust other to maintain sum
                    a_new[other_idx] = a[other_idx] - (new_val - a[idx])
                    b_new = b[:]
                else:
                    a_new = a[:]
                    b_new = b[:]
                    b_new[idx] = new_val
                    b_new[other_idx] = b[other_idx] - (new_val - b[idx])
                
                # Check bounds
                if any(x < -50 or x > 50 for x in a_new + b_new):
                    continue
                
                err = _compute_error_fast(a_new, b_new)
                if err < best_local:
                    best_local = err
                    best_val = new_val
                    if is_a:
                        a[idx] = new_val
                        a[other_idx] = a[other_idx] - (new_val - a[idx])
                    else:
                        b[idx] = new_val
                        b[other_idx] = b[other_idx] - (new_val - b[idx])
        
        return best_local
    
    # Multiple passes of coordinate descent
    for pass_num in range(5):
        if _remaining() < 10:
            break
        
        for idx in range(12):
            if _remaining() < 5:
                break
            cur_err = _compute_error_fast(best_a, best_b)
            new_err = _line_search(best_a, best_b, idx, True, cur_err)
            if new_err < best_err:
                best_err = new_err
        
        for idx in range(12):
            if _remaining() < 5:
                break
            cur_err = _compute_error_fast(best_a, best_b)
            new_err = _line_search(best_a, best_b, idx, False, cur_err)
            if new_err < best_err:
                best_err = new_err
    
    # --- Phase 3: Nelder-Mead on reduced parameter set (pairs) ---
    # Use 12 parameters: for each k, define d_k = a_k - b_k, with a_k+b_k fixed by symmetry
    # Actually, let's just do a final Nelder-Mead with tighter bounds
    
    def objective(x):
        a = list(x[:12])
        b = list(x[12:24])
        sa = sum(a)
        sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        a = [max(-50.0, min(50.0, x)) for x in a]
        b = [max(-50.0, min(50.0, x)) for x in b]
        sa = sum(a); sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        return _compute_error_fast(a, b)
    
    if _remaining() > 30:
        x0 = best_a + best_b
        n_params = 24
        simplex = [x0[:]]
        step = 0.01
        for i in range(n_params):
            xi = x0[:]
            xi[i] += step
            simplex.append(xi)
        
        fvals = [objective(x) for x in simplex]
        
        no_improve = 0
        for it in range(200):
            if _remaining() < 8:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            centroid = np.mean(simplex[:n_params], axis=0)
            
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[n_params])
            f_ref = objective(x_ref)
            
            if f_ref < fvals[0]:
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[n_params])
                f_exp = objective(x_exp)
                if f_exp < f_ref:
                    simplex[n_params] = x_exp
                    fvals[n_params] = f_exp
                else:
                    simplex[n_params] = x_ref
                    fvals[n_params] = f_ref
                no_improve = 0
            elif f_ref < fvals[n_params]:
                simplex[n_params] = x_ref
                fvals[n_params] = f_ref
                no_improve = 0
            else:
                rho = 0.5
                x_con = centroid + rho * (simplex[n_params] - centroid)
                f_con = objective(x_con)
                if f_con < fvals[n_params]:
                    simplex[n_params] = x_con
                    fvals[n_params] = f_con
                    no_improve = 0
                else:
                    for i in range(1, n_params+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
                    no_improve += 1
            
            if fvals[0] < best_err:
                best_err = fvals[0]
                best_a = list(simplex[0][:12])
                best_b = list(simplex[0][12:24])
            
            if no_improve > 30:
                break
    
    # Final normalization
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    # Clip
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    # Re-normalize after clipping
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    return best_a, best_b
