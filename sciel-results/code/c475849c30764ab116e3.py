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
    """Precompute eigendecompositions of A, B, and H using eigh for Hermitian matrices."""
    A_evals, A_evecs = np.linalg.eigh(A)
    B_evals, B_evecs = np.linalg.eigh(B)
    H_evals, H_evecs = np.linalg.eigh(H)
    return (A_evals, A_evecs), (B_evals, B_evecs), (H_evals, H_evecs)

def _expm_from_eigh(evals, evecs, scale):
    """Compute exp(scale * M) from precomputed eigh decomposition."""
    exp_vals = np.exp(scale * evals)
    return (evecs * exp_vals) @ evecs.conj().T

def coefficients(stages):
    """
    12-stage product formula with fast evaluation and Nelder-Mead optimization.
    """
    start_time = time.time()
    # Evaluator kills at 150s. We use 140s as hard limit to be safe.
    deadline = start_time + 140.0
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Precompute eigendecompositions
    A_eig, B_eig, H_eig = _precompute_eigs(A, B, H)
    t_vals = [0.25, 0.5, 1.0]

    def eval_score(a, b):
        """Fast evaluation of spectral norm error."""
        # Quick sum check to avoid expensive calc if wildly off (though NM should stay close)
        # We assume inputs are reasonably normalized by the optimizer parameterization
        
        errors = []
        
        for t in t_vals:
            # Exact propagator
            exact = _expm_from_eigh(H_eig[0], H_eig[1], -1j * t)
            
            # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
            # Order k = 0..11
            U = np.eye(dim, dtype=complex)
            
            for k in range(stages):
                # Compute exps
                exp_a = _expm_from_eigh(A_eig[0], A_eig[1], -1j * a[k] * t)
                exp_b = _expm_from_eigh(B_eig[0], B_eig[1], -1j * b[k] * t)
                
                # U = exp_b @ exp_a @ U
                # Optimize multiplication order? 
                # (exp_b @ exp_a) @ U is usually fine
                temp = exp_a @ U
                U = exp_b @ temp
                
            diff = U - exact
            s = np.linalg.svd(diff, compute_uv=False)
            errors.append(s[0] if len(s) > 0 else 0.0)
        
        return np.mean(errors)

    # Parameterization: 
    # We optimize 24 variables a[0..11], b[0..11] directly.
    # To handle constraints sum(a)=1, sum(b)=1, we can use a transformation.
    # But Nelder-Mead works well on unconstrained spaces if we penalize violations.
    # Alternatively, parameterize a[11] = 1 - sum(a[0..10]).
    # This reduces dimension to 22.
    
    def constrained_obj(params):
        a_free = params[:11]
        b_free = params[11:22]
        a_full = np.zeros(stages)
        b_full = np.zeros(stages)
        
        a_full[:11] = a_free
        a_full[11] = 1.0 - np.sum(a_free)
        
        b_full[:11] = b_free
        b_full[11] = 1.0 - np.sum(b_free)
        
        # Bounds check: if any coeff is outside [-50, 50], return large penalty
        # This prevents the optimizer from wandering into invalid regions
        if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10
            
        return eval_score(a_full, b_full)

    # Initial Guess: 4th Order Symmetric Coefficients
    # For 4th order, we typically have specific ratios.
    # Let's use the known 4th order coefficients for a 6-step (3A, 3B) or 12-step variant.
    # A simple robust start is equal weights, but 4th order is better.
    # Yoshida 4th order coefficients for (A+B):
    # c1 = 1/(2-c), c2 = -c/(2-c), c3 = 1-2*c1 where c = 2^(1/3)
    # The sequence is usually:
    # B(c1) A(c2) B(c1) A(c2) B(c1)  ?? No.
    # Standard 4th order: 
    # exp(-B c1 t) exp(-A c2 t) exp(-B c3 t) exp(-A c2 t) exp(-B c1 t)
    # This is 5 steps? No, 5 blocks.
    # Let's stick to a simple symmetric start for 12 stages.
    # a = [1/12]*12, b = [1/12]*12 is the 1st order.
    
    # Let's try a "2nd order" like start: 
    # a = [0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6] -> sum=1
    # b = [1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0] -> sum=1
    # This is the Strang splitting duplicated.
    
    a_init = np.zeros(stages)
    b_init = np.zeros(stages)
    for k in range(stages):
        if k % 2 == 0:
            a_init[k] = 1.0 / 6.0
            b_init[k] = 0.0
        else:
            a_init[k] = 0.0
            b_init[k] = 1.0 / 6.0
    
    x0 = np.concatenate([a_init[:11], b_init[:11]])
    
    # Nelder-Mead Implementation
    n_params = 22
    
    def nelder_mead(f, x0, max_iter=1000, timeout=None):
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(n_params):
            x_new = x0.copy()
            # Perturbation size
            if x0[i] == 0:
                x_new[i] += 0.1
            else:
                x_new[i] *= 1.1
            simplex.append(x_new)
            
        fvals = [f(x) for x in simplex]
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        best_x = simplex[0]
        best_f = fvals[0]
        
        for i in range(max_iter):
            if timeout and time.time() > timeout:
                break
            
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[idx] for idx in order]
            fvals = [fvals[idx] for idx in order]
            
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < 1e-10:
                break
                
            # Reflection
            centroid = np.mean(simplex[:-1], axis=0)
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = f(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = f(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = f(x_c)
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for j in range(1, n_params + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        fvals[j] = f(simplex[j])
                        
        return best_x, best_f

    best_params = x0
    best_score = constrained_obj(best_params)
    
    # Run multiple restarts
    np.random.seed(42)
    restarts = 5
    
    for r in range(restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            x_start = x0.copy()
        else:
            # Add noise to current best
            noise = np.random.normal(0, 0.05, n_params)
            x_start = best_params + noise
            
        # Ensure bounds are not violated by noise in the free params
        # The constrained_obj handles the penalty, so it's safe
        
        x_res, f_res = nelder_mead(constrained_obj, x_start, max_iter=500, timeout=deadline - 5)
        
        if f_res < best_score:
            best_score = f_res
            best_params = x_res.copy()
            
    # Reconstruct final a and b
    a_best = np.zeros(stages)
    b_best = np.zeros(stages)
    a_best[:11] = best_params[:11]
    a_best[11] = 1.0 - np.sum(best_params[:11])
    
    b_best[:11] = best_params[11:22]
    b_best[11] = 1.0 - np.sum(best_params[11:22])
    
    # Final sanity checks
    a_best = np.clip(a_best, -50, 50)
    b_best = np.clip(b_best, -50, 50)
    
    # Enforce sum=1 exactly
    a_best = a_best + (1.0 - np.sum(a_best)) / stages
    b_best = b_best + (1.0 - np.sum(b_best)) / stages
    
    return list(a_best), list(b_best)
