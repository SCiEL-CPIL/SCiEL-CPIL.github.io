"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_cut_diag(n, edges):
    """Precompute the cut diagonal for all 2^n states using vectorized bit ops."""
    num_states = 1 << n
    states = np.arange(num_states, dtype=np.int64)
    C_diag = np.zeros(num_states, dtype=float)
    for (u, v) in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        C_diag += (bits_u != bits_v).astype(float)
    return C_diag


def _expected_cut_value(n, gammas, betas, C_diag):
    """
    Compute the expected cut value for a given QAOA circuit.
    Uses exact state simulation with vectorized operations.
    """
    num_states = 1 << n
    psi = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i*gamma*C): diagonal
        phase = np.exp(-1j * gamma * C_diag)
        psi = psi * phase
        
        # Apply exp(-i*beta*M) where M = sum_i X_i
        c = np.cos(beta)
        s = np.sin(beta)
        
        for i in range(n):
            mask = 1 << i
            states = np.arange(num_states, dtype=np.int64)
            partner = states ^ mask
            
            psi_partner = psi[partner]
            psi = c * psi - 1j * s * psi_partner
    
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * C_diag)
    
    return expected_cut


def _exhaustive_maxcut(n, edges, C_diag):
    """Compute the exact maximum cut."""
    return float(np.max(C_diag))


def _nelder_mead(func, x0, max_iter=200, tol=1e-8, time_limit=None):
    """Nelder-Mead optimization."""
    n_params = len(x0)
    simplex = [x0.copy()]
    for i in range(n_params):
        xi = x0.copy()
        xi[i] += 0.1 if abs(xi[i]) < 1e-10 else 0.1 * xi[i]
        simplex.append(xi)
    
    fvals = [func(x) for x in simplex]
    
    iter_count = 0
    while iter_count < max_iter:
        if time_limit is not None and (time.time() > time_limit):
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[0]:
            gamma_exp = 2.0
            xe = centroid + gamma_exp * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            rho = 0.5
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n_params + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        iter_count += 1
    
    return simplex[0], fvals[0]


def _optimize_angles_for_graph(n, edges, p, time_budget=50.0):
    """Optimize QAOA angles for a specific graph using multi-start Nelder-Mead."""
    start_time = time.time()
    
    C_diag = _build_cut_diag(n, edges)
    opt_cut = _exhaustive_maxcut(n, edges, C_diag)
    if opt_cut == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    best_angles = np.array([0.35, 0.25, 0.35, 0.25])
    best_value = -1
    
    def objective(params):
        nonlocal best_angles, best_value
        gammas = params[:p]
        betas = params[p:]
        try:
            val = _expected_cut_value(n, gammas, betas, C_diag)
            ratio = val / opt_cut
            if ratio > best_value:
                best_value = ratio
                best_angles = params.copy()
            return -ratio
        except Exception:
            return 1e10
    
    # Generate diverse initial points
    initial_points = [
        np.array([0.35, 0.25, 0.35, 0.25]),
        np.array([0.5, 0.3, 0.4, 0.25]),
        np.array([0.25, 0.35, 0.3, 0.3]),
        np.array([0.4, 0.2, 0.5, 0.35]),
        np.array([0.3, 0.4, 0.25, 0.35]),
        np.array([0.45, 0.3, 0.35, 0.2]),
        np.array([0.2, 0.45, 0.4, 0.3]),
        np.array([0.55, 0.2, 0.3, 0.4]),
    ]
    
    time_limit = start_time + time_budget
    
    for i, x0 in enumerate(initial_points):
        if time.time() > time_limit:
            break
        
        remaining = time_limit - time.time()
        if remaining < 3.0:
            break
        
        num_starts = len(initial_points)
        alloc_time = remaining / max(1, num_starts - i)
        max_iter = max(30, int(alloc_time * 8))
        
        try:
            x_opt, f_opt = _nelder_mead(objective, x0, max_iter=max_iter, time_limit=time_limit)
            if f_opt < -1:
                if -f_opt > best_value:
                    best_value = -f_opt
                    best_angles = x_opt
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < time_limit - 5.0:
        try:
            x_opt, f_opt = _nelder_mead(objective, best_angles, max_iter=200, tol=1e-10, time_limit=time_limit)
            if f_opt < -1 and -f_opt > best_value:
                best_value = -f_opt
                best_angles = x_opt
        except Exception:
            pass
    
    best_angles = np.clip(best_angles, -np.pi, np.pi)
    
    result = np.zeros((p, 2))
    result[:, 0] = best_angles[:p]
    result[:, 1] = best_angles[p:]
    
    return result


def angles(n, edges, p):
    start_time = time.time()
    time_budget = 100.0
    
    try:
        result = _optimize_angles_for_graph(n, edges, p, time_budget=time_budget)
    except Exception:
        result = np.tile([0.35, 0.25], (p, 1))
    
    return result
# EVOLVE-BLOCK-END
