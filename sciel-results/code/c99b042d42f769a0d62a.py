"""Sutton-Chen cluster global minima via coarse-to-fine annealing basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    for i in range(n - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        if d.size and d.min() < 0.35:
            return False
    return True


def _truncate_by_radius(pts, n, center=None):
    if len(pts) <= n:
        while len(pts) < n:
            p = pts[np.random.default_rng(len(pts)).integers(0, len(pts))]
            p = p + np.random.default_rng(len(pts) * 7).normal(0, 0.1, 3)
            pts.append(p)
        return np.array(pts[:n])
    if center is None:
        center = np.mean(pts, axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _fcc_cluster(n, lattice=0.72):
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _hcp_cluster(n, lattice=0.72):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                pts.append([x, y, z])
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _icosahedral_shells(n, r1=1.0, r2=1.27, r3=1.51):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v * r1)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * r2
                pts.append(mid)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * r3
                    pts.append(fc)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            v = verts[rng.integers(0, len(verts))]
            v2 = verts[rng.integers(0, len(verts))]
            p = (v + v2) / 2
            p = p / np.linalg.norm(p) * (r3 + 0.3)
            pts.append(p)
    return _truncate_by_radius(np.array(pts), n)


def _decahedral_cluster(n):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    r = 1.0
    pts = []
    for v in verts:
        pts.append(np.array([v[0], v[1], v[2] - 0.6]) * r)
    pts.append(np.array([0, 0, -0.6]))
    for v in verts:
        pts.append(np.array([v[0], v[1], v[2] + 0.6]) * r)
    pts.append(np.array([0, 0, 0.6]))
    rng = np.random.default_rng(11)
    while len(pts) < n:
        p = rng.normal(0, 1.5, 3)
        pts.append(p)
    return _truncate_by_radius(np.array(pts), n)


def _truncated_fcc(n, lattice=0.72):
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    p = base + np.array(off)
                    if np.linalg.norm(p) < 0.85 * n ** (1.0 / 3.0) * 0.9:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                for k in range(-size, size + 1):
                    base = np.array([i, j, k]) * lattice
                    for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                                [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                        pts = np.vstack([pts, base + np.array(off)])
                        if len(pts) >= n:
                            break
                    if len(pts) >= n:
                        break
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
    return _truncate_by_radius(pts, n)


def _random_fcc_perturbed(n, seed, lattice=0.72, pert=0.15):
    base = _fcc_cluster(n, lattice)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, pert, size=(n, 3))


def _random_hcp_perturbed(n, seed, lattice=0.72, pert=0.15):
    base = _hcp_cluster(n, lattice)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, pert, size=(n, 3))


def _random_compact(n, seed, min_dist=0.75):
    rng = np.random.default_rng(seed)
    max_radius = 0.85 * n ** (1.0 / 3.0)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 500:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            if pts:
                arr = np.array(pts)
                if np.min(np.linalg.norm(arr - p, axis=1)) < min_dist:
                    ok = False
            if ok:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        pts.append(p)
    return np.array(pts)


def _soft_relax(positions, potential, kind, n_exp_soft, m_exp_soft, c_soft):
    """
    Relax using a softened potential by temporarily modifying the kind parameters.
    This is a hack: we create a temporary kind dict with softened exponents.
    """
    # We need to mimic the relax function but with different exponents.
    # Since we can't easily modify the imported relax, we will implement a simple
    # gradient-based relaxation using the provided energy_gradient but with modified
    # parameters? No, sutton_chen_energy_gradient also uses the kind.
    
    # Alternative: Use scipy.optimize.minimize with a custom energy function that
    # uses the softened parameters.
    from scipy.optimize import minimize
    
    def soft_energy(pos):
        pos = pos.reshape(-1, 3)
        n = pos.shape[0]
        e = 0.0
        pair_e = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(pos[i] - pos[j])
                if r < 1e-6:
                    r = 1e-6
                r_n = r ** (-n_exp_soft)
                r_m = r ** (-m_exp_soft)
                pair_e[i, j] = r_n
                pair_e[j, i] = r_n
                e += r_n - c_soft * np.sqrt(r_m)
        # Sutton-Chen form: sum_i [1/2 sum_j r^-n - c sqrt(sum_j r^-m)]
        # My loop above added pair terms to e directly, which is wrong for the second term.
        # Let's re-calculate properly.
        e = 0.0
        pair_rep = np.zeros((n, n))
        pair_att = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(pos[i] - pos[j])
                if r < 1e-6:
                    r = 1e-6
                pair_rep[i, j] = r ** (-n_exp_soft)
                pair_rep[j, i] = r ** (-n_exp_soft)
                pair_att[i, j] = r ** (-m_exp_soft)
                pair_att[j, i] = r ** (-m_exp_soft)
        
        rep_sum = 0.5 * np.sum(pair_rep)
        att_sum = np.array([np.sqrt(np.sum(pair_att[i])) for i in range(n)])
        e = rep_sum - c_soft * np.sum(att_sum)
        return e

    def soft_gradient(pos):
        pos = pos.reshape(-1, 3)
        n = pos.shape[0]
        grad = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = pos[i] - pos[j]
                r = np.linalg.norm(r_vec)
                if r < 1e-6:
                    continue
                r2 = r * r
                # d/dr of r^-n = -n r^-(n+1)
                # d/dr of r^-m = -m r^-(m+1)
                
                # Repulsive part: 0.5 * sum_{j!=i} r^-n
                # Gradient for i: -0.5 * n * r^-(n+1) * r_vec / r = -0.5 * n * r^-(n+2) * r_vec
                # Wait, d/dr (r^-n) = -n r^-(n+1).
                # d/dx_i (r^-n) = -n r^-(n+1) * (x_i - x_j) / r = -n r^-(n+2) (x_i - x_j)
                # The energy term is 0.5 * sum_{j!=i} r_ij^-n.
                # Contribution to grad_i from pair (i,j) is 0.5 * (-n r^-(n+2)) * (x_i - x_j)
                # Contribution to grad_j from pair (i,j) is 0.5 * (-n r^-(n+2)) * (x_j - x_i) = - (contribution to i)
                
                # Attractive part: -c * sum_i sqrt(sum_{j!=i} r_ij^-m)
                # Let S_i = sum_{j!=i} r_ij^-m. Term is -c * sqrt(S_i).
                # d/dx_i (-c * sqrt(S_i)) = -c * (1/(2 sqrt(S_i))) * d(S_i)/dx_i
                # d(S_i)/dx_i = sum_{j!=i} d(r_ij^-m)/dx_i = sum_{j!=i} -m r_ij^-(m+2) (x_i - x_j)
                
                # This is getting complex to vectorize correctly in a loop.
                # Given the time constraint and complexity, let's just use L-BFGS-B
                # with the standard potential but from a very perturbed start?
                # No, the strategy is "soft potential".
                
                # Let's implement the gradient numerically if analytical is too hard,
                # or just return a dummy gradient and let L-BFGS-B use finite differences?
                # No, L-BFGS-B with finite differences is slow.
                
                # Let's stick to the analytical derivation for a pair (i,j):
                # E_rep = 0.5 * sum_{k!=l} r_kl^-n
                # dE_rep/dx_i = 0.5 * sum_{j!=i} d(r_ij^-n)/dx_i = 0.5 * sum_{j!=i} (-n r_ij^-(n+1) * (x_i-x_j)/r_ij)
                # = -0.5 * n * sum_{j!=i} r_ij^-(n+2) (x_i-x_j)
                
                # E_att = -c * sum_i sqrt(S_i), where S_i = sum_{j!=i} r_ij^-m
                # dE_att/dx_i = -c * d/dx_i [sqrt(S_i)] = -c * (1/(2 sqrt(S_i))) * dS_i/dx_i
                # dS_i/dx_i = sum_{j!=i} d(r_ij^-m)/dx_i = sum_{j!=i} (-m r_ij^-(m+1) * (x_i-x_j)/r_ij)
                # = -m * sum_{j!=i} r_ij^-(m+2) (x_i-x_j)
                
                # So dE_att/dx_i = -c * (1/(2 sqrt(S_i))) * (-m * sum_{j!=i} r_ij^-(m+2) (x_i-x_j))
                # = (c * m / (2 sqrt(S_i))) * sum_{j!=i} r_ij^-(m+2) (x_i-x_j)
                
                # We need S_i for each i.
                pass
        
        # This is too slow in Python loops for n=75.
        # I will abandon the custom soft potential relaxation and instead
        # use the standard relax but with a "soft" start by scaling coordinates?
        # No, the inspiration says "soft-potential relaxation smooths the energy landscape".
        
        # Let's try a different approach: 
        # Use the provided relax function but call it on a "softened" version of the problem?
        # We can't change the potential parameters in the imported function.
        
        # Okay, I will implement a simple steepest descent or Nelder-Mead with the soft energy
        # function defined above. Nelder-Mead is robust and doesn't need gradients.
        
        # Let's rewrite _soft_relax to use Nelder-Mead with the soft energy.
        
        # First, calculate S_i for the gradient if we were doing it analytically.
        # But for now, let's just return the energy and let the caller use Nelder-Mead.
        # Actually, the function signature expects it to return (energy, positions).
        # I will use scipy.optimize.minimize with method='Nelder-Mead' inside _soft_relax.
        
        # Refactor: _soft_relax should do the optimization.
        pass

    # Re-implementing _soft_relax properly with Nelder-Mead
    def _soft_relax_impl(positions, potential, kind, n_exp_soft, m_exp_soft, c_soft):
        n = positions.shape[0]
        
        def soft_energy_flat(x):
            pos = x.reshape(-1, 3)
            e = 0.0
            S = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    r = np.linalg.norm(pos[i] - pos[j])
                    if r < 1e-6: continue
                    S[i] += r ** (-m_exp_soft)
            
            for i in range(n):
                for j in range(i + 1, n):
                    r = np.linalg.norm(pos[i] - pos[j])
                    if r < 1e-6: continue
                    e += 0.5 * r ** (-n_exp_soft)
            
            # Attractive term
            for i in range(n):
                if S[i] > 0:
                    e -= c_soft * np.sqrt(S[i])
            return e

        result = minimize(soft_energy_flat, positions.flatten(), method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.success:
            return result.fun, result.x.reshape(-1, 3)
        else:
            # Return best found
            best_e = np.inf
            best_x = positions
            # Nelder-Mead result.x is the best point even if not converged
            return soft_energy_flat(result.x), result.x.reshape(-1, 3)

    return _soft_relax_impl(positions, potential, kind, n_exp_soft, m_exp_soft, c_soft)


def construct(instance):
    start_time = time.time()
    deadline = start_time + 310.0  # safety margin vs 330s kill

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    # Extract parameters
    if kind == '12-6':
        n_exp = 12; m_exp = 6; c = 144.41
    elif kind == '9-6':
        n_exp = 9; m_exp = 6; c = 39.432
    elif kind == '10-8':
        n_exp = 10; m_exp = 8; c = 34.408
    else:
        # Fallback to 12-6
        n_exp = 12; m_exp = 6; c = 144.41

    # --- Phase 1: Generate diverse structural seeds ---
    seeds = []
    seeds.append(_fcc_cluster(n, lattice=0.72))
    seeds.append(_fcc_cluster(n, lattice=0.70))
    seeds.append(_fcc_cluster(n, lattice=0.74))
    seeds.append(_hcp_cluster(n, lattice=0.72))
    seeds.append(_hcp_cluster(n, lattice=0.70))
    seeds.append(_icosahedral_shells(n))
    seeds.append(_icosahedral_shells(n, r1=0.95, r2=1.22, r3=1.45))
    seeds.append(_decahedral_cluster(n))
    seeds.append(_truncated_fcc(n, lattice=0.72))
    seeds.append(_truncated_fcc(n, lattice=0.70))
    
    for s in range(5):
        seeds.append(_random_fcc_perturbed(n, seed=100 + s, pert=0.1 + s * 0.05))
        seeds.append(_random_hcp_perturbed(n, seed=200 + s, pert=0.1 + s * 0.05))
    
    for s in range(8):
        seeds.append(_random_compact(n, seed=500 + s, min_dist=0.7 + s * 0.05))

    # --- Phase 2: Coarse-to-Fine Annealing ---
    # We will take the best few seeds and apply soft relaxation, then ramp to hard.
    # To save time, we'll pick the top 3 seeds by standard relax energy first?
    # No, standard relax is fast. Let's do standard relax on all seeds to get a baseline.
    
    best_e = np.inf
    best_pos = None
    
    # Standard relax on all seeds to find a good starting point for annealing
    for idx, s in enumerate(seeds):
        if time.time() > deadline - 60.0:  # Leave 60s for annealing
            break
        if not _validate_positions(s):
            continue
        try:
            e, p = relax(s, potential, kind)
            if np.isfinite(e) and _validate_positions(p):
                if e < best_e:
                    best_e = e
                    best_pos = p.copy()
        except Exception:
            pass

    if best_pos is None:
        # Fallback
        pos = _random_compact(n, seed=999, min_dist=0.8)
        try:
            best_e, best_pos = relax(pos, potential, kind)
        except Exception:
            best_pos = pos
        if not _validate_positions(best_pos):
            return _random_compact(n, seed=12345, min_dist=0.9)

    # Now, perform coarse-to-fine annealing starting from best_pos
    # We will run a sequence of soft relaxations with decreasing softness.
    # Softness parameter t goes from 0 (soft) to 1 (hard).
    # n_eff = n_exp - (n_exp - n_min) * (1-t)
    # m_eff = m_exp - (m_exp - m_min) * (1-t)
    # c_eff = c * (0.5 + 0.5*t)  # Scale c down in soft phase
    
    n_min = 4  # Very soft
    m_min = 2
    
    num_steps = 20
    current_pos = best_pos.copy()
    current_e = best_e
    
    for step in range(num_steps):
        if time.time() > deadline - 10.0:
            break
            
        t = (step + 1) / num_steps  # 0.05 to 1.0
        
        # Interpolate exponents
        n_eff = n_min + (n_exp - n_min) * t
        m_eff = m_min + (m_exp - m_min) * t
        c_eff = c * (0.5 + 0.5 * t)
        
        try:
            e_soft, pos_soft = _soft_relax(current_pos, potential, kind, n_eff, m_eff, c_eff)
            if np.isfinite(e_soft) and _validate_positions(pos_soft):
                # Evaluate with the HARD potential to see if we improved
                e_hard, _ = sutton_chen_energy_gradient(pos_soft, kind)
                if np.isfinite(e_hard) and e_hard < current_e:
                    current_e = e_hard
                    current_pos = pos_soft.copy()
                else:
                    # If soft relaxation didn't improve hard energy, 
                    # maybe we should just keep the previous pos?
                    # Or try a small perturbation?
                    pass
        except Exception:
            pass

    # --- Final Polish with Standard Relax ---
    if time.time() < deadline - 5.0:
        try:
            e_final, p_final = relax(current_pos, potential, kind)
            if np.isfinite(e_final) and _validate_positions(p_final):
                if e_final < best_e:
                    best_e = e_final
                    best_pos = p_final
                else:
                    # Keep the one from annealing if it was better?
                    # We already updated best_e/best_pos in the loop? No, we updated current_e/current_pos.
                    # Let's compare.
                    if current_e < best_e:
                        best_e = current_e
                        best_pos = current_pos
        except Exception:
            pass

    # Ensure we return the best found
    if best_pos is None or not _validate_positions(best_pos):
        return _random_compact(n, seed=12345, min_dist=0.9)

    return best_pos
# EVOLVE-BLOCK-END
