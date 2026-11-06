"""Antireflection coating design via differential evolution + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def _reflectance_stack(thetas, n_layers, n_sub, n_amb=1.0):
    """
    Mean normal-incidence reflectance (%) over wavelengths.
    thetas: (N, n_layers) optical thicknesses (n*d) per wavelength in radians.
    """
    n = n_layers
    if n == 0:
        rr = ((n_amb - n_sub) / (n_amb + n_sub)) ** 2
        return 100.0 * float(np.mean(rr))
    
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    
    # Start from the substrate side
    # B = n_sub * cos, A = 1 (normalized)
    # Actually, let's use the standard characteristic matrix approach.
    # M_total = M_1 * M_2 * ... * M_n
    # [B, A] = M_total * [n_sub, 1]
    # r = (n_amb * A - B) / (n_amb * A + B)
    
    # Initialize B and A as vectors for all wavelengths
    B = n_sub * cos[:, -1]
    A = cos[:, -1]
    # Wait, the last matrix is M_n.
    # M_j = [[cos_j, i sin_j / n_j], [i n_j sin_j, cos_j]]
    # So for the last layer (index n-1):
    # B = n_sub * cos_{n-1}
    # A = cos_{n-1}
    # But we need to multiply by the previous matrices.
    # Let's propagate from the last layer to the first.
    
    # Correct initialization:
    # The vector [B, A] starts as [n_sub, 1] at the substrate interface.
    # Then we multiply by M_{n-1}, M_{n-2}, ..., M_0.
    
    B = n_sub * np.ones_like(thetas[:, 0])
    A = np.ones_like(thetas[:, 0])
    
    # Multiply by matrices from last layer to first
    for j in range(n - 1, -1, -1):
        c = cos[:, j]
        s = sin[:, j]
        n_j = 1.0  # This is wrong, we need the index of layer j.
        
    # Let's rewrite this more carefully.
    # We need the indices n_j. They are not passed to this function directly,
    # but thetas are n_j * d_j * k. We can't recover n_j from thetas alone.
    # So this function signature is insufficient for a general implementation unless
    # we pass the indices or precompute the matrices.
    
    # Let's change the approach: pass the indices or precompute cos/sin and 1/n.
    # For efficiency, I will implement the objective function inside design()
    # where I have access to the indices.
    
    return 0.0 # Placeholder

def design(problem):
    n_amb = float(problem.get("ambient_index", 1.0))
    n_sub = float(problem.get("substrate_index", 1.52))
    nL = float(problem["materials"]["L"])
    nH = float(problem["materials"]["H"])
    w0, w1 = problem["band_nm"]
    max_layers = int(problem.get("max_layers", 20))
    max_total = float(problem.get("max_total_nm", 1000))
    max_layer = float(problem.get("max_layer_nm", 400))
    n_w = 401

    t_start = time.time()
    DEADLINE = 145.0  # Safety margin from 150s

    # Precompute wavelength grid and k
    wv = np.linspace(w0, w1, n_w)
    k = 2.0 * np.pi / wv  # (n_w,)

    # We will search over thicknesses for a fixed number of layers.
    # Let's try a few different layer counts and patterns.
    # Pattern: Alternating L, H, L, H... is usually good for broadband.
    
    def compute_objective(thicknesses, indices):
        """
        thicknesses: (n_layers,) array of physical thicknesses in nm.
        indices: (n_layers,) array of refractive indices.
        Returns mean reflectance in percent.
        """
        n_layers = len(thicknesses)
        if n_layers == 0:
            return 100.0 * ((n_amb - n_sub) / (n_amb + n_sub)) ** 2
        
        # thetas = k * indices * thicknesses
        # k is (n_w,), indices is (n_layers,), thicknesses is (n_layers,)
        thetas = k[:, None] * (indices * thicknesses)[None, :]  # (n_w, n_layers)
        
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        inv_n = 1.0 / indices  # (n_layers,)
        
        # Initialize B and A at the substrate
        B = n_sub * np.ones(n_w, dtype=complex)
        A = np.ones(n_w, dtype=complex)
        
        # Propagate from last layer to first
        for j in range(n_layers - 1, -1, -1):
            c = cos_t[:, j]
            s = sin_t[:, j]
            n_j = indices[j]
            
            # M_j = [[c, i*s/n_j], [i*n_j*s, c]]
            # New B = c*B + i*(s/n_j)*A
            # New A = i*n_j*s*B + c*A
            new_B = c * B + 1j * (s * inv_n[j]) * A
            new_A = 1j * (n_j * s) * B + c * A
            B = new_B
            A = new_A
        
        r = (n_amb * A - B) / (n_amb * A + B)
        R = np.abs(r) ** 2
        return float(np.mean(R) * 100.0)

    def nelder_mead(x0, f, max_iter=200, tol=1e-8):
        d0 = np.array(x0, dtype=float)
        npar = d0.size
        f0 = f(d0)
        simplex = [d0.copy()]
        fvals = [f0]
        
        # Initial simplex
        for i in range(npar):
            step = d0.copy()
            step[i] += 0.1 * max(abs(d0[i]), 1.0)
            simplex.append(step)
            fvals.append(f(step))
        
        for _ in range(max_iter):
            if time.time() > DEADLINE - 5.0:
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                xe = centroid + 2.0 * (xr - centroid)
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
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, npar + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            
            if abs(max(fvals) - min(fvals)) < tol:
                break
        
        order = np.argsort(fvals)
        return simplex[order[0]].copy(), fvals[order[0]]

    def differential_evolution(f, bounds, pop_size=20, max_iter=50, F=0.8, CR=0.9):
        npar = len(bounds)
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        
        # Initialize population
        pop = np.random.uniform(lb, ub, (pop_size, npar))
        fpop = np.array([f(ind) for ind in pop])
        
        best_idx = np.argmin(fpop)
        best_ind = pop[best_idx].copy()
        best_f = fpop[best_idx]
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 5.0:
                break
            for i in range(pop_size):
                # Select three distinct random individuals
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                while a == b or a == c or b == c:
                    a, b, c = np.random.choice(pop_size, 3, replace=False)
                
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, lb, ub)
                
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(npar)
                for j in range(npar):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                
                ft = f(trial)
                if ft <= fpop[i]:
                    pop[i] = trial
                    fpop[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_ind = trial.copy()
        
        return best_ind, best_f

    # Search strategy:
    # 1. Try different layer counts (e.g., 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    # 2. For each count, use alternating L/H pattern.
    # 3. Use DE for global search, then NM for local polish.
    
    best_overall_d = None
    best_overall_f = float('inf')
    best_overall_labels = None
    
    rng = np.random.default_rng(42)
    
    # Candidate layer counts
    layer_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    for n_layers in layer_counts:
        if time.time() > DEADLINE - 10.0:
            break
        
        # Pattern: L, H, L, H...
        seq = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        indices = np.array([nL if m == 'L' else nH for m in seq])
        
        # Bounds for thicknesses
        # Each layer must be <= max_layer and sum <= max_total.
        # Simple bounds: [0, max_layer] for each.
        # We will handle the sum constraint by scaling or rejection.
        # For DE, it's better to use [0, max_layer] and then project.
        
        def objective_wrapped(d):
            # Check sum constraint
            if np.sum(d) > max_total:
                # Penalize or scale? Scaling might break local structure.
                # Let's just return a high value if constraint is violated?
                # No, better to scale down to satisfy constraint.
                d_scaled = d * (max_total / np.sum(d))
                return compute_objective(d_scaled, indices)
            return compute_objective(d, indices)
        
        # Initial population for DE
        # Some structured starts + random
        pop_size = 20
        max_iter_de = 30
        
        # Run DE
        d_de, f_de = differential_evolution(
            objective_wrapped, 
            [(0.0, max_layer) for _ in range(n_layers)], 
            pop_size=pop_size, 
            max_iter=max_iter_de
        )
        
        # Polish with NM
        d_nm, f_nm = nelder_mead(d_de, objective_wrapped, max_iter=100)
        
        # Check if this is better
        if f_nm < best_overall_f:
            # Ensure constraints are met for the final result
            d_final = d_nm.copy()
            if np.sum(d_final) > max_total:
                d_final = d_final * (max_total / np.sum(d_final))
            
            # Clip to [0, max_layer]
            d_final = np.clip(d_final, 0.0, max_layer)
            
            # Re-evaluate to be sure
            f_final = compute_objective(d_final, indices)
            
            if f_final < best_overall_f:
                best_overall_f = f_final
                best_overall_d = d_final
                best_overall_labels = seq
        
        # Also try a few random starts with NM directly for speed
        for _ in range(3):
            if time.time() > DEADLINE - 10.0:
                break
            d_rand = rng.uniform(0.0, 50.0, size=n_layers)
            d_rand, f_rand = nelder_mead(d_rand, objective_wrapped, max_iter=50)
            if f_rand < best_overall_f:
                d_final = np.clip(d_rand, 0.0, max_layer)
                if np.sum(d_final) > max_total:
                    d_final = d_final * (max_total / np.sum(d_final))
                f_final = compute_objective(d_final, indices)
                if f_final < best_overall_f:
                    best_overall_f = f_final
                    best_overall_d = d_final
                    best_overall_labels = seq

    # Fallback
    if best_overall_d is None:
        # Single quarter-wave L layer
        lam_c = 0.5 * (w0 + w1)
        d_qw = lam_c / (4.0 * nL)
        best_overall_d = np.array([d_qw])
        best_overall_labels = ['L']
        best_overall_f = compute_objective(best_overall_d, np.array([nL]))

    # Post-process: remove zero-thickness layers
    labels = []
    thicks = []
    for i in range(len(best_overall_labels)):
        if best_overall_d[i] > 1e-6:
            labels.append(best_overall_labels[i])
            thicks.append(float(best_overall_d[i]))
    
    if not labels:
        labels = ['L']
        thicks = [lam_c / (4.0 * nL)]
        
    return labels, thicks
# EVOLVE-BLOCK-END
