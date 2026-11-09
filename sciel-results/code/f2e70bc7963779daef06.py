"""
Broadband antireflection coating design: Continuous index profile optimization with Differential Evolution.
"""
import time
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # delta_j = 2*pi*n_j*d_j/lambda -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    N = max_layers
    
    # Strategy:
    # 1. Define a continuous vector x of length N, where x[i] represents an "effective index" n_eff[i].
    # 2. Material is determined by thresholding n_eff: if n_eff < (n_L+n_H)/2, use L, else H.
    # 3. Thickness is determined by constant Optical Path (OD) constraint: n_eff[i] * d[i] = OD.
    #    d[i] = OD / n_eff[i].
    # 4. We optimize x (the n_eff values) and OD (the optical path length).
    #    Total thickness sum(d[i]) = OD * sum(1/n_eff[i]) <= max_total.
    #    Also d[i] <= max_layer.
    
    # To ensure constraints, we can parameterize the search space such that constraints are naturally satisfied or easily checked.
    # Let's optimize:
    #   z[i]: raw values for n_eff, mapped to [n_L, n_H]
    #   y: log(OD) or just OD, mapped to a valid range.
    
    # Valid OD range:
    # Min OD: If all n_eff are n_H (max index), d_i = OD/n_H. Sum d_i = N * OD/n_H <= max_total => OD <= max_total * n_H / N.
    # Also d_i <= max_layer => OD/n_H <= max_layer => OD <= max_layer * n_H.
    # Max OD: If all n_eff are n_L (min index), d_i = OD/n_L. Sum d_i = N * OD/n_L <= max_total => OD <= max_total * n_L / N.
    # Wait, if n_eff varies, sum(1/n_eff) varies.
    # Let's just optimize OD directly in a safe range and penalize violations heavily.
    
    # Range for n_eff: [n_L, n_H]
    # Range for OD: 
    #   Lower bound: very small (thin coating).
    #   Upper bound: To satisfy sum(OD/n_eff) <= max_total.
    #   Since 1/n_eff >= 1/n_H, sum(OD/n_eff) >= N * OD / n_H.
    #   So N * OD / n_H <= max_total => OD <= max_total * n_H / N.
    #   Let's set OD_max = max_total * n_H / N.
    #   Let's set OD_min = 1.0 nm (arbitrary small).
    
    OD_max = max_total * n_H / N
    OD_min = 1.0
    
    def decode(x):
        """
        x: array of length N+1.
        x[0:N] -> n_eff values in [n_L, n_H]
        x[N] -> OD value in [OD_min, OD_max]
        Returns materials, thicknesses
        """
        n_effs = x[:N]
        # Map n_effs to [n_L, n_H]
        # Assume x[0:N] are in [0, 1]
        n_effs_mapped = n_L + n_effs * (n_H - n_L)
        
        # Map OD
        OD = OD_min + x[N] * (OD_max - OD_min)
        
        materials = []
        thicknesses = []
        threshold = (n_L + n_H) / 2.0
        
        for i in range(N):
            n_eff = n_effs_mapped[i]
            if n_eff < threshold:
                mat = 'L'
                n_real = n_L
            else:
                mat = 'H'
                n_real = n_H
            
            d = OD / n_eff # Use n_eff for thickness calculation to maintain continuous mapping?
            # Actually, the physical thickness must correspond to the REAL material index for the physics.
            # The "n_eff" is just a proxy for the material choice.
            # So d = OD / n_real? 
            # If we use d = OD / n_real, then the optical path is n_real * d = OD.
            # This ensures constant optical path for the REAL material.
            # This is physically consistent.
            
            # Check constraints
            if d > max_layer:
                # Penalize or clamp? Clamping breaks the smoothness. 
                # Let's clamp and let the optimizer learn to avoid it.
                d = max_layer
            if d < 0:
                d = 0.0
                
            materials.append(mat)
            thicknesses.append(d)
            
        # Check total thickness
        total_th = sum(thicknesses)
        if total_th > max_total:
            # Scale down
            scale = max_total / total_th
            thicknesses = [t * scale for t in thicknesses]
            
        return materials, thicknesses

    def objective(x):
        materials, thicknesses = decode(x)
        return _mean_reflectance(materials, thicknesses, problem)

    # Differential Evolution
    pop_size = 50
    F = 0.8  # Mutation factor
    CR = 0.9 # Crossover probability
    
    # Initialize population
    # x in [0, 1]^(N+1)
    pop = np.random.rand(pop_size, N + 1)
    scores = np.array([objective(ind) for ind in pop])
    
    best_score = np.min(scores)
    best_x = pop[np.argmin(scores)]
    
    # DE Loop
    while time.time() < deadline:
        # Check convergence or time
        if np.std(scores) < 1e-6:
            break
            
        new_pop = np.empty_like(pop)
        new_pop[0] = pop[0]
        
        for i in range(1, pop_size):
            # Select 3 distinct random indices
            a, b, c = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            
            # Mutation
            v = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            r = np.random.rand()
            j_rand = np.random.randint(0, N + 1)
            cross_mask = (np.random.rand(N + 1) < CR)
            cross_mask[j_rand] = True
            
            u = np.where(cross_mask, v, pop[i])
            
            # Bounds [0, 1]
            u = np.clip(u, 0.0, 1.0)
            
            new_pop[i] = u
            
        # Evaluate new population (only the mutated ones, index 0 is kept)
        # We can reuse the old score for index 0
        new_scores = np.empty_like(scores)
        new_scores[0] = scores[0]
        
        # Batch evaluation for speed? 
        # objective is vectorized inside _mean_reflectance but called per individual.
        # We can loop.
        for i in range(1, pop_size):
            new_scores[i] = objective(new_pop[i])
            
        # Selection
        for i in range(pop_size):
            if new_scores[i] < scores[i]:
                pop[i] = new_pop[i]
                scores[i] = new_scores[i]
                
        if np.min(scores) < best_score:
            best_score = np.min(scores)
            best_x = pop[np.argmin(scores)]
            
    # Final Polish with Nelder-Mead on the best DE result
    def nelder_mead(x0, problem, deadline, max_iters=100):
        x = list(x0)
        n = len(x)
        
        # Initial simplex
        simplex = [list(x)]
        for i in range(n):
            pt = list(x)
            pt[i] += 0.01
            pt[i] = np.clip(pt[i], 0.0, 1.0)
            simplex.append(pt)
            
        scores = [objective(ind) for ind in simplex]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, 0.0, 1.0)
            fr = objective(xr)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, 0.0, 1.0)
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                simplex[-1] = xr
                scores[-1] = fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, 0.0, 1.0)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    scores[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = np.clip(simplex[i], 0.0, 1.0)
                        scores[i] = objective(simplex[i])
                    if np.max(scores) - np.min(scores) < 1e-7:
                        break
                        
        return simplex[0], scores[0]

    if time.time() < deadline - 5:
        final_x, final_score = nelder_mead(best_x, problem, deadline, max_iters=100)
        if final_score < best_score:
            best_score = final_score
            best_x = final_x

    # Decode best_x to final materials and thicknesses
    materials, thicknesses = decode(best_x)
    
    # Final sanity check and constraint enforcement
    total_th = sum(thicknesses)
    if total_th > max_total:
        scale = max_total / total_th
        thicknesses = [t * scale for t in thicknesses]
    for i in range(len(thicknesses)):
        if thicknesses[i] > max_layer:
            thicknesses[i] = max_layer
        if thicknesses[i] < 0:
            thicknesses[i] = 0.0
            
    return materials, thicknesses
