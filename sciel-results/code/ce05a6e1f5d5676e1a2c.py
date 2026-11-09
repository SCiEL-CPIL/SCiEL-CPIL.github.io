"""
Broadband antireflection coating design: 
Represent the coating as a continuous refractive index profile defined by a piecewise-linear 
function with exactly 20 nodes. The physical thickness of each layer is derived by enforcing 
a constant optical path length (OD) across the entire stack to minimize dispersion-induced 
phase errors. Optimize the 20 continuous node values (effective indices between 1.38 and 2.35) 
using a multi-start Nelder-Mead approach.
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
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    
    N = 20  # Number of nodes/layers
    
    # 1. Define the continuous profile representation
    # We have N layers. We define N+1 node values n[0..N].
    # n[0] = n_0, n[N] = n_s.
    # The effective index of layer i (0-indexed, i in 0..N-1) is the average of node i and i+1.
    # n_eff[i] = 0.5 * (n[i] + n[i+1])
    
    # We optimize the internal nodes n[1] .. n[N-1] (18 variables).
    
    # 2. Mapping to discrete layers
    def profile_to_discrete(nodes):
        """
        nodes: array of length N+1 (n[0]..n[N])
        Returns: materials (list of 'L'/'H'), thicknesses (list of floats)
        """
        materials = []
        thicknesses = []
        
        # Calculate effective indices
        n_eff = np.zeros(N)
        for i in range(N):
            n_eff[i] = 0.5 * (nodes[i] + nodes[i+1])
        
        # Determine materials by thresholding
        # Midpoint between L and H
        mid = 0.5 * (n_L + n_H)
        for i in range(N):
            materials.append('L' if n_eff[i] < mid else 'H')
        
        # Determine thicknesses to enforce constant optical path length
        # Total optical path OD_total = max_total
        # OD_i = n_eff_i * d_i = OD_total
        # d_i = OD_total / n_eff_i
        # Check constraints: d_i <= max_layer
        # If d_i > max_layer, we have a problem. 
        # Since n_eff is between 1.38 and 2.35, and max_total=1000, 
        # max d_i = 1000/1.38 = 724 > 400. 
        # So we must constrain the profile such that n_eff is high enough, or handle scaling.
        # To strictly satisfy d_i <= max_layer, we need n_eff_i >= max_total / max_layer = 1000/400 = 2.5.
        # But max n_eff is 2.35. So it is IMPOSSIBLE to have constant OD = 1000 with all layers <= 400.
        
        # Alternative: Enforce constant OD, but allow total OD to vary? 
        # The prompt says "enforcing a constant optical path length (OD = n_eff * d) across the entire stack".
        # It doesn't specify the value of OD. We can choose OD.
        # Let OD be a variable? Or fixed?
        # If we fix OD, we have 18 variables for nodes + 1 for OD = 19 variables.
        # Let's optimize OD as well.
        # Constraints:
        # 1. d_i = OD / n_eff_i <= max_layer  => OD <= max_layer * n_eff_i for all i => OD <= max_layer * min(n_eff)
        # 2. sum(d_i) <= max_total => OD * sum(1/n_eff_i) <= max_total => OD <= max_total / sum(1/n_eff_i)
        
        # So for a given profile (nodes), the maximum valid OD is:
        # OD_max = min( max_layer * min(n_eff), max_total / sum(1/n_eff) )
        # We should probably use OD = OD_max to maximize the "quality" of the stack (more layers = better AR).
        # Using less OD might be suboptimal.
        
        min_n_eff = np.min(n_eff)
        sum_inv_n = np.sum(1.0 / n_eff)
        
        od_limit_1 = max_layer * min_n_eff
        od_limit_2 = max_total / sum_inv_n
        
        OD = min(od_limit_1, od_limit_2)
        
        # Calculate thicknesses
        d = OD / n_eff
        
        # Due to floating point, d might slightly exceed max_layer or sum(d) might exceed max_total.
        # Clamp and scale if necessary.
        d = np.clip(d, 0, max_layer)
        total_d = np.sum(d)
        if total_d > max_total:
            d = d * (max_total / total_d)
            d = np.clip(d, 0, max_layer)
            
        return materials, d.tolist()

    # 3. Objective function
    def objective(nodes_internal, OD_override=None):
        """
        nodes_internal: array of length N-1 (n[1]..n[N-1])
        Returns: mean reflectance
        """
        # Reconstruct full nodes array
        nodes = np.zeros(N + 1)
        nodes[0] = n_0
        nodes[N] = n_s
        nodes[1:N] = nodes_internal
        
        materials, thicknesses = profile_to_discrete(nodes)
        
        # Quick constraint check (should be handled in profile_to_discrete)
        # But let's be safe.
        if any(t < 0 or t > max_layer for t in thicknesses):
            return 100.0
        if sum(thicknesses) > max_total + 1e-6:
            return 100.0
            
        return _mean_reflectance(materials, thicknesses, problem)

    # 4. Nelder-Mead implementation
    def nelder_mead(f, x0, bounds, max_iters=200, deadline=None):
        """
        Simple Nelder-Mead.
        f: objective function
        x0: initial point
        bounds: list of (min, max) for each variable
        """
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex
        simplex = [np.array(x0)]
        for i in range(n):
            pt = np.array(x0)
            # Step size
            step = 0.1 * (bounds[i][1] - bounds[i][0])
            pt[i] += step
            pt = np.clip(pt, bounds[:,0], bounds[:,1])
            simplex.append(pt)
            
        scores = [f(simplex[i]) for i in range(n + 1)]
        
        it = 0
        while it < max_iters:
            if deadline and time.time() > deadline:
                break
            it += 1
            
            # Sort
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            # Check convergence
            if abs(scores[0] - scores[-1]) < 1e-7:
                break
                
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, bounds[:,0], bounds[:,1])
            fr = f(xr)
            
            if fr < scores[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, bounds[:,0], bounds[:,1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                # Reflection accepted
                simplex[-1] = xr
                scores[-1] = fr
            else:
                # Contraction
                if fr < scores[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, bounds[:,0], bounds[:,1])
                    fc = f(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break # Shrink
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, bounds[:,0], bounds[:,1])
                    fc = f(xc)
                    if fc < scores[-1]:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], bounds[:,0], bounds[:,1])
                            scores[i] = f(simplex[i])
                        if np.max(scores) - np.min(scores) < 1e-7:
                            break
                        
        return simplex[0], scores[0]

    # 5. Multi-start search
    best_score = 100.0
    best_nodes = None
    
    # Bounds for internal nodes
    # n_eff must be between n_L and n_H?
    # If nodes are between n_L and n_H, n_eff is too.
    # But nodes can go outside? 
    # If a node is > n_H, n_eff might be > n_H. 
    # The discrete material selection handles this.
    # However, if n_eff is very high, OD is limited by max_layer.
    # Let's restrict nodes to [n_L, n_H] to keep n_eff in range.
    lower_bound = n_L
    upper_bound = n_H
    bounds = [(lower_bound, upper_bound)] * (N - 1)
    
    # Initial guesses
    # 1. Linear gradient
    x_linear = np.linspace(n_0, n_s, N + 1)[1:N]
    # 2. Random starts
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        if i == 0:
            x0 = x_linear
        else:
            # Random start
            x0 = np.random.uniform(lower_bound, upper_bound, N - 1)
            
        # Run NM
        res_x, res_f = nelder_mead(objective, x0, bounds, max_iters=100, deadline=deadline)
        
        if res_f < best_score:
            best_score = res_f
            best_nodes = res_x
            
        # Periodic restart if stuck? 
        # For now, just do fixed number of starts.
        
    # 6. Finalize
    if best_nodes is None:
        # Fallback
        best_nodes = x_linear
        best_score = objective(best_nodes)
        
    nodes = np.zeros(N + 1)
    nodes[0] = n_0
    nodes[N] = n_s
    nodes[1:N] = best_nodes
    
    materials, thicknesses = profile_to_discrete(nodes)
    
    return materials, thicknesses
