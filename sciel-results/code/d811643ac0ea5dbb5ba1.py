"""Broadband antireflection coating design via simulated annealing on (labels, thicknesses)."""
import time
import numpy as np
import random

def _mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Compute mean reflectance in percent over 401 equally spaced wavelengths."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r) ** 2 * 100.0
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    
    # Vectorized computation over wavelengths
    total_r = 0.0
    n_wav = len(wavelengths)
    
    for start in range(0, n_wav, 100):
        end = min(start + 100, n_wav)
        lams = wavelengths[start:end]
        n_lam = len(lams)
        
        # Initialize M as identity for each wavelength
        M = np.zeros((n_lam, 2, 2), dtype=complex)
        M[:, 0, 0] = 1.0
        M[:, 1, 1] = 1.0
        
        for i in range(len(labels)):
            delta = 2.0 * np.pi * n_mat[i] * d[i] / lams
            c = np.cos(delta)
            s = np.sin(delta)
            
            Mi_00 = c
            Mi_01 = 1j * s / n_mat[i]
            Mi_10 = 1j * n_mat[i] * s
            Mi_11 = c
            
            M_new = np.zeros_like(M)
            M_new[:, 0, 0] = M[:, 0, 0] * Mi_00 + M[:, 0, 1] * Mi_10
            M_new[:, 0, 1] = M[:, 0, 0] * Mi_01 + M[:, 0, 1] * Mi_11
            M_new[:, 1, 0] = M[:, 1, 0] * Mi_00 + M[:, 1, 1] * Mi_10
            M_new[:, 1, 1] = M[:, 1, 0] * Mi_01 + M[:, 1, 1] * Mi_11
            M = M_new
        
        Y = M[:, 1, 0] / M[:, 0, 0]
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2 * 100.0
        total_r += np.sum(R)
    
    return total_r / n_wav


def design(problem):
    """Design a broadband antireflection coating stack using simulated annealing."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    random.seed(42)
    np.random.seed(42)
    
    def is_valid(labels, thicknesses):
        if len(labels) > max_layers:
            return False
        if len(labels) != len(thicknesses):
            return False
        for t in thicknesses:
            if t < -1e-9 or t > max_layer_nm + 1e-9:
                return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        return True
    
    def clamp_state(labels, thicknesses):
        """Clamp thicknesses to valid range."""
        thicknesses = [max(0.0, min(t, max_layer_nm)) for t in thicknesses]
        total = sum(thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicknesses = [t * scale for t in thicknesses]
        return labels, thicknesses
    
    def evaluate(labels, thicknesses):
        if not is_valid(labels, thicknesses):
            return float('inf')
        return _mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm)
    
    # Generate diverse initial solutions
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    def make_initial_solution():
        """Create a random valid initial solution."""
        n_layers = random.randint(1, min(8, max_layers))
        labels = [random.choice(['L', 'H']) for _ in range(n_layers)]
        
        # Ensure alternation is somewhat preferred but not strict
        # Randomize with some structure
        if random.random() < 0.5:
            # Alternating pattern
            start_char = random.choice(['L', 'H'])
            labels = [start_char if i % 2 == 0 else ('H' if start_char == 'L' else 'L') 
                     for i in range(n_layers)]
        
        # Initial thicknesses: quarter-wave at center with some randomness
        thicknesses = []
        for l in labels:
            n_mat = n_L if l == 'L' else n_H
            base = centre / (4.0 * n_mat)
            # Add random perturbation
            t = base * random.uniform(0.5, 1.5)
            t = max(0.0, min(t, max_layer_nm))
            thicknesses.append(t)
        
        # Ensure total thickness constraint
        total = sum(thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicknesses = [t * scale for t in thicknesses]
        
        return labels, thicknesses
    
    # Start with a few well-known configurations
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Try some structured initial guesses
    initial_configs = [
        (["L"], [centre / (4.0 * n_L)]),
        (["H"], [centre / (4.0 * n_H)]),
        (["L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]),
    ]
    
    for labels, thicknesses in initial_configs:
        if is_valid(labels, thicknesses):
            score = evaluate(labels, thicknesses)
            if score < best_score:
                best_score = score
                best_labels = labels[:]
                best_thicknesses = thicknesses[:]
    
    # Also try some random initial solutions
    for _ in range(20):
        labels, thicknesses = make_initial_solution()
        score = evaluate(labels, thicknesses)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    # Simulated annealing
    current_labels = best_labels[:]
    current_thicknesses = best_thicknesses[:]
    current_score = best_score
    
    # Annealing parameters
    T_start = 5.0  # Initial temperature (in percent reflectance units)
    T_end = 0.01
    cooling_rate = 0.995  # Geometric cooling
    max_iter = 5000
    
    T = T_start
    
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Generate a neighbor
        neighbor_labels, neighbor_thicknesses = current_labels[:], current_thicknesses[:]
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Perturb a random thickness
            if len(neighbor_thicknesses) > 0:
                idx = random.randint(0, len(neighbor_thicknesses) - 1)
                delta = random.gauss(0, max_layer_nm * 0.1)
                neighbor_thicknesses[idx] += delta
                neighbor_thicknesses[idx] = max(0.0, min(neighbor_thicknesses[idx], max_layer_nm))
                
                # Check total thickness
                total = sum(neighbor_thicknesses)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    neighbor_thicknesses = [t * scale for t in neighbor_thicknesses]
        
        elif move_type < 0.6:
            # Change a random layer's material
            if len(neighbor_labels) > 0:
                idx = random.randint(0, len(neighbor_labels) - 1)
                neighbor_labels[idx] = 'H' if neighbor_labels[idx] == 'L' else 'L'
        
        elif move_type < 0.75:
            # Add a layer
            if len(neighbor_labels) < max_layers:
                pos = random.randint(0, len(neighbor_labels))
                new_layer = random.choice(['L', 'H'])
                n_mat = n_L if new_layer == 'L' else n_H
                new_thickness = centre / (4.0 * n_mat) * random.uniform(0.5, 1.0)
                new_thickness = max(0.0, min(new_thickness, max_layer_nm))
                
                neighbor_labels.insert(pos, new_layer)
                neighbor_thicknesses.insert(pos, new_thickness)
                
                # Check total thickness
                total = sum(neighbor_thicknesses)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    neighbor_thicknesses = [t * scale for t in neighbor_thicknesses]
        
        elif move_type < 0.85:
            # Remove a layer
            if len(neighbor_labels) > 1:
                idx = random.randint(0, len(neighbor_labels) - 1)
                neighbor_labels.pop(idx)
                neighbor_thicknesses.pop(idx)
        
        else:
            # Swap two layers
            if len(neighbor_labels) >= 2:
                i, j = random.sample(range(len(neighbor_labels)), 2)
                neighbor_labels[i], neighbor_labels[j] = neighbor_labels[j], neighbor_labels[i]
                neighbor_thicknesses[i], neighbor_thicknesses[j] = neighbor_thicknesses[j], neighbor_thicknesses[i]
        
        # Evaluate neighbor
        neighbor_score = evaluate(neighbor_labels, neighbor_thicknesses)
        
        # Acceptance criterion
        if neighbor_score < current_score:
            # Always accept better solutions
            current_labels = neighbor_labels
            current_thicknesses = neighbor_thicknesses
            current_score = neighbor_score
            
            if current_score < best_score:
                best_score = current_score
                best_labels = current_labels[:]
                best_thicknesses = current_thicknesses[:]
        else:
            # Metropolis criterion: accept with probability exp(-delta/T)
            delta = neighbor_score - current_score
            if delta < 0:
                # This shouldn't happen since we already checked neighbor_score < current_score
                current_labels = neighbor_labels
                current_thicknesses = neighbor_thicknesses
                current_score = neighbor_score
                if current_score < best_score:
                    best_score = current_score
                    best_labels = current_labels[:]
                    best_thicknesses = current_thicknesses[:]
            else:
                # delta >= 0, accept with probability exp(-delta/T)
                if T > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_labels = neighbor_labels
                        current_thicknesses = neighbor_thicknesses
                        current_score = neighbor_score
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    # Final local optimization: Nelder-Mead on thicknesses for the best label sequence
    if time.time() < deadline - 20.0 and best_labels is not None:
        def nm_optimize(labels, x0, max_iter=100):
            """Nelder-Mead optimization for thicknesses."""
            n = len(labels)
            if n == 0:
                return x0, evaluate(labels, x0)
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Initial simplex
            simplex = [np.array(x0, dtype=float)]
            for i in range(n):
                perturbed = np.array(x0, dtype=float)
                step = 0.1 * max_layer_nm if x0[i] == 0 else 0.1 * x0[i]
                perturbed[i] += step
                perturbed = np.clip(perturbed, 0.0, max_layer_nm)
                simplex.append(perturbed)
            
            fvals = []
            for x in simplex:
                if np.sum(x) > max_total_nm:
                    fvals.append(float('inf'))
                else:
                    fvals.append(evaluate(labels, list(x)))
            
            for it in range(max_iter):
                if time.time() > deadline - 15.0:
                    break
                
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if abs(fvals[-1] - fvals[0]) < 1e-8:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, 0.0, max_layer_nm)
                if np.sum(xr) > max_total_nm:
                    fr = float('inf')
                else:
                    fr = evaluate(labels, list(xr))
                
                if fr < fvals[0]:
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, 0.0, max_layer_nm)
                    if np.sum(xe) > max_total_nm:
                        fe = float('inf')
                    else:
                        fe = evaluate(labels, list(xe))
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
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, 0.0, max_layer_nm)
                    if np.sum(xc) > max_total_nm:
                        fc = float('inf')
                    else:
                        fc = evaluate(labels, list(xc))
                    
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], 0.0, max_layer_nm)
                            if np.sum(simplex[i]) > max_total_nm:
                                fvals[i] = float('inf')
                            else:
                                fvals[i] = evaluate(labels, list(simplex[i]))
            
            best_idx = np.argmin(fvals)
            return list(simplex[best_idx]), fvals[best_idx]
        
        # Optimize thicknesses for the best label sequence
        opt_thicknesses, opt_score = nm_optimize(best_labels, best_thicknesses)
        if opt_score < best_score:
            best_thicknesses = opt_thicknesses
            best_score = opt_score
        
        # Also try optimizing a few nearby label sequences
        if time.time() < deadline - 15.0:
            for _ in range(5):
                if time.time() > deadline - 15.0:
                    break
                
                # Create a variant of the best label sequence
                variant_labels = best_labels[:]
                if len(variant_labels) > 0:
                    idx = random.randint(0, len(variant_labels) - 1)
                    variant_labels[idx] = 'H' if variant_labels[idx] == 'L' else 'L'
                
                # Initial thicknesses: quarter-wave
                centre = 0.5 * (band_nm[0] + band_nm[1])
                init_thick = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in variant_labels]
                total = sum(init_thick)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    init_thick = [t * scale for t in init_thick]
                
                if is_valid(variant_labels, init_thick):
                    opt_thick, opt_sc = nm_optimize(variant_labels, init_thick, max_iter=50)
                    if opt_sc < best_score:
                        best_labels = variant_labels[:]
                        best_thicknesses = opt_thick
                        best_score = opt_sc
    
    # Final constraint check
    if best_labels is None:
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
