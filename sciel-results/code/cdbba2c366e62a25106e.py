"""AR coating design using simulated annealing on layer sequences with local thickness optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _compute_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrix method."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses, dtype=float)
    total_R = 0.0
    for wl in wls:
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for i in range(len(materials)):
            n_i = n_mat[i]
            phase = 2.0 * np.pi * n_i * d[i] / wl
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            B = np.array([[cos_p, 1j * sin_p / n_i],
                          [1j * n_i * sin_p, cos_p]], dtype=complex)
            M = M @ B
        Y = M[0, 1] / M[1, 1]
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2
        total_R += R * 100.0
    return total_R / len(wls)

def _optimize_thicknesses(mats, problem, t0, deadline):
    """Optimize thicknesses for a fixed material sequence using L-BFGS-B."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    n_layers = len(mats)
    if n_layers == 0:
        return None, float('inf')
    
    # Initial guess: quarter-wave at center wavelength
    x0 = np.array([centre / (4.0 * (n_L if m == 'L' else n_H)) for m in mats])
    
    def obj(x):
        if time.time() > deadline:
            return 1000.0
        if np.any(x < 0) or np.any(x > max_layer_nm):
            return 1000.0
        if np.sum(x) > max_total_nm:
            return 1000.0 + (np.sum(x) - max_total_nm) * 10
        return _compute_reflectance(mats, list(x), problem)
    
    try:
        from scipy.optimize import minimize
        res = minimize(obj, x0, method='L-BFGS-B',
                      bounds=[(0, max_layer_nm)] * n_layers,
                      options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.fun < 100.0:
            x_opt = np.clip(res.x, 0, max_layer_nm)
            if np.sum(x_opt) > max_total_nm:
                scale = max_total_nm / np.sum(x_opt)
                x_opt = x_opt * scale
            score = _compute_reflectance(mats, list(x_opt), problem)
            return list(x_opt), score
    except Exception:
        pass
    
    return None, float('inf')

def _simulate_annealing(problem, t0, deadline):
    """Simulated annealing to find good layer sequences, with local thickness optimization."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    # Start with a simple quarter-wave stack
    best_mats = ["L", "H"] * 3  # 6 layers
    best_thick, best_score = _optimize_thicknesses(best_mats, problem, t0, deadline)
    
    if best_thick is None:
        best_mats = ["L"]
        best_thick = [centre / (4.0 * n_L)]
        best_score = _compute_reflectance(best_mats, best_thick, problem)
    
    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.1
    T_decay = 0.995
    T = T_start
    
    # Cache for evaluated sequences
    seq_cache = {}
    
    def get_score(mats):
        """Get score for a material sequence, using cache if available."""
        key = tuple(mats)
        if key in seq_cache:
            return seq_cache[key]
        
        if len(mats) > max_layers:
            return float('inf')
        
        thick, score = _optimize_thicknesses(mats, problem, t0, deadline)
        if thick is None:
            return float('inf')
        
        seq_cache[key] = score
        return score
    
    current_mats = best_mats[:]
    current_score = best_score
    
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        # Generate a neighbor
        new_mats = current_mats[:]
        move_type = random.choice(['add', 'remove', 'swap', 'replace'])
        
        if move_type == 'add' and len(new_mats) < max_layers:
            pos = random.randint(0, len(new_mats))
            new_mats.insert(pos, random.choice(['L', 'H']))
        elif move_type == 'remove' and len(new_mats) > 1:
            pos = random.randint(0, len(new_mats) - 1)
            new_mats.pop(pos)
        elif move_type == 'swap' and len(new_mats) >= 2:
            i, j = random.sample(range(len(new_mats)), 2)
            new_mats[i], new_mats[j] = new_mats[j], new_mats[i]
        elif move_type == 'replace' and len(new_mats) > 0:
            pos = random.randint(0, len(new_mats) - 1)
            new_mats[pos] = 'L' if new_mats[pos] == 'H' else 'H'
        
        # Evaluate neighbor
        new_score = get_score(new_mats)
        
        # Accept or reject
        if new_score < current_score:
            current_mats = new_mats
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_mats = new_mats[:]
                best_thick, _ = _optimize_thicknesses(best_mats, problem, t0, deadline)
        else:
            delta = new_score - current_score
            if delta < 0:
                delta = 0
            accept_prob = np.exp(-delta / T)
            if random.random() < accept_prob:
                current_mats = new_mats
                current_score = new_score
        
        # Cool down
        T *= T_decay
    
    return best_mats, best_thick, best_score

def design(problem):
    """Design an AR coating using simulated annealing on layer sequences."""
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    # Fallback: single quarter-wave layer
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _compute_reflectance(best_mats, best_thick, problem)
    
    # Also try single H layer
    thick_h = [centre / (4.0 * n_H)]
    score_h = _compute_reflectance(["H"], thick_h, problem)
    if score_h < best_score:
        best_score = score_h
        best_mats = ["H"]
        best_thick = thick_h
    
    # Run simulated annealing
    sa_mats, sa_thick, sa_score = _simulate_annealing(problem, t0, deadline)
    
    if sa_thick is not None and sa_score < best_score:
        best_score = sa_score
        best_mats = sa_mats
        best_thick = sa_thick
    
    # Final refinement on the best design
    if time.time() < deadline - 5 and len(best_mats) >= 2:
        try:
            from scipy.optimize import minimize
            n_layers = len(best_mats)
            x0 = np.array(best_thick, dtype=float)
            
            def obj(x):
                if time.time() > deadline:
                    return 1000.0
                if np.any(x < 0) or np.any(x > max_layer_nm):
                    return 1000.0
                if np.sum(x) > max_total_nm:
                    return 1000.0 + (np.sum(x) - max_total_nm) * 10
                return _compute_reflectance(best_mats, list(x), problem)
            
            res = minimize(obj, x0, method='L-BFGS-B',
                          bounds=[(0, max_layer_nm)] * n_layers,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_score and res.fun < 100.0:
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm + 1e-9:
                    best_score = float(res.fun)
                    best_thick = list(x_opt)
        except Exception:
            pass
    
    return best_mats, best_thick
# EVOLVE-BLOCK-END
