"""AR coating design with gradient-based optimization and simulated annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _reflectance_matrix(mats, thicknesses, problem):
    """Compute reflectance at all 401 wavelengths using characteristic matrix method.
    Returns (R_array, M00, M01, M10, M11, wls) for gradient computation."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    
    n_layers = len(mats)
    if n_layers == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        R = np.full(401, abs(r)**2 * 100.0)
        return R, None, None, None, None, wls
    
    n_mat = np.array([n_L if m == 'L' else n_H for m in mats])
    d = np.array(thicknesses, dtype=float)
    
    M00 = np.ones(401, dtype=complex)
    M01 = np.zeros(401, dtype=complex)
    M10 = np.zeros(401, dtype=complex)
    M11 = np.ones(401, dtype=complex)
    
    for i in range(n_layers):
        n_i = n_mat[i]
        d_i = d[i]
        if d_i == 0:
            continue
        phase = 2.0 * np.pi * n_i * d_i / wls
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        
        B00 = cos_p
        B01 = 1j * sin_p / n_i
        B10 = 1j * n_i * sin_p
        B11 = cos_p
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y = M01 / M11
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r)**2 * 100.0
    return R, M00, M01, M10, M11, wls

def _mean_reflectance(mats, thicknesses, problem):
    """Compute mean reflectance (percent)."""
    R, _, _, _, _, _ = _reflectance_matrix(mats, thicknesses, problem)
    return float(np.mean(R))

def _objective_and_gradient(mats, thicknesses, problem):
    """Compute objective and gradient using complex-step differentiation."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    
    n_layers = len(mats)
    if n_layers == 0:
        return 4.26, np.array([])
    
    n_mat = np.array([n_L if m == 'L' else n_H for m in mats])
    d = np.array(thicknesses, dtype=float)
    
    def _compute_R(d_arr):
        M00 = np.ones(401, dtype=complex)
        M01 = np.zeros(401, dtype=complex)
        M10 = np.zeros(401, dtype=complex)
        M11 = np.ones(401, dtype=complex)
        
        for i in range(n_layers):
            n_i = n_mat[i]
            d_i = d_arr[i]
            if d_i == 0:
                continue
            phase = 2.0 * np.pi * n_i * d_i / wls
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            
            B00 = cos_p
            B01 = 1j * sin_p / n_i
            B10 = 1j * n_i * sin_p
            B11 = cos_p
            
            new_M00 = M00 * B00 + M01 * B10
            new_M01 = M00 * B01 + M01 * B11
            new_M10 = M10 * B00 + M11 * B10
            new_M11 = M10 * B01 + M11 * B11
            
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
        
        Y = M01 / M11
        r = (n_amb - Y) / (n_amb + Y)
        return np.abs(r)**2 * 100.0
    
    # Objective
    R = _compute_R(d)
    obj = float(np.mean(R))
    
    # Gradient via complex-step differentiation
    h = 1e-20
    grad = np.zeros(n_layers)
    for i in range(n_layers):
        d_plus = d.copy()
        d_plus[i] += h
        R_plus = _compute_R(d_plus)
        # Complex step: derivative = Im(f(z+h))/h
        # But R is real-valued, so use forward difference for reliability
        # Actually, let's use central difference for accuracy
        d_minus = d.copy()
        d_minus[i] -= h
        R_minus = _compute_R(d_minus)
        grad[i] = (np.mean(R_plus) - np.mean(R_minus)) / (2.0 * h)
    
    return obj, grad

def _optimize_thicknesses(mats, x0, problem, deadline):
    """Optimize thicknesses using L-BFGS-B with numerical gradient."""
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    n_layers = len(mats)
    
    def obj(x):
        if time.time() > deadline - 3:
            return 100.0
        x = np.clip(x, 0, max_layer_nm)
        if np.sum(x) > max_total_nm:
            x = x * (max_total_nm / np.sum(x))
        return _mean_reflectance(mats, list(x), problem)
    
    def grad(x):
        if time.time() > deadline - 3:
            return np.zeros(n_layers)
        x = np.clip(x, 0, max_layer_nm)
        if np.sum(x) > max_total_nm:
            x = x * (max_total_nm / np.sum(x))
        _, g = _objective_and_gradient(mats, list(x), problem)
        return g
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                      bounds=[(0, max_layer_nm)] * n_layers,
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxls': 50})
        x_opt = np.clip(res.x, 0, max_layer_nm)
        if np.sum(x_opt) > max_total_nm:
            x_opt = x_opt * (max_total_nm / np.sum(x_opt))
        score = _mean_reflectance(mats, list(x_opt), problem)
        if score < 100.0:
            return list(x_opt), score
    except Exception:
        pass
    
    # Fallback: coordinate descent
    return _coordinate_descent(mats, x0, problem, deadline)

def _coordinate_descent(mats, x0, problem, deadline, max_iter=40):
    """Coordinate descent with golden section search."""
    n_layers = len(mats)
    x = np.array(x0, dtype=float)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    current_obj = _mean_reflectance(mats, list(x), problem)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(n_layers):
            if time.time() > deadline - 5:
                break
            
            best_val = x[i]
            best_obj = current_obj
            
            # Golden section search
            a = 0.0
            b = max_layer_nm
            # Ensure constraint: sum of others + x[i] <= max_total_nm
            others_sum = np.sum(x) - x[i]
            b = min(b, max_total_nm - others_sum)
            if b <= a:
                continue
            
            invphi = (np.sqrt(5.0) - 1.0) / 2.0
            invphi2 = invphi * invphi
            
            c = b - invphi * (b - a)
            d = a + invphi * (b - a)
            
            fc = _mean_reflectance(mats, _set_coord(x, i, c), problem)
            fd = _mean_reflectance(mats, _set_coord(x, i, d), problem)
            
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - invphi * (b - a)
                    fc = _mean_reflectance(mats, _set_coord(x, i, c), problem)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + invphi * (b - a)
                    fd = _mean_reflectance(mats, _set_coord(x, i, d), problem)
            
            best_val = (a + b) / 2.0
            best_obj = _mean_reflectance(mats, _set_coord(x, i, best_val), problem)
            
            if best_obj < best_obj and best_obj < current_obj:
                x[i] = best_val
                current_obj = best_obj
                improved = True
        
        if not improved:
            break
    
    x = np.clip(x, 0, max_layer_nm)
    if np.sum(x) > max_total_nm:
        x = x * (max_total_nm / np.sum(x))
    final_obj = _mean_reflectance(mats, list(x), problem)
    return list(x), final_obj

def _set_coord(x, i, val):
    """Return copy of x with coordinate i set to val."""
    x_new = x.copy()
    x_new[i] = val
    return x_new

def _generate_structures(max_layers):
    """Generate diverse candidate material sequences."""
    structures = set()
    
    # Quarter-wave alternating stacks
    for n in range(1, max_layers + 1):
        structures.add(tuple('L' if i % 2 == 0 else 'H' for i in range(n)))
        structures.add(tuple('H' if i % 2 == 0 else 'L' for i in range(n)))
    
    # Pairs
    for n_pairs in range(1, max_layers // 2 + 1):
        for pair in ['LH', 'HL']:
            s = tuple(pair * n_pairs)
            if len(s) <= max_layers:
                structures.add(s)
    
    # Triplets
    for n_trip in range(1, max_layers // 3 + 1):
        for triplet in ['LHL', 'HLH', 'LLH', 'HLL', 'LHH', 'HHL']:
            s = tuple(triplet * n_trip)
            if len(s) <= max_layers:
                structures.add(s)
    
    # Some random sequences
    rng = np.random.RandomState(42)
    for _ in range(100):
        n = rng.randint(1, max_layers + 1)
        s = tuple(rng.choice(['L', 'H'], size=n))
        structures.add(s)
    
    return [list(s) for s in structures]

def _simulated_annealing(mats, thick, problem, deadline, n_iter=500):
    """Simulated annealing on material sequence."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * sum(problem["band_nm"])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    current_mats = list(mats)
    current_thick = list(thick)
    current_score = _mean_reflectance(current_mats, current_thick, problem)
    
    best_mats = current_mats[:]
    best_thick = current_thick[:]
    best_score = current_score
    
    T = 5.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(n_iter, 1))
    
    for it in range(n_iter):
        if time.time() > deadline - 10:
            break
        T *= cooling
        
        # Random move: flip a layer, add a layer, or remove a layer
        move_type = np.random.choice(3)
        
        if move_type == 0 and len(current_mats) > 0:
            # Flip a random layer
            i = np.random.randint(len(current_mats))
            new_mats = current_mats[:]
            new_mats[i] = 'L' if new_mats[i] == 'H' else 'H'
            new_thick = current_thick[:]
        elif move_type == 1 and len(current_mats) < max_layers:
            # Add a layer
            i = np.random.randint(len(current_mats) + 1)
            new_mats = current_mats[:]
            new_mats.insert(i, np.random.choice(['L', 'H']))
            new_thick = current_thick[:]
            new_thick.insert(i, centre / (4.0 * n_L))
        elif move_type == 2 and len(current_mats) > 1:
            # Remove a layer
            i = np.random.randint(len(current_mats))
            new_mats = current_mats[:]
            new_mats.pop(i)
            new_thick = current_thick[:]
            new_thick.pop(i)
        else:
            continue
        
        # Quick evaluation
        if sum(new_thick) > max_total_nm:
            continue
        if any(t > max_layer_nm for t in new_thick):
            continue
        
        new_score = _mean_reflectance(new_mats, new_thick, problem)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_mats = new_mats
            current_thick = new_thick
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_mats = current_mats[:]
                best_thick = current_thick[:]
    
    return best_mats, best_thick, best_score

def design(problem):
    """Multi-layer AR coating design with gradient-based optimization."""
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
    
    # Fallback: single quarter-wave L layer
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_reflectance(best_mats, best_thick, problem)
    
    # Try single H layer
    thick_h = [centre / (4.0 * n_H)]
    score_h = _mean_reflectance(["H"], thick_h, problem)
    if score_h < best_score:
        best_score = score_h
        best_mats = ["H"]
        best_thick = thick_h
    
    # Generate candidate structures
    candidates = _generate_structures(max_layers)
    
    # Sort by length (shorter first)
    candidates.sort(key=lambda s: (len(s), s[0]))
    
    # Limit candidates
    max_candidates = 60
    candidates = candidates[:max_candidates]
    
    # Optimize each candidate
    for mats in candidates:
        if time.time() > deadline - 30:
            break
        
        n_layers = len(mats)
        # Initial guess: quarter-wave at center wavelength
        x0 = np.array([centre / (4.0 * (n_L if m == 'L' else n_H)) for m in mats])
        
        # Optimize thicknesses
        thick_opt, score = _optimize_thicknesses(mats, x0, problem, deadline)
        
        if thick_opt is not None and score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thick = thick_opt[:]
    
    # Simulated annealing on the best structure
    if time.time() < deadline - 20 and len(best_mats) >= 2:
        sa_mats, sa_thick, sa_score = _simulated_annealing(
            best_mats, best_thick, problem, deadline, n_iter=300)
        if sa_score < best_score:
            best_score = sa_score
            best_mats = sa_mats[:]
            best_thick = sa_thick[:]
            
            # Refine with L-BFGS-B
            if time.time() < deadline - 10:
                thick_refined, score_refined = _optimize_thicknesses(
                    best_mats, best_thick, problem, deadline)
                if thick_refined is not None and score_refined < best_score:
                    best_score = score_refined
                    best_thick = thick_refined[:]
    
    # Final safety check
    if time.time() > deadline - 5:
        if any(t < 0 or t > max_layer_nm for t in best_thick):
            best_thick = [max(0, min(max_layer_nm, t)) for t in best_thick]
        if sum(best_thick) > max_total_nm:
            scale = max_total_nm / sum(best_thick)
            best_thick = [t * scale for t in best_thick]
    
    return best_mats, best_thick
# EVOLVE-BLOCK-END
