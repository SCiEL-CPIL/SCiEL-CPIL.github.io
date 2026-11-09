"""Rewritten program: multi-start local search + simulated annealing for broadband AR coating."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (%) over the band using characteristic matrix method."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    n_wl = 401
    wl = np.linspace(band[0], band[1], n_wl)
    
    n_mat = np.array([problem["materials"][m] for m in materials])
    d = np.array(thicknesses, dtype=float)
    N = len(materials)
    
    # Phase delta for each layer at each wavelength: 2*pi*n*d*1000/lambda (lambda in nm, d in nm)
    # delta = 2*pi*n*d/lambda  (all in nm)
    delta = 2.0 * np.pi * n_mat[:, None] * d[:, None] / wl[None, :]  # shape (N, n_wl)
    
    # Characteristic matrix for each layer: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # We compute the product M = M1 * M2 * ... * MN for each wavelength
    # Use complex matrices
    
    # Initialize identity matrices for each wavelength
    M = np.ones((2, 2, n_wl), dtype=complex)
    
    for i in range(N):
        c = np.cos(delta[i])  # shape (n_wl,)
        s = np.sin(delta[i])  # shape (n_wl,)
        n_i = n_mat[i]
        
        # Layer matrix: [[c, 1j*s/n_i], [1j*n_i*s, c]]
        M_layer = np.ones((2, 2, n_wl), dtype=complex)
        M_layer[0, 0] = c
        M_layer[0, 1] = 1j * s / n_i
        M_layer[1, 0] = 1j * n_i * s
        M_layer[1, 1] = c
        
        # M = M_layer @ M  (matrix multiply for each wavelength)
        # M[0,0] = Ml[0,0]*M[0,0] + Ml[0,1]*M[1,0]
        # M[0,1] = Ml[0,0]*M[0,1] + Ml[0,1]*M[1,1]
        # M[1,0] = Ml[1,0]*M[0,0] + Ml[1,1]*M[1,0]
        # M[1,1] = Ml[1,0]*M[0,1] + Ml[1,1]*M[1,1]
        M00 = M_layer[0,0]*M[0,0] + M_layer[0,1]*M[1,0]
        M01 = M_layer[0,0]*M[0,1] + M_layer[0,1]*M[1,1]
        M10 = M_layer[1,0]*M[0,0] + M_layer[1,1]*M[1,0]
        M11 = M_layer[1,0]*M[0,1] + M_layer[1,1]*M[1,1]
        M[0,0] = M00
        M[0,1] = M01
        M[1,0] = M10
        M[1,1] = M11
    
    # Reflectance: R = |(M11 + M12*n0 - M21*ns - M22*ns*n0) / (M11 + M12*n0 + M21*ns + M22*ns*n0)|^2
    # Wait, the standard formula: For a stack between ambient n0 and substrate ns,
    # The characteristic matrix M relates [E, H] at air side to [E, H] at substrate side.
    # At substrate: H = (ns/n0) * E (for normal incidence, traveling wave into substrate)
    # Actually, the standard formula is:
    # Y = (M11*n0 + M12) / (M21*n0 + M22)  (admittance)
    # R = |(Y - ns)/(Y + ns)|^2
    # Or equivalently:
    # r = (M11 + M12/n0 - M21*ns - M22*ns/n0) / (M11 + M12/n0 + M21*ns + M22*ns/n0)
    # Hmm, let me use the standard thin-film formula.
    
    # The admittance Y = (M11 * n0 + M12) / (M21 * n0 + M22)
    # Then r = (Y - ns) / (Y + ns)
    # R = |r|^2
    
    Y = (M[0,0] * n0 + M[0,1]) / (M[1,0] * n0 + M[1,1])
    r = (Y - ns) / (Y + ns)
    R = np.abs(r)**2
    return np.mean(R) * 100.0


def _valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True


def _clamp_thicknesses(thicknesses, problem):
    """Clamp thicknesses to valid range and ensure total doesn't exceed max."""
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    clamped = [max(0.0, min(max_layer_nm, t)) for t in thicknesses]
    
    total = sum(clamped)
    if total > max_total_nm:
        scale = max_total_nm / total
        clamped = [t * scale for t in clamped]
    
    return clamped


def _random_stack(problem, n_layers=None):
    """Generate a random valid stack."""
    if n_layers is None:
        n_layers = np.random.randint(2, problem["max_layers"] + 1)
    
    materials = []
    thicknesses = []
    
    # Avoid adjacent same materials for better performance
    last = None
    for _ in range(n_layers):
        if last is None:
            m = np.random.choice(['L', 'H'])
        else:
            # Prefer alternating
            if np.random.random() < 0.7:
                m = 'H' if last == 'L' else 'L'
            else:
                m = np.random.choice(['L', 'H'])
        materials.append(m)
        thicknesses.append(np.random.uniform(10, problem["max_layer_nm"]))
        last = m
    
    thicknesses = _clamp_thicknesses(thicknesses, problem)
    return materials, thicknesses


def _optimize_local(materials, thicknesses, problem, deadline):
    """Run local optimization on a given stack."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    def objective(x):
        if time.time() > deadline:
            return 100.0
        t = _clamp_thicknesses(x, problem)
        if not _valid(materials, t, problem):
            return 100.0
        return _reflectance(materials, t, problem)
    
    x0 = np.array(thicknesses, dtype=float)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-4,
                                  'adaptive': True})
        if result.success or result.fun < objective(x0):
            t_opt = _clamp_thicknesses(result.x, problem)
            if _valid(materials, t_opt, problem):
                return materials, t_opt, result.fun
    except Exception:
        pass
    
    return materials, thicknesses, objective(x0)


def _simulated_annealing(problem, deadline, initial_materials=None, initial_thicknesses=None):
    """Simulated annealing to find good thicknesses."""
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    if initial_materials is None or initial_thicknesses is None:
        materials, thicknesses = _random_stack(problem)
    else:
        materials = list(initial_materials)
        thicknesses = list(initial_thicknesses)
    
    current_score = _reflectance(materials, thicknesses, problem)
    best_materials = list(materials)
    best_thicknesses = list(thicknesses)
    best_score = current_score
    
    T = 5.0  # Initial temperature
    cooling = 0.995
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            break
        
        # Random perturbation
        new_materials = list(materials)
        new_thicknesses = list(thicknesses)
        
        action = np.random.random()
        if action < 0.6:
            # Perturb a random thickness
            idx = np.random.randint(len(new_thicknesses))
            delta = np.random.normal(0, 20)
            new_thicknesses[idx] = max(0, min(max_layer_nm, new_thicknesses[idx] + delta))
        elif action < 0.8:
            # Change a random layer material
            idx = np.random.randint(len(new_materials))
            new_materials[idx] = 'H' if new_materials[idx] == 'L' else 'L'
        elif action < 0.9:
            # Add a layer
            if len(new_materials) < max_layers:
                pos = np.random.randint(len(new_materials) + 1)
                m = 'L' if np.random.random() < 0.5 else 'H'
                new_materials.insert(pos, m)
                new_thicknesses.insert(pos, np.random.uniform(20, 100))
        else:
            # Remove a layer
            if len(new_materials) > 2:
                pos = np.random.randint(len(new_materials))
                new_materials.pop(pos)
                new_thicknesses.pop(pos)
        
        new_thicknesses = _clamp_thicknesses(new_thicknesses, problem)
        if not _valid(new_materials, new_thicknesses, problem):
            continue
        
        new_score = _reflectance(new_materials, new_thicknesses, problem)
        
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
            materials = new_materials
            thicknesses = new_thicknesses
            current_score = new_score
            
            if new_score < best_score:
                best_materials = list(materials)
                best_thicknesses = list(thicknesses)
                best_score = new_score
    
    return best_materials, best_thicknesses, best_score


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial candidates: various multi-layer stacks
    candidates = []
    
    # Candidate 1: 4-layer alternating stack (L-H-L-H or H-L-H-L)
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    # Quarter-wave thicknesses at center wavelength
    qwL = centre / (4.0 * nL)
    qwH = centre / (4.0 * nH)
    
    # Try several initial patterns
    patterns = [
        (['L', 'H', 'L', 'H'], [qwL, qwH, qwL, qwH]),
        (['H', 'L', 'H', 'L'], [qwH, qwL, qwH, qwL]),
        (['L', 'H', 'L', 'H', 'L'], [qwL, qwH, qwL, qwH, qwL]),
        (['H', 'L', 'H', 'L', 'H'], [qwH, qwL, qwH, qwL, qwH]),
        (['L', 'H', 'L', 'H', 'L', 'H'], [qwL, qwH, qwL, qwH, qwL, qwH]),
    ]
    
    for mats, thicks in patterns:
        thicks = _clamp_thicknesses(thicks, problem)
        if _valid(mats, thicks, problem):
            score = _reflectance(mats, thicks, problem)
            candidates.append((score, mats, thicks))
    
    # Add random initializations
    for _ in range(5):
        if time.time() > deadline:
            break
        mats, thicks = _random_stack(problem)
        score = _reflectance(mats, thicks, problem)
        candidates.append((score, mats, thicks))
    
    # Sort by initial score
    candidates.sort(key=lambda x: x[0])
    
    best_materials = candidates[0][1] if candidates else ['L']
    best_thicknesses = candidates[0][2] if candidates else [qwL]
    best_score = candidates[0][0] if candidates else 100.0
    
    # Local optimization on top candidates
    for score, mats, thicks in candidates[:8]:
        if time.time() > deadline:
            break
        opt_mats, opt_thicks, opt_score = _optimize_local(mats, thicks, problem, deadline)
        if opt_score < best_score:
            best_materials = opt_mats
            best_thicknesses = opt_thicks
            best_score = opt_score
    
    # Simulated annealing with remaining time
    if time.time() < deadline - 10:
        sa_mats, sa_thicks, sa_score = _simulated_annealing(
            problem, deadline, best_materials, best_thicknesses)
        if sa_score < best_score:
            best_materials = sa_mats
            best_thicknesses = sa_thicks
            best_score = sa_score
        
        # Local polish after SA
        if time.time() < deadline - 5:
            opt_mats, opt_thicks, opt_score = _optimize_local(
                best_materials, best_thicknesses, problem, deadline)
            if opt_score < best_score:
                best_materials = opt_mats
                best_thicknesses = opt_thicks
                best_score = opt_score
    
    # Final validation
    best_thicknesses = _clamp_thicknesses(best_thicknesses, problem)
    if not _valid(best_materials, best_thicknesses, problem):
        # Fallback to simple valid design
        best_materials = ['L']
        best_thicknesses = [qwL]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
