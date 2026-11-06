# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _compute_reflectance_vec(materials, thicknesses, problem):
    """
    Compute mean reflectance in percent over the band using vectorized ops.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    
    total_thick = sum(thicknesses)
    if total_thick > problem["max_total_nm"] + 1e-6:
        return float('inf')
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-6 for t in thicknesses):
        return float('inf')
    
    n_vals = np.array([n_dict[m] for m in materials])
    d_vals = np.array(thicknesses)
    
    # Vectorize over wavelengths: shape (n_waves, n_layers)
    # delta[i,j] = 2*pi*n_j*d_j/lam_i
    deltas = 2.0 * np.pi * n_vals[None, :] * d_vals[None, :] / lambdas[:, None]
    cos_d = np.cos(deltas)
    sin_d = np.sin(deltas)
    
    # Build matrices layer by layer (sequential, but vectorized over wavelengths)
    # M_total = M_1 * M_2 * ... * M_n
    # Each M_i has shape (n_waves, 2, 2)
    A = np.ones(n_waves, dtype=complex)
    B = np.zeros(n_waves, dtype=complex)
    C = np.zeros(n_waves, dtype=complex)
    D = np.ones(n_waves, dtype=complex)
    
    for i in range(n_layers):
        n_i = n_vals[i]
        # Layer matrix elements
        M11 = cos_d[:, i]
        M12 = 1j * sin_d[:, i] / n_i
        M21 = 1j * n_i * sin_d[:, i]
        M22 = cos_d[:, i]
        
        # M_new = M_old * M_layer (element-wise for each wavelength)
        nA = A * M11 + B * M21
        nB = A * M12 + B * M22
        nC = C * M11 + D * M21
        nD = C * M12 + D * M22
        
        A, B, C, D = nA, nB, nC, nD
    
    Y_in = (C + D * ns) / (A + B * ns)
    r = (n0 - Y_in) / (n0 + Y_in)
    R = np.abs(r) ** 2
    
    return 100.0 * np.mean(R)

def _eval_individual(materials, thicknesses, problem):
    """Evaluate a single individual."""
    try:
        return _compute_reflectance_vec(materials, thicknesses, problem)
    except:
        return float('inf')

def design(problem):
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Initialize with smart starting points
    # Quarter-wave stack: thickness = lambda/4 at center wavelength
    center_lam = 600.0
    
    def make_qw_stack(n_layers):
        """Create a quarter-wave stack."""
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L')
                thicks.append(center_lam / (4 * nL))
            else:
                mats.append('H')
                thicks.append(center_lam / (4 * nH))
        return mats, thicks
    
    # Generate diverse initial population
    population = []
    
    # Add some quarter-wave stacks
    for n in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        mats, thicks = make_qw_stack(n)
        # Scale to fit constraint
        total = sum(thicks)
        if total > max_total_nm:
            factor = max_total_nm / total
            thicks = [t * factor for t in thicks]
        population.append((mats, thicks))
    
    # Add random individuals
    while len(population) < 30:
        n_layers = random.randint(2, max_layers)
        mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
        # Random thicknesses
        weights = np.random.random(n_layers)
        total_w = weights.sum()
        target = random.uniform(0.5 * max_total_nm, max_total_nm)
        thicks = (weights / total_w * target).tolist()
        thicks = [min(t, max_layer_nm) for t in thicks]
        s = sum(thicks)
        if s > max_total_nm:
            factor = max_total_nm / s
            thicks = [t * factor for t in thicks]
        population.append((mats, thicks))
    
    # Evaluate initial population
    scores = [_eval_individual(m, t, problem) for m, t in population]
    
    best_idx = np.argmin(scores)
    best_mats = population[best_idx][0]
    best_thicks = population[best_idx][1]
    best_score = scores[best_idx]
    
    # Differential Evolution
    pop_size = len(population)
    F = 0.8  # Mutation scale
    CR = 0.9  # Crossover rate
    max_generations = 80
    
    for gen in range(max_generations):
        new_population = []
        new_scores = []
        
        for i in range(pop_size):
            # Select three distinct individuals
            a, b, c = random.sample(range(pop_size), 3)
            
            # Mutation: v = x_a + F * (x_b - x_c)
            # Handle variable length by using the max length
            max_len = max(len(population[a][0]), len(population[b][0]), len(population[c][0]))
            
            # Pad shorter ones with zeros (they'll be clipped)
            def pad_mats(mats, length):
                if len(mats) >= length:
                    return mats[:length]
                return mats + ['L'] * (length - len(mats))
            
            def pad_thicks(thicks, length):
                if len(thicks) >= length:
                    return thicks[:length]
                return thicks + [0.0] * (length - len(thicks))
            
            mats_a = pad_mats(population[a][0], max_len)
            mats_b = pad_mats(population[b][0], max_len)
            mats_c = pad_mats(population[c][0], max_len)
            thicks_a = pad_thicks(population[a][1], max_len)
            thicks_b = pad_thicks(population[b][1], max_len)
            thicks_c = pad_thicks(population[c][1], max_len)
            
            # For materials, we can't do arithmetic, so we use a different mutation strategy
            # Use the material from the best of the three or random
            if random.random() < 0.5:
                trial_mats = list(mats_a)
            else:
                trial_mats = list(mats_b)
            
            # For thicknesses, do DE mutation
            trial_thicks = []
            for j in range(max_len):
                if random.random() < CR:
                    trial_thicks.append(thicks_a[j] + F * (thicks_b[j] - thicks_c[j]))
                else:
                    trial_thicks.append(thicks_a[j])
            
            # Clip thicknesses
            trial_thicks = [max(0.0, min(t, max_layer_nm)) for t in trial_thicks]
            
            # Remove zero-thickness layers
            trial_mats = [trial_mats[j] for j in range(max_len) if trial_thicks[j] > 0.1]
            trial_thicks = [trial_thicks[j] for j in range(max_len) if trial_thicks[j] > 0.1]
            
            if not trial_mats:
                trial_mats = ['L']
                trial_thicks = [100.0]
            
            # Ensure total thickness constraint
            total = sum(trial_thicks)
            if total > max_total_nm:
                factor = max_total_nm / total
                trial_thicks = [t * factor for t in trial_thicks]
            
            # Evaluate
            trial_score = _eval_individual(trial_mats, trial_thicks, problem)
            
            # Selection
            if trial_score <= scores[i]:
                new_population.append((trial_mats, trial_thicks))
                new_scores.append(trial_score)
            else:
                new_population.append(population[i])
                new_scores.append(scores[i])
        
        population = new_population
        scores = new_scores
        
        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_mats = population[current_best_idx][0]
            best_thicks = population[current_best_idx][1]
    
    # Local refinement with Nelder-Mead on the best individual
    # Only optimize thicknesses, keep material structure fixed
    def objective(thick_vec):
        thick_vec = np.clip(thick_vec, 0, max_layer_nm)
        total = np.sum(thick_vec)
        if total > max_total_nm:
            return float('inf')
        return _compute_reflectance_vec(best_mats, thick_vec.tolist(), problem)
    
    x0 = np.array(best_thicks, dtype=float)
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 0.1, 'fatol': 1e-4})
        if result.fun < best_score:
            best_thicks = np.clip(result.x, 0, max_layer_nm).tolist()
            total = sum(best_thicks)
            if total > max_total_nm:
                factor = max_total_nm / total
                best_thicks = [t * factor for t in best_thicks]
            best_score = _eval_individual(best_mats, best_thicks, problem)
    except:
        pass
    
    # Try a few more local optimizations with different initial structures
    for _ in range(5):
        # Pick a random individual from population
        idx = random.randint(0, pop_size - 1)
        trial_mats = list(population[idx][0])
        trial_thicks = list(population[idx][1])
        
        def obj2(thick_vec, mats=trial_mats):
            thick_vec = np.clip(thick_vec, 0, max_layer_nm)
            total = np.sum(thick_vec)
            if total > max_total_nm:
                return float('inf')
            return _compute_reflectance_vec(mats, thick_vec.tolist(), problem)
        
        x0 = np.array(trial_thicks, dtype=float)
        try:
            result = minimize(obj2, x0, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 0.1, 'fatol': 1e-4})
            if result.fun < best_score:
                trial_thicks = np.clip(result.x, 0, max_layer_nm).tolist()
                total = sum(trial_thicks)
                if total > max_total_nm:
                    factor = max_total_nm / total
                    trial_thicks = [t * factor for t in trial_thicks]
                new_score = _eval_individual(trial_mats, trial_thicks, problem)
                if new_score < best_score:
                    best_score = new_score
                    best_mats = trial_mats
                    best_thicks = trial_thicks
        except:
            pass
    
    return best_mats, best_thicks
# EVOLVE-BLOCK-END
