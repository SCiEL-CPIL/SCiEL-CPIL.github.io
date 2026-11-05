# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so

def _eval_reflectance(layer_types, thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """
    Vectorized reflectance calculation.
    """
    n_sub = substrate_index
    n_amb = ambient_index
    num_layers = len(thicknesses)
    
    if num_layers == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r)**2

    # Wavelengths shape (W,)
    # M_total shape (W, 2, 2)
    M_total = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    for j in range(num_layers):
        n = materials[layer_types[j]]
        d = thicknesses[j]
        # delta = 2 * pi * n * d / lambda
        delta = 2.0 * np.pi * n * d / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)
        
        # M_layer shape (W, 2, 2)
        M_layer = np.zeros((len(wavelengths), 2, 2), dtype=complex)
        M_layer[:, 0, 0] = c
        M_layer[:, 0, 1] = 1j * s / n
        M_layer[:, 1, 0] = 1j * n * s
        M_layer[:, 1, 1] = c
        
        # M_total = M_total @ M_layer
        M_total = np.matmul(M_total, M_layer)
        
    # Y = M[1,0] / M[0,0]
    Y = M_total[:, 1, 0] / M_total[:, 0, 0]
    
    # r = (n_amb - Y) / (n_amb + Y)
    r = (n_amb - Y) / (n_amb + Y)
    reflectance = np.abs(r)**2
    
    return np.mean(reflectance)

def design(problem):
    """
    Design an antireflection coating using a hybrid GA and Local Optimization.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    def is_valid(lt, t):
        if len(lt) != len(t):
            return False
        if len(lt) > max_layers or len(lt) == 0:
            return False
        for i in range(1, len(lt)):
            if lt[i] == lt[i-1]:
                return False
        for th in t:
            if th < 0 or th > max_layer_nm:
                return False
        if sum(t) > max_total_nm:
            return False
        return True

    def score(ind):
        lt, t = ind
        if not is_valid(lt, t):
            return 1e6
        return _eval_reflectance(lt, t, materials, ambient_index, substrate_index, wavelengths) * 100.0

    def random_individual():
        num_layers = np.random.randint(1, max_layers + 1)
        start_type = np.random.choice(['L', 'H'])
        lt = []
        for i in range(num_layers):
            if i % 2 == 0:
                lt.append(start_type)
            else:
                lt.append('H' if start_type == 'L' else 'L')
        
        # Random thicknesses
        t = np.random.uniform(0, max_layer_nm, num_layers)
        # Scale to fit total
        total = t.sum()
        if total > max_total_nm:
            t = t * (max_total_nm / total)
        return lt, t.tolist()

    # Initialization
    population = []
    scores = []
    
    # Specific good starting points (Quarter wave at 550nm)
    lam_center = 550.0
    d_L = lam_center / (4.0 * materials['L'])
    d_H = lam_center / (4.0 * materials['H'])
    
    candidates = [
        (['L'], [d_L]),
        (['L', 'H'], [d_L, d_H]),
        (['H', 'L'], [d_H, d_L]),
        (['L', 'H', 'L', 'H'], [d_L, d_H, d_L, d_H]),
        (['H', 'L', 'H', 'L'], [d_H, d_L, d_H, d_L]),
        (['L', 'H', 'L', 'H', 'L', 'H'], [d_L, d_H, d_L, d_H, d_L, d_H]),
    ]
    
    for cand in candidates:
        if is_valid(cand[0], cand[1]):
            population.append(cand)
            scores.append(score(cand))

    # Fill rest with random
    while len(population) < 50:
        ind = random_individual()
        if is_valid(ind[0], ind[1]):
            population.append(ind)
            scores.append(score(ind))

    best_ind = population[np.argmin(scores)]
    best_score = min(scores)
    
    # GA Loop
    pop_size = 50
    generations = 100
    
    for gen in range(generations):
        # Sort
        indices = np.argsort(scores)
        population = [population[i] for i in indices]
        scores = [scores[i] for i in indices]
        
        best_ind = population[0]
        best_score = scores[0]
        
        new_pop = [population[0]] # Elitism
        
        while len(new_pop) < pop_size:
            # Tournament
            p1_idx = np.random.choice(len(population), 3, replace=False)
            p1 = population[p1_idx[np.argmin([scores[i] for i in p1_idx])]]
            p2_idx = np.random.choice(len(population), 3, replace=False)
            p2 = population[p2_idx[np.argmin([scores[i] for i in p2_idx])]]
            
            # Crossover
            if np.random.rand() < 0.7:
                 # Uniform crossover
                 lt1, t1 = p1
                 lt2, t2 = p2
                 max_len = max(len(lt1), len(lt2))
                 child_lt = []
                 child_t = []
                 for i in range(max_len):
                     if np.random.rand() < 0.5:
                         if i < len(lt1):
                             child_lt.append(lt1[i])
                             child_t.append(t1[i])
                     else:
                         if i < len(lt2):
                             child_lt.append(lt2[i])
                             child_t.append(t2[i])
                 
                 # Fix alternating pattern
                 fixed_lt = []
                 fixed_t = []
                 if len(child_lt) > 0:
                     fixed_lt.append(child_lt[0])
                     fixed_t.append(child_t[0])
                     for i in range(1, len(child_lt)):
                         if child_lt[i] != fixed_lt[-1]:
                             fixed_lt.append(child_lt[i])
                             fixed_t.append(child_t[i])
                 
                 # Truncate
                 if len(fixed_lt) > max_layers:
                     fixed_lt = fixed_lt[:max_layers]
                     fixed_t = fixed_t[:max_layers]
                 
                 # Scale thickness
                 if len(fixed_t) > 0:
                     total = sum(fixed_t)
                     if total > max_total_nm:
                         fixed_t = [x * (max_total_nm / total) for x in fixed_t]
                     fixed_t = [max(0, min(max_layer_nm, x)) for x in fixed_t]
                 
                 child = (fixed_lt, fixed_t)
            else:
                 child = (list(p1[0]), list(p1[1]))
                 
            # Mutation
            lt, t = child
            lt = list(lt)
            t = list(t)
            
            # Change layer count
            if np.random.rand() < 0.1:
                if len(lt) < max_layers and np.random.rand() < 0.5:
                    pos = np.random.randint(0, len(lt) + 1)
                    if pos == 0:
                        new_type = 'H' if lt[0] == 'L' else 'L'
                    else:
                        new_type = 'H' if lt[pos-1] == 'L' else 'L'
                    lt.insert(pos, new_type)
                    t.insert(pos, np.random.uniform(0, max_layer_nm))
                elif len(lt) > 1:
                    pos = np.random.randint(0, len(lt))
                    lt.pop(pos)
                    t.pop(pos)
            
            # Perturb thicknesses
            for i in range(len(t)):
                if np.random.rand() < 0.3:
                    t[i] = max(0, min(max_layer_nm, t[i] + np.random.normal(0, 20)))
            
            # Scale to fit
            total = sum(t)
            if total > max_total_nm and total > 0:
                t = [x * (max_total_nm / total) for x in t]
            
            child = (lt, t)
            
            if is_valid(child[0], child[1]):
                new_pop.append(child)
            else:
                new_pop.append(random_individual())
                
        population = new_pop
        scores = [score(ind) for ind in population]

    # Local Optimization on Best
    best_lt, best_t = best_ind
    x0 = np.array(best_t)
    
    def objective(x):
        if np.any(x < 0) or np.any(x > max_layer_nm):
            return 1e6
        if np.sum(x) > max_total_nm:
            return 1e6
        t_list = x.tolist()
        return _eval_reflectance(best_lt, t_list, materials, ambient_index, substrate_index, wavelengths) * 100.0
        
    try:
        res = so.minimize(objective, x0, method='L-BFGS-B', 
                          bounds=[(0, max_layer_nm)] * len(x0),
                          options={'maxiter': 200})
        if res.fun < best_score:
            best_t = res.x.tolist()
            best_score = res.fun
    except Exception:
        pass
        
    return best_lt, best_t
# EVOLVE-BLOCK-END
