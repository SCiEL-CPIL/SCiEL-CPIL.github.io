import time
import math
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
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    # Build per-layer optical path lengths
    n_ds = np.array([n * d for n, d in zip(ns, ds)])  # (N,)
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]  # (N, K)
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
        # Update A,B,C,D
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    # Reflectance: r = (n_0 * (A + B * n_s) - (C + D * n_s)) / (n_0 * (A + B * n_s) + (C + D * n_s))
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def _nelder_mead(f, x0, max_iter=100, tol=1e-6, deadline=None):
    """
    Nelder-Mead optimization.
    f: objective function
    x0: initial guess (numpy array)
    """
    n = len(x0)
    # Create simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(1, n + 1):
        simplex[i] = x0.copy()
        simplex[i][i - 1] += 0.1 * abs(x0[i - 1]) + 1e-6
        
    # Evaluate initial simplex
    values = np.array([f(x) for x in simplex])
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Sort simplex
        idx = np.argsort(values)
        simplex = simplex[idx]
        values = values[idx]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = f(x_r)
        
        if f_r < values[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = f(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = f(simplex[i])
                    
    # Return best
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]

def _ga_optimize(problem, max_layers, max_total, max_layer, deadline):
    """
    Global search using Genetic Algorithm to find optimal sequence and thicknesses.
    Returns best materials list and thicknesses list.
    """
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    # Precompute quarter-wave thicknesses
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    pop_size = 40
    max_gen = 30
    mutation_rate = 0.1
    
    # Initialize population
    # Each individual: (sequence, thicknesses)
    # We represent sequence as a list of 'L'/'H'
    # Thicknesses as a list of floats
    
    population = []
    for _ in range(pop_size):
        # Random length between 3 and max_layers
        N = random.randint(3, max_layers)
        seq = []
        # Bias towards alternating but allow some randomness
        # Start with alternating, then mutate
        base_seq = []
        start = random.choice(['L', 'H'])
        for i in range(N):
            if i % 2 == 0:
                base_seq.append(start)
            else:
                base_seq.append('H' if start == 'L' else 'L')
        
        # Mutate sequence: flip a few layers
        seq = base_seq.copy()
        for _ in range(random.randint(0, 3)):
            idx = random.randint(0, N-1)
            seq[idx] = 'H' if seq[idx] == 'L' else 'L'
            
        # Random thicknesses
        ths = [random.uniform(0, max_layer) for _ in range(N)]
        # Ensure total thickness constraint roughly
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
            
        population.append((seq, ths))
        
    def evaluate(ind):
        seq, ths = ind
        # Enforce constraints strictly for evaluation score
        # If invalid, return high penalty
        if sum(ths) > max_total + 1e-6:
            return 100.0 # High penalty
        if len(seq) > max_layers:
            return 100.0
        for t in ths:
            if t < 0 or t > max_layer:
                return 100.0
        return _mean_reflectance(seq, ths, problem)
        
    best_ind = None
    best_score = float('inf')
    
    for gen in range(max_gen):
        if time.time() > deadline:
            break
            
        # Evaluate population
        scored_pop = []
        for ind in population:
            score = evaluate(ind)
            scored_pop.append((score, ind))
            
        scored_pop.sort(key=lambda x: x[0])
        
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_ind = scored_pop[0][1]
            
        # Elitism: keep top few
        elite_count = 5
        new_population = [ind for _, ind in scored_pop[:elite_count]]
        
        # Create offspring
        while len(new_population) < pop_size:
            # Selection: Tournament
            def select():
                candidates = random.sample(scored_pop, 3)
                return min(candidates, key=lambda x: x[0])[1]
                
            parent1 = select()
            parent2 = select()
            
            # Crossover: Mix sequences and thicknesses
            # Since lengths might differ, we handle this carefully
            seq1, ths1 = parent1
            seq2, ths2 = parent2
            
            # Use length of parent1 for offspring
            N = len(seq1)
            
            # Truncate or pad parent2 if needed
            if len(seq2) < N:
                # Pad with alternating pattern
                extra = N - len(seq2)
                last = seq2[-1] if seq2 else 'L'
                for i in range(extra):
                    next_mat = 'H' if last == 'L' else 'L'
                    seq2.append(next_mat)
                    last = next_mat
                    ths2.append(random.uniform(0, max_layer))
            elif len(seq2) > N:
                seq2 = seq2[:N]
                ths2 = ths2[:N]
                
            # Crossover point
            cross_point = random.randint(1, N-1) if N > 1 else 0
            
            new_seq = seq1[:cross_point] + seq2[cross_point:]
            new_ths = ths1[:cross_point] + ths2[cross_point:]
            
            # Mutation
            # Mutate sequence
            for i in range(N):
                if random.random() < mutation_rate:
                    new_seq[i] = 'H' if new_seq[i] == 'L' else 'L'
                    
            # Mutate thicknesses
            for i in range(N):
                if random.random() < mutation_rate:
                    # Add noise
                    noise = random.gauss(0, 10)
                    new_ths[i] += noise
                    if new_ths[i] < 0: new_ths[i] = 0
                    if new_ths[i] > max_layer: new_ths[i] = max_layer
                    
            # Enforce total thickness
            total = sum(new_ths)
            if total > max_total:
                scale = max_total / total
                new_ths = [t * scale for t in new_ths]
                
            new_population.append((new_seq, new_ths))
            
        population = new_population
        
    return best_ind

def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit

    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Global Search with GA
    # We run GA for a fixed number of iterations or until time is up
    # GA handles both sequence and thicknesses
    
    # To speed up GA, we can limit the max layers in GA to a reasonable subset
    # or just let it explore. Let's let it explore up to max_layers.
    
    ga_result = _ga_optimize(problem, max_layers, max_total, max_layer, deadline - 5)
    
    if ga_result:
        mats, ths = ga_result
        score = _mean_reflectance(mats, ths, problem)
        if score < best_score:
            best_score = score
            best_result = (mats, ths)
            
    # Phase 2: Local Refinement with Nelder-Mead
    # Refine the best result from GA
    if best_result:
        mats, ths = best_result
        N = len(mats)
        
        # Define objective for NM
        # NM works on continuous variables. We fix the sequence and optimize thicknesses.
        # We also need to handle the total thickness constraint.
        # We can parameterize thicknesses to automatically satisfy sum <= max_total?
        # Or just use penalty.
        
        def objective_nm(ind):
            ths = np.clip(ind, 0, max_layer)
            total = np.sum(ths)
            if total > max_total:
                # Scale down to valid region
                ths = ths * (max_total / total)
                # Add small penalty to discourage boundary
                penalty = 0.01 * (total - max_total)
            else:
                penalty = 0
            score = _mean_reflectance(mats, list(ths), problem)
            return score + penalty
            
        # Initial guess
        x0 = np.array(ths)
        
        # Run NM
        try:
            best_x, best_val = _nelder_mead(objective_nm, x0, max_iter=200, deadline=deadline)
            best_ths = np.clip(best_x, 0, max_layer)
            total_ths = np.sum(best_ths)
            if total_ths > max_total:
                best_ths = best_ths * (max_total / total_ths)
                
            final_score = _mean_reflectance(mats, list(best_ths), problem)
            
            if final_score < best_score:
                best_score = final_score
                best_result = (mats, list(best_ths))
                
        except Exception as e:
            pass
            
    # If no valid result found, return a simple 1-layer AR coating
    if best_result is None:
        wl_center = 0.5 * (band[0] + band[1])
        n_L = problem["materials"]["L"]
        qw_L = wl_center / (4.0 * n_L)
        return ["L"], [qw_L]
        
    mats, ths = best_result
    return list(mats), list(ths)
