"""Broadband antireflection coating design via Simulated Annealing with structural moves."""
import numpy as np
import time
import random

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _evaluate(labels, thicknesses, wavelengths_nm, ambient_idx, substrate_idx, materials,
              max_layer_nm, max_total_nm):
    """
    Evaluate a design. Returns mean reflectance in percent, or 100.0 if invalid.
    """
    if len(labels) == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return r0 * r0 * 100.0
    
    if len(labels) != len(thicknesses):
        return 100.0
    
    if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in thicknesses):
        return 100.0
    
    if sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(np.array(thicknesses), 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    indices = [materials[label] for label in labels]
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped,
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _sa_optimize(labels, thicknesses, wavelengths_nm, ambient_idx, substrate_idx, materials,
                 max_layers, max_layer_nm, max_total_nm, time_budget, rng):
    """
    Simulated Annealing with structural moves:
    - Single-layer thickness perturbation
    - Material swap (L <-> H)
    - Layer insertion
    - Layer deletion
    """
    n = len(labels)
    current_labels = list(labels)
    current_thick = np.array(thicknesses, dtype=float)
    current_score = _evaluate(current_labels, current_thick, wavelengths_nm, ambient_idx,
                              substrate_idx, materials, max_layer_nm, max_total_nm)
    
    best_labels = list(current_labels)
    best_thick = current_thick.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Multi-temperature schedule
    T_start = 5.0
    T_end = 0.001
    n_steps = 20000
    
    step = 0
    while step < n_steps:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Temperature schedule: exponential cooling with occasional reheating
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        if rng.random() < 0.01:
            T = T_start * 0.5  # Reheat
        
        # Choose move type
        move_type = rng.random()
        
        new_labels = list(current_labels)
        new_thick = current_thick.copy()
        
        if move_type < 0.5:
            # Thickness perturbation
            if len(new_labels) == 0:
                continue
            i = rng.randint(0, len(new_labels) - 1)
            delta = rng.gauss(0, 30)
            new_thick[i] = np.clip(new_thick[i] + delta, 0, max_layer_nm)
            
        elif move_type < 0.7:
            # Material swap
            if len(new_labels) == 0:
                continue
            i = rng.randint(0, len(new_labels) - 1)
            new_labels[i] = 'H' if new_labels[i] == 'L' else 'L'
            
        elif move_type < 0.85 and len(new_labels) < max_layers:
            # Layer insertion
            i = rng.randint(0, len(new_labels))
            label = 'L' if rng.random() < 0.5 else 'H'
            thick = rng.uniform(20, min(200, max_layer_nm))
            # Check total thickness constraint
            if np.sum(new_thick) + thick > max_total_nm:
                thick = max(0, max_total_nm - np.sum(new_thick))
                if thick < 1:
                    continue
            new_labels.insert(i, label)
            new_thick = np.insert(new_thick, i, thick)
            
        elif move_type < 1.0 and len(new_labels) > 1:
            # Layer deletion
            i = rng.randint(0, len(new_labels) - 1)
            new_labels.pop(i)
            new_thick = np.delete(new_thick, i)
        
        else:
            continue
        
        new_score = _evaluate(new_labels, new_thick, wavelengths_nm, ambient_idx,
                              substrate_idx, materials, max_layer_nm, max_total_nm)
        
        if new_score < current_score:
            current_labels = new_labels
            current_thick = new_thick
            current_score = new_score
            
            if new_score < best_score:
                best_labels = list(new_labels)
                best_thick = new_thick.copy()
                best_score = new_score
        
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if delta < 50:  # Avoid overflow
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_labels = new_labels
                    current_thick = new_thick
                    current_score = new_score
        
        step += 1
    
    return best_labels, best_thick, best_score

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    best_labels = None
    best_thick = None
    best_score = np.inf
    
    # Generate diverse initial designs
    initial_designs = []
    
    # Quarter-wave stack starting with L
    for n in [3, 5, 7, 9, 11]:
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        thick = [center_wl / (4.0 * materials[l]) for l in labels]
        if sum(thick) <= max_total_nm:
            initial_designs.append((labels, thick))
    
    # Quarter-wave stack starting with H
    for n in [3, 5, 7, 9]:
        labels = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        thick = [center_wl / (4.0 * materials[l]) for l in labels]
        if sum(thick) <= max_total_nm:
            initial_designs.append((labels, thick))
    
    # Random designs
    for _ in range(10):
        n = rng.randint(3, min(15, max_layers))
        labels = ['L' if rng.random() < 0.5 else 'H' for _ in range(n)]
        # Random thicknesses with total constraint
        weights = np_rng.dirichlet(np.ones(n))
        thick = weights * max_total_nm * 0.9
        thick = np.clip(thick, 0, max_layer_nm)
        if np.sum(thick) <= max_total_nm:
            initial_designs.append((labels, thick.tolist()))
    
    # Single layer
    initial_designs.append((['L'], [center_wl / (4.0 * materials['L'])]))
    initial_designs.append((['H'], [center_wl / (4.0 * materials['H'])]))
    
    # Evaluate all initial designs
    for labels, thick in initial_designs:
        score = _evaluate(labels, thick, wavelengths_nm, ambient_idx, substrate_idx,
                          materials, max_layer_nm, max_total_nm)
        if score < best_score:
            best_score = score
            best_labels = list(labels)
            best_thick = np.array(thick, dtype=float)
    
    # Run SA from each promising initial design
    # Sort by score and run SA on top ones
    scored_designs = []
    for labels, thick in initial_designs:
        score = _evaluate(labels, thick, wavelengths_nm, ambient_idx, substrate_idx,
                          materials, max_layer_nm, max_total_nm)
        scored_designs.append((score, labels, thick))
    
    scored_designs.sort(key=lambda x: x[0])
    
    # Run SA on top 5 designs
    for score, labels, thick in scored_designs[:5]:
        elapsed = time.time() - start_time
        remaining = deadline - 10 - elapsed
        if remaining < 5:
            break
        
        # Allocate time proportionally
        time_budget = min(remaining * 0.8, 30.0)
        
        try:
            sa_labels, sa_thick, sa_score = _sa_optimize(
                labels, thick, wavelengths_nm, ambient_idx, substrate_idx, materials,
                max_layers, max_layer_nm, max_total_nm, time_budget, rng)
            
            if sa_score < best_score:
                best_score = sa_score
                best_labels = sa_labels
                best_thick = sa_thick
        except Exception:
            pass
    
    # Final local polish: fine-tune thicknesses with small perturbations
    if best_labels is not None and time.time() < deadline - 5:
        polish_time = deadline - 5 - time.time()
        if polish_time > 1:
            n_polish = 5000
            for _ in range(n_polish):
                if time.time() > deadline - 5:
                    break
                
                i = rng.randint(0, len(best_labels) - 1)
                delta = rng.gauss(0, 5)
                new_thick = best_thick.copy()
                new_thick[i] = np.clip(new_thick[i] + delta, 0, max_layer_nm)
                
                if np.sum(new_thick) > max_total_nm:
                    continue
                
                score = _evaluate(best_labels, new_thick, wavelengths_nm, ambient_idx,
                                  substrate_idx, materials, max_layer_nm, max_total_nm)
                if score < best_score:
                    best_score = score
                    best_thick = new_thick
    
    # Ensure constraints on final result
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    thicknesses = np.clip(best_thick, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
