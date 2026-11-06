"""
Broadband antireflection coating design using characteristic-matrix method.
Strategy: Deterministic constructive search over layer sequences + Differential Evolution for thicknesses.
"""
import time
import numpy as np
import random
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute reflectance at all wavelengths using the characteristic matrix method (vectorized).
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r*r)
        
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for mat, thick in zip(materials, thicknesses):
        n = materials_dict[mat]
        if thick < 1e-9:
            continue
            
        beta = 2.0 * np.pi * n * thick / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = 1.0j * sin_b / n
        L21 = n * 1.0j * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = substrate_index
    Y = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    
    Y_0 = ambient_index
    r = (Y_0 - Y) / (Y_0 + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(materials, thicknesses, band_nm=(400, 800), n_points=401,
                       ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute mean reflectance in percent over the band."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    wls = np.linspace(band_nm[0], band_nm[1], n_points)
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
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

def _generate_candidate_sequences(max_layers=20):
    """
    Generate physically plausible layer sequences.
    We focus on alternating L and H layers, starting with L or H.
    We limit the number of layers to reasonable values for broadband AR coatings (typically 3-12).
    """
    sequences = []
    # Typical good designs are 3 to 12 layers. 
    # Going up to 20 is allowed but often diminishing returns or complex to optimize within time.
    # We will try a few different lengths and starting materials.
    for n_layers in range(2, 13):
        # Start with L
        seq1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        sequences.append(seq1)
        # Start with H
        seq2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        sequences.append(seq2)
        
    # Also add some specific known good patterns if time permits, 
    # but the alternating ones cover the main physics (quarter wave stacks).
    return sequences

def design(problem):
    """
    Design using constructive search over sequences and Differential Evolution for thicknesses.
    """
    start_time = time.time()
    # 150s wall clock limit. Keep a safety margin of 10s.
    deadline = start_time + 140.0

    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem.get("max_total_nm", 1000)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Baseline: Bare glass
    r_bare = (n_amb - n_sub) / (n_amb + n_sub)
    bare_reflectance = 100.0 * r_bare * r_bare
    
    # Initial best: A simple 1-layer quarter wave L
    best_materials = ["L"]
    best_thicknesses = [wl_center / (4.0 * nL)]
    best_score = _mean_reflectance(best_materials, best_thicknesses, band_nm, 401, 
                                    n_amb, n_sub, materials_dict)
    
    # Generate candidate sequences
    candidates = _generate_candidate_sequences(20)
    
    # Shuffle candidates to mix up the order, but prioritize shorter ones which are faster to evaluate?
    # Actually, DE is fast enough. Let's sort by length so we try simple ones first to establish a good baseline quickly.
    candidates.sort(key=lambda x: len(x))
    
    # We will iterate through candidates. 
    # For each candidate, we run Differential Evolution.
    # DE is stochastic but deterministic if we fix the seed. 
    # We must manage the time budget carefully.
    
    # Estimate time per DE run. 
    # DE with popsize=10, maxiter=50 is roughly 500-1000 function evaluations.
    # Each evaluation is vectorized numpy over 401 points. ~1ms.
    # So 1s per DE run. 
    # We have 140s. We can afford ~100 DE runs.
    # We have ~20 candidates. We can run multiple seeds or more iterations for each.
    
    # Strategy: 
    # 1. Run a quick DE on all candidates to find the best sequence type.
    # 2. Run a longer, more refined DE on the top 3 sequences.
    
    # Phase 1: Quick scan
    quick_candidates = candidates[:15] # First 15 sequences
    best_score_phase1 = best_score
    best_mat_phase1 = best_materials
    best_thick_phase1 = best_thicknesses
    
    for seq in quick_candidates:
        if time.time() > deadline - 60: # Leave 60s for phase 2 and finalization
            break
            
        n_layers = len(seq)
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        # Objective function for DE
        def objective(x):
            # x is array of thicknesses
            # Check total thickness constraint
            if np.sum(x) > max_total_nm:
                return 100.0 + 10.0 * (np.sum(x) - max_total_nm)
            
            t_list = [float(t) for t in x]
            score = _mean_reflectance(seq, t_list, band_nm, 401, 
                                      n_amb, n_sub, materials_dict)
            return score
        
        # Run DE
        # popsize=10, maxiter=30 for quick scan
        try:
            result = differential_evolution(objective, bounds, 
                                             maxiter=30, popsize=10, 
                                             seed=42, polish=False, 
                                             init='latinhypercube')
            
            if result.success:
                x_opt = result.x
                # Clip just in case
                x_opt = np.clip(x_opt, 0.0, max_layer_nm)
                
                # Final check on validity and score
                if np.sum(x_opt) <= max_total_nm:
                    final_score = objective(x_opt)
                    if final_score < best_score_phase1:
                        best_score_phase1 = final_score
                        best_mat_phase1 = list(seq)
                        best_thick_phase1 = [float(t) for t in x_opt]
        except Exception:
            pass
        
        # Update global best
        if best_score_phase1 < best_score:
            best_score = best_score_phase1
            best_materials = best_mat_phase1
            best_thicknesses = best_thick_phase1
            
    # Phase 2: Refined search on top candidates
    # We keep the best material sequence found so far, and maybe a few others.
    # Let's just refine the current best sequence with more iterations and polish.
    # Also, try a few other sequences that were close in Phase 1 if we have time.
    
    # For simplicity and robustness, we will just run a longer DE on the best sequence found so far.
    # And maybe one more alternative (e.g. different start).
    
    sequences_to_refine = [best_materials]
    # Add a couple of other common good structures if they weren't the best
    if "H" not in best_materials[0]:
        sequences_to_refine.append(["H"] + list(best_materials[1:]))
    else:
        sequences_to_refine.append(["L"] + list(best_materials[1:]))
    
    # Remove duplicates
    seen = set()
    unique_seqs = []
    for s in sequences_to_refine:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_seqs.append(s)
            
    for seq in unique_seqs:
        if time.time() > deadline - 10:
            break
            
        n_layers = len(seq)
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        def objective(x):
            if np.sum(x) > max_total_nm:
                return 100.0 + 10.0 * (np.sum(x) - max_total_nm)
            t_list = [float(t) for t in x]
            score = _mean_reflectance(seq, t_list, band_nm, 401, 
                                      n_amb, n_sub, materials_dict)
            return score
        
        try:
            # Longer run: maxiter=100, popsize=20, polish=True (uses L-BFGS-B)
            result = differential_evolution(objective, bounds, 
                                             maxiter=100, popsize=20, 
                                             seed=123, polish=True, 
                                             init='latinhypercube')
            
            if result.success:
                x_opt = result.x
                x_opt = np.clip(x_opt, 0.0, max_layer_nm)
                
                if np.sum(x_opt) <= max_total_nm:
                    final_score = objective(x_opt)
                    if final_score < best_score:
                        best_score = final_score
                        best_materials = list(seq)
                        best_thicknesses = [float(t) for t in x_opt]
        except Exception:
            pass
            
    # Final validity check
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to simple 1-layer
        best_materials = ["L"]
        best_thicknesses = [wl_center / (4.0 * nL)]
        
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
