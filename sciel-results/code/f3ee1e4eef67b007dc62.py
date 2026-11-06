import time
import numpy as np

def _char_matrix(n, d, lam):
    """Characteristic matrix for a single layer at wavelength lam (nm)."""
    beta = 2.0 * np.pi * n * d / lam
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # For normal incidence: M = [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    return np.array([[cos_b, 1j * sin_b / n],
                     [1j * n * sin_b, cos_b]], dtype=complex)

def _reflectance(materials, thicknesses, lam, n_sub):
    """Compute reflectance at wavelength lam for the given stack."""
    n0 = 1.0
    # Build total characteristic matrix
    M_total = np.eye(2, dtype=complex)
    for i in range(len(materials)):
        n = materials[materials[i]]
        d = thicknesses[i]
        M_layer = _char_matrix(n, d, lam)
        M_total = M_total @ M_layer
    # Substrate admittance
    Y_s = n_sub  # for TE at normal incidence
    # Input admittance
    Y_in = (M_total[0, 1]) / (M_total[0, 0])
    # Actually: Y_in = (M11 + M12*Y_s) / (M21 + M22*Y_s)
    Y_in = (M_total[0, 0] + M_total[0, 1] * Y_s) / (M_total[1, 0] + M_total[1, 1] * Y_s)
    # Reflectance
    r = (n0 - Y_in) / (n0 + Y_in)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    n_sub = problem["substrate_index"]
    m = problem["materials"]
    wl = np.linspace(band[0], band[1], 401)
    Rs = []
    for w in wl:
        Rs.append(_reflectance(materials, thicknesses, w, n_sub))
    return 100.0 * np.mean(Rs)

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_sub = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    # Try several fixed layer sequences and optimize thicknesses
    # Good broadband AR structures: alternate L and H, start and end with L
    
    # Define candidate sequences (from air side to substrate)
    # Length up to 20
    sequences = []
    # Symmetric structures
    for n_pairs in range(1, 11):
        # L H L H ... L H with n_pairs of (L,H)
        seq = []
        for _ in range(n_pairs):
            seq.append('L')
            seq.append('H')
        sequences.append(seq)
    # Also try starting and ending with L: L H L H ... H L
    for n_mid in range(0, 10):
        seq = ['L']
        for _ in range(n_mid):
            seq.append('H')
            seq.append('L')
        sequences.append(seq)
    # Filter to valid lengths
    sequences = [s for s in sequences if 1 <= len(s) <= max_layers]
    
    # Remove duplicates
    seen = set()
    unique_seqs = []
    for s in sequences:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_seqs.append(s)
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Initialize with quarter-wave thicknesses
    def init_thicknesses(seq):
        t = []
        for m in seq:
            n = nL if m == 'L' else nH
            t.append(centre / (4.0 * n))
        return t
    
    # Evaluate initial design
    for seq in unique_seqs:
        if time.time() > deadline:
            break
        t0 = init_thicknesses(seq)
        # Check constraints
        if sum(t0) > max_total:
            continue
        score = _mean_reflectance(seq, t0, problem)
        if score < best_score:
            best_score = score
            best_materials = list(seq)
            best_thicknesses = list(t0)
    
    # Now optimize each promising sequence
    # Use scipy.optimize.minimize with Nelder-Mead or Powell
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
    
    # Sort sequences by initial score, optimize best ones first
    seq_scores = []
    for seq in unique_seqs:
        t0 = init_thicknesses(seq)
        if sum(t0) > max_total:
            continue
        score = _mean_reflectance(seq, t0, problem)
        seq_scores.append((score, seq, t0))
    seq_scores.sort(key=lambda x: x[0])
    
    # Optimize top sequences
    for score0, seq, t0 in seq_scores:
        if time.time() > deadline:
            break
        if score0 > best_score * 1.5:  # Skip if much worse
            continue
        
        n_layers = len(seq)
        
        # Define objective with penalty for constraint violations
        def objective(x):
            # x: thicknesses
            if any(xi < 0 for xi in x):
                return 100.0
            if sum(x) > max_total:
                return 100.0
            if any(xi > max_layer for xi in x):
                return 100.0
            return _mean_reflectance(seq, x, problem)
        
        # Bounds
        bounds = [(0.0, max_layer) for _ in range(n_layers)]
        
        # Try multiple starting points
        starts = [t0]
        # Add perturbed starts
        np.random.seed(42)
        for _ in range(3):
            pert = [max(0.0, min(max_layer, ti * np.random.uniform(0.5, 1.5))) for ti in t0]
            if sum(pert) <= max_total:
                starts.append(pert)
        
        for x0 in starts:
            if time.time() > deadline:
                break
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                  bounds=bounds,
                                  options={'maxiter': 200, 'xatol': 1.0, 'fatol': 0.01})
                if result.fun < best_score:
                    best_score = result.fun
                    best_materials = list(seq)
                    best_thicknesses = list(result.x)
            except Exception:
                pass
        
        # Also try Powell
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, t0, method='Powell',
                              bounds=bounds,
                              options={'maxiter': 100, 'xtol': 1.0, 'ftol': 0.01})
            if result.fun < best_score:
                best_score = result.fun
                best_materials = list(seq)
                best_thicknesses = list(result.x)
        except Exception:
            pass
    
    # Final check: if no optimization worked, return best initial
    if best_materials is None:
        # Fallback: single L quarter-wave
        centre = 0.5 * (band[0] + band[1])
        return ["L"], [centre / (4.0 * nL)]
    
    # Clamp thicknesses to valid range
    best_thicknesses = [max(0.0, min(max_layer, t)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total:
        # Scale down
        scale = max_total / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
