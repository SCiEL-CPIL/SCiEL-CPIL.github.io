import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def calc_reflectance(materials, thicknesses):
        N = len(materials)
        if N == 0:
            # Bare substrate
            R = ((n0 - ns) / (n0 + ns))**2
            return R
            
        n_layers = np.array([nL if m == 'L' else nH for m in materials])
        d_layers = np.array(thicknesses)
        
        n_d = n_layers * d_layers
        deltas = 2.0 * np.pi * n_d[:, None] / wavelengths[None, :]
        
        cos_d = np.cos(deltas)
        sin_d = np.sin(deltas)
        
        M = np.eye(2, dtype=complex)
        
        for i in range(N):
            mi_11 = cos_d[i]
            mi_12 = 1j * sin_d[i] / n_layers[i]
            mi_21 = 1j * n_layers[i] * sin_d[i]
            mi_22 = cos_d[i]
            
            M = M @ np.array([[mi_11, mi_12], [mi_21, mi_22]])
            
        B = M[0, 0] + M[0, 1] * ns
        C = M[1, 0] + M[1, 1] * ns
        
        Y = C / B
        R = np.abs((n0 - Y) / (n0 + Y))**2
        return np.mean(R)

    def objective_pct(materials, thicknesses):
        if time.time() > deadline:
            return 1e9
        t = np.clip(thicknesses, 0, max_layer)
        total_t = np.sum(t)
        penalty = 0.0
        if total_t > max_total:
            penalty = 1000.0 * (total_t - max_total)
        R_frac = calc_reflectance(materials, t)
        return R_frac * 100.0 + penalty

    best_score = 100.0
    best_materials = ["L"]
    best_thicknesses = [600 / (4 * nL)]
    
    # Initial evaluation
    best_score = objective_pct(best_materials, best_thicknesses)
    
    # Try different N and sequences
    for N in [1, 2, 4, 6, 8]:
        if time.time() > deadline:
            break
            
        seqs = []
        if N == 1:
            seqs = [['L'], ['H']]
        else:
            seq1 = ['L' if i % 2 == 0 else 'H' for i in range(N)]
            seq2 = ['H' if i % 2 == 0 else 'L' for i in range(N)]
            seqs = [seq1, seq2]
            
        for seq in seqs:
            if time.time() > deadline:
                break
                
            init_t = np.array([600 / (4 * (nL if m=='L' else nH)) for m in seq])
            
            bounds = [(0, max_layer) for _ in range(N)]
            
            def de_obj(x):
                if time.time() > deadline:
                    return 1e9
                return objective_pct(seq, x)
                
            try:
                result = differential_evolution(
                    de_obj, 
                    bounds, 
                    maxiter=20, 
                    popsize=5, 
                    tol=1e-3, 
                    seed=42,
                    polish=False # We will polish manually
                )
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_thicknesses = list(result.x)
                    best_materials = seq
                    
            except Exception:
                pass
            
            # Polish with Nelder-Mead
            if time.time() < deadline - 5:
                try:
                    nm_result = minimize(
                        de_obj, 
                        result.x, 
                        method='Nelder-Mead', 
                        options={'maxiter': 100, 'xatol': 1e-4}
                    )
                    if nm_result.fun < best_score:
                        best_score = nm_result.fun
                        best_thicknesses = list(nm_result.x)
                        best_materials = seq
                except Exception:
                    pass

    return best_materials, best_thicknesses
