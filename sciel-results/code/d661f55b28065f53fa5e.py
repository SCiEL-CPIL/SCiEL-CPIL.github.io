import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


def _reflectance_spectrum(materials, thicknesses, n_layers,
                          ambient=1.0, substrate=1.52,
                          wl_start=400.0, wl_end=800.0, n_points=401):
    """Compute mean reflectance using characteristic matrix method."""
    nL = 1.38
    nH = 2.35
    wavelengths = np.linspace(wl_start, wl_end, n_points)
    
    reflectances = np.zeros(n_points)
    for idx, wl in enumerate(wavelengths):
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for i in range(n_layers):
            n = nL if materials[i] == 'L' else nH
            d = thicknesses[i]
            delta = 2.0 * np.pi * n * d / wl
            c = np.cos(delta)
            s = np.sin(delta)
            Mi = np.array([[c, 1j * s / n], [1j * n * s, c]], dtype=complex)
            M = M @ Mi
        
        B = M[0, 0]
        C = M[1, 0]
        if abs(B) < 1e-15:
            reflectances[idx] = 1.0
        else:
            Y = C / B
            r = (ambient - Y) / (ambient + Y)
            reflectances[idx] = np.abs(r) ** 2
    
    return np.mean(reflectances) * 100.0


def design(problem):
    """Design an antireflection coating using differential evolution + local refinement."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    ambient = problem.get("ambient_index", 1.0)
    substrate = problem.get("substrate_index", 1.52)
    materials_dict = problem.get("materials", {'L': 1.38, 'H': 2.35})
    band = problem.get("band_nm", [400, 800])
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)
    
    nL = materials_dict['L']
    nH = materials_dict['H']
    centre_wl = 0.5 * (band[0] + band[1])
    
    best_result = None
    best_score = float('inf')
    
    # Helper to update best
    def try_update(materials, thicknesses):
        nonlocal best_result, best_score
        if len(materials) > max_layers:
            return
        total = sum(thicknesses)
        if total > max_total + 1e-6:
            return
        if any(t < -1e-6 or t > max_layer + 1e-6 for t in thicknesses):
            return
        score = _reflectance_spectrum(materials, thicknesses, len(materials),
                                      ambient, substrate, band[0], band[1])
        if score < best_score:
            best_score = score
            best_result = (list(materials), list(thicknesses))
    
    # Initial guess: quarter-wave stack
    for n_layers in [4, 6, 8, 10]:
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L')
                thicks.append(centre_wl / (4.0 * nL))
            else:
                mats.append('H')
                thicks.append(centre_wl / (4.0 * nH))
        try_update(mats, thicks)
    
    # Differential evolution for different layer counts
    for n_layers in [6, 8, 10, 12]:
        if time.time() > deadline - 40:
            break
        
        bounds = []
        for i in range(n_layers):
            bounds.append((0.0, 1.0))
            bounds.append((1.0, max_layer))
        
        def make_objective(nl):
            def objective(params):
                if time.time() > deadline:
                    return best_score
                mat_codes = params[0::2]
                thicknesses = params[1::2]
                total_thick = np.sum(thicknesses)
                if total_thick > max_total:
                    return 100.0
                materials = ['L' if mc < 0.5 else 'H' for mc in mat_codes]
                return _reflectance_spectrum(materials, thicknesses, nl,
                                            ambient, substrate, band[0], band[1])
            return objective
        
        try:
            result = differential_evolution(
                make_objective(n_layers), bounds,
                seed=42 + n_layers,
                maxiter=40,
                popsize=12,
                tol=1e-5,
                polish=False,
                init='latinhypercube'
            )
            
            if result.fun < best_score:
                mat_codes = result.x[0::2]
                thicknesses = result.x[1::2]
                materials = ['L' if mc < 0.5 else 'H' for mc in mat_codes]
                try_update(materials, thicknesses)
                
                # Local refinement with Nelder-Mead on thicknesses only
                if time.time() < deadline - 20:
                    def local_obj(th):
                        if np.sum(th) > max_total:
                            return 100.0
                        if any(t < 0 or t > max_layer for t in th):
                            return 100.0
                        return _reflectance_spectrum(materials, th, n_layers,
                                                    ambient, substrate, band[0], band[1])
                    
                    try:
                        local_res = minimize(local_obj, thicknesses,
                                             method='Nelder-Mead',
                                             options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-5})
                        if local_res.fun < best_score:
                            try_update(materials, local_res.x)
                    except Exception:
                        pass
        except Exception:
            pass
    
    if best_result is None:
        return ["L"], [centre_wl / (4.0 * nL)]
    
    return best_result[0], best_result[1]
