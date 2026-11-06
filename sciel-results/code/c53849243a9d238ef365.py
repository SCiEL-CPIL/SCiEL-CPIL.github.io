"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    total_r = 0.0
    # Pre-compute layer indices and n values for speed
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    # Vectorize wavelength loop? No, matrix multiplication is sequential per layer.
    # But we can vectorize the wavelength dimension if we structure it carefully.
    # However, Nelder-Mead calls the function many times. Speed is critical.
    # Let's stick to the loop but optimize constants.
    
    # Pre-calculate 2*pi/lambda
    inv_lambdas = 2.0 * np.pi / lambdas
    
    for i_lam in range(n_waves):
        k = inv_lambdas[i_lam]
        
        A = 1.0 + 0j
        B = 0.0 + 0j
        C = 0.0 + 0j
        D = 1.0 + 0j
        
        for i in range(n_layers):
            n_i = n_dict[materials[i]]
            d_i = thicknesses[i]
            
            delta = k * n_i * d_i
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            M11 = cos_d
            M12 = 1j * sin_d / n_i
            M21 = 1j * n_i * sin_d
            M22 = cos_d
            
            nA = A * M11 + B * M21
            nB = A * M12 + B * M22
            nC = C * M11 + D * M21
            nD = C * M12 + D * M22
            
            A, B, C, D = nA, nB, nC, nD
        
        Y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - Y_in) / (n0 + Y_in)
        R = np.abs(r) ** 2
        total_r += R
        
    return 100.0 * total_r / n_waves

def _objective(x, materials, problem):
    """Objective function for optimizer. x is array of thicknesses."""
    # Ensure thicknesses are within bounds [0, max_layer_nm]
    # The optimizer might step out of bounds, so we clip or penalize.
    # Clipping is safer for Nelder-Mead which doesn't respect bounds strictly.
    max_layer = problem["max_layer_nm"]
    x_clipped = np.clip(x, 0, max_layer)
    
    # Check total thickness constraint
    if np.sum(x_clipped) > problem["max_total_nm"]:
        # Penalize heavily
        return 100.0 + (np.sum(x_clipped) - problem["max_total_nm"]) * 0.1
    
    return _compute_reflectance(materials, x_clipped, problem)

def design(problem):
    """Design a stack by optimizing thicknesses for several candidate layer sequences."""
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    # Define candidate sequences.
    # We want to cover different structural possibilities.
    # Starting with H is often good for broadband AR on glass (n=1.52) with air (n=1).
    # Alternating L/H is standard.
    
    sequences = []
    
    # Length 1
    sequences.append(["H"])
    sequences.append(["L"])
    
    # Length 2
    sequences.append(["H", "L"])
    sequences.append(["L", "H"])
    
    # Length 3
    sequences.append(["H", "L", "H"])
    sequences.append(["L", "H", "L"])
    
    # Length 4
    sequences.append(["H", "L", "H", "L"])
    sequences.append(["L", "H", "L", "H"])
    
    # Length 5
    sequences.append(["H", "L", "H", "L", "H"])
    sequences.append(["L", "H", "L", "H", "L"])
    
    # Length 6
    sequences.append(["H", "L", "H", "L", "H", "L"])
    sequences.append(["L", "H", "L", "H", "L", "H"])
    
    # Length 7
    sequences.append(["H", "L", "H", "L", "H", "L", "H"])
    sequences.append(["L", "H", "L", "H", "L", "H", "L"])
    
    # Length 8
    sequences.append(["H", "L", "H", "L", "H", "L", "H", "L"])
    sequences.append(["L", "H", "L", "H", "L", "H", "L", "H"])
    
    # Length 9
    sequences.append(["H", "L", "H", "L", "H", "L", "H", "L", "H"])
    sequences.append(["L", "H", "L", "H", "L", "H", "L", "H", "L"])
    
    # Length 10
    sequences.append(["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"])
    sequences.append(["L", "H", "L", "H", "L", "H", "L", "H", "L", "H"])
    
    # Remove duplicates if any (not expected here but safe)
    seen = set()
    unique_sequences = []
    for seq in sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_sequences.append(seq)
            
    best_r = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Initial guess: quarter-wave at center wavelength
    lam_center = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    for seq in unique_sequences:
        n_layers = len(seq)
        # Check if max total thickness allows even minimum layers?
        # Actually, we just need to check if a valid solution exists.
        # Min thickness is 0. So always possible.
        
        # Initial guess: quarter wave
        x0 = np.array([lam_center / (4.0 * problem["materials"][mat]) for mat in seq])
        
        # Bounds for each layer
        bounds = [(0, max_layer) for _ in range(n_layers)]
        
        # Use Nelder-Mead. It doesn't use bounds directly, so we rely on clipping in objective.
        # However, Nelder-Mead can be slow. Let's try L-BFGS-B which handles bounds.
        # L-BFGS-B is better for bounded problems.
        
        # But L-BFGS-B requires gradient or approximates it. Reflectance is smooth.
        # Let's use L-BFGS-B with bounds.
        
        try:
            # We need to ensure the total thickness constraint is handled.
            # L-BFGS-B handles simple bounds. The sum constraint is a linear constraint.
            # scipy.optimize.minimize supports 'A' and 'b' for linear constraints only in some methods?
            # No, 'A' and 'b' are for linear inequality constraints A @ x <= b.
            # So we can add: sum(x) <= max_total.
            
            A_eq = []
            b_eq = []
            A_ineq = [[1.0] * n_layers]
            b_ineq = [max_total]
            
            # L-BFGS-B supports bounds but NOT general linear constraints A@x <= b.
            # It only supports simple bounds.
            # So we must handle the sum constraint via penalty or by scaling bounds?
            # Scaling bounds is hard because the sum constraint couples variables.
            
            # Alternative: Use SLSQP which supports linear constraints.
            # Or use Nelder-Mead with penalty.
            
            # Given the small dimension (<=10), SLSQP is a good choice.
            
            res = minimize(
                _objective, 
                x0, 
                args=(seq, problem), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[{'type': 'ineq', 'fun': lambda x: max_total - np.sum(x)}],
                options={'ftol': 1e-8, 'maxiter': 1000, 'disp': False}
            )
            
            if res.success:
                x_opt = res.x
                # Clip to be safe
                x_opt = np.clip(x_opt, 0, max_layer)
                # Check sum again
                if np.sum(x_opt) <= max_total + 1e-5:
                    r = _compute_reflectance(seq, x_opt, problem)
                    if r < best_r:
                        best_r = r
                        best_materials = seq
                        best_thicknesses = x_opt.tolist()
                        
        except Exception:
            pass
            
    # Fallback if optimization failed for all
    if best_materials is None:
        # Return a simple quarter-wave doublet
        return ["H", "L"], [lam_center / (4.0 * nH), lam_center / (4.0 * nL)]
        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
