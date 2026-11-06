"""Antireflection coating design via characteristic-matrix method and optimization."""
import numpy as np
import time

def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Problem parameters
    ambient_index = problem["ambient_index"]       # 1.0
    substrate_index = problem["substrate_index"]   # 1.52
    nL = problem["materials"]["L"]                 # 1.38
    nH = problem["materials"]["H"]                 # 2.35
    band_nm = problem["band_nm"]                   # [400, 800]
    max_layers = problem["max_layers"]             # 20
    max_total_nm = problem["max_total_nm"]         # 1000
    max_layer_nm = problem["max_layer_nm"]         # 400

    # Wavelength grid for evaluation (401 points)
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)

    def compute_reflectance(materials, thicknesses, wls):
        """Compute mean reflectance (%) using characteristic matrix method.
        
        materials: list of 'L' or 'H'
        thicknesses: list of physical thicknesses in nm
        wls: array of wavelengths in nm
        """
        n_layers = len(materials)
        if n_layers == 0:
            # Bare substrate
            r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
            return np.mean(r * r) * 100.0

        # Refractive indices of each layer
        n_arr = np.array([nL if m == 'L' else nH for m in materials])
        t_arr = np.array(thicknesses, dtype=float)

        # For each wavelength, compute the characteristic matrix
        reflectances = np.zeros(len(wls))
        
        for i, wl in enumerate(wls):
            # Phase thickness: delta = 2*pi*n*d / lambda
            delta = 2.0 * np.pi * n_arr * t_arr / wl
            
            # Build the total characteristic matrix
            # Each layer has matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
            # The total matrix is the product of all layer matrices
            # M_total = M_1 * M_2 * ... * M_N
            
            # Start with identity
            M = np.array([[1.0 + 0j, 0.0 + 0j],
                          [0.0 + 0j, 1.0 + 0j]])
            
            for j in range(n_layers):
                c = np.cos(delta[j])
                s = np.sin(delta[j])
                Mj = np.array([[c, 1j * s / n_arr[j]],
                               [1j * n_arr[j] * s, c]])
                M = M @ Mj
            
            # Admittance: Y = M[1,0] / M[0,0]
            # Reflectance: R = | (Y0 - Y) / (Y0 + Y) |^2 where Y0 = ambient_index
            Y = M[1, 0] / M[0, 0]
            r = (ambient_index - Y) / (ambient_index + Y)
            reflectances[i] = np.abs(r) ** 2
        
        return np.mean(reflectances) * 100.0

    def validate(materials, thicknesses):
        """Check if the design is valid."""
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        if any(m not in ('L', 'H') for m in materials):
            return False
        return True

    # Initial best: single quarter-wave layer at center wavelength
    centre = 0.5 * (band_nm[0] + band_nm[1])
    best_materials = ["L"]
    best_thicknesses = [centre / (4.0 * nL)]
    best_score = compute_reflectance(best_materials, best_thicknesses, wavelengths)

    # Also try a few analytical designs
    # Two-layer quarter-wave stack: L then H (or H then L)
    for mat in [["L", "H"], ["H", "L"]]:
        t = [centre / (4.0 * nL), centre / (4.0 * nH)]
        score = compute_reflectance(mat, t, wavelengths)
        if score < best_score:
            best_materials = mat
            best_thicknesses = t
            best_score = score

    # Multi-layer quarter-wave stack
    # Try alternating L, H, L, H, ... with quarter-wave thicknesses
    for n_qw in range(3, 15):
        if time.time() > deadline:
            break
        mat = []
        t = []
        for k in range(n_qw):
            if k % 2 == 0:
                mat.append("L")
                t.append(centre / (4.0 * nL))
            else:
                mat.append("H")
                t.append(centre / (4.0 * nH))
        if sum(t) > max_total_nm:
            break
        score = compute_reflectance(mat, t, wavelengths)
        if score < best_score:
            best_materials = mat
            best_thicknesses = t
            best_score = score

    # Now do a local optimization using Nelder-Mead or similar
    # We'll optimize the thicknesses of a fixed layer structure
    # Try different structures and optimize their thicknesses

    import scipy.optimize

    def objective(params):
        """Objective function for optimization."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return best_score  # Return current best to stop optimization
        
        # params: first len(materials) values are thicknesses
        thicknesses = np.clip(params, 0, max_layer_nm)
        if sum(thicknesses) > max_total_nm:
            return 100.0  # Penalty
        return compute_reflectance(best_materials, thicknesses, wavelengths)

    # Try optimizing various structures
    structures_to_try = [
        ["L"],
        ["L", "H"],
        ["H", "L"],
        ["L", "H", "L"],
        ["H", "L", "H"],
        ["L", "H", "L", "H"],
        ["H", "L", "H", "L"],
        ["L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L", "H", "L"],
    ]

    for mat in structures_to_try:
        if time.time() > deadline - 10.0:
            break
        
        n = len(mat)
        # Initial guess: quarter-wave at center
        init_t = []
        for m in mat:
            nm = nL if m == 'L' else nH
            init_t.append(centre / (4.0 * nm))
        
        # Check if total exceeds max
        if sum(init_t) > max_total_nm:
            continue
        
        # Try Nelder-Mead optimization
        try:
            result = scipy.optimize.minimize(
                objective,
                init_t,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1.0, 'fatol': 1e-6, 'disp': False}
            )
            elapsed = time.time() - start_time
            if elapsed > deadline - 5.0:
                break
            
            opt_t = np.clip(result.x, 0, max_layer_nm)
            if sum(opt_t) <= max_total_nm + 1e-9:
                score = compute_reflectance(mat, opt_t, wavelengths)
                if score < best_score:
                    best_materials = mat
                    best_thicknesses = [float(t) for t in opt_t]
                    best_score = score
        except Exception:
            pass

    # Try a few more random structures with optimization
    np.random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        
        # Random structure
        n = np.random.randint(3, 11)
        mat = []
        prev = None
        for k in range(n):
            if prev is None or np.random.random() < 0.7:
                # Prefer alternating
                m = "L" if (prev != "L" or np.random.random() < 0.3) else "H"
            else:
                m = "H" if prev == "L" else "L"
            mat.append(m)
            prev = m
        
        # Initial guess
        init_t = []
        for m in mat:
            nm = nL if m == 'L' else nH
            init_t.append(centre / (4.0 * nm) * np.random.uniform(0.8, 1.2))
        
        if sum(init_t) > max_total_nm:
            continue
        
        try:
            result = scipy.optimize.minimize(
                objective,
                init_t,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1.0, 'fatol': 1e-6, 'disp': False}
            )
            elapsed = time.time() - start_time
            if elapsed > deadline - 5.0:
                break
            
            opt_t = np.clip(result.x, 0, max_layer_nm)
            if sum(opt_t) <= max_total_nm + 1e-9:
                score = compute_reflectance(mat, opt_t, wavelengths)
                if score < best_score:
                    best_materials = mat
                    best_thicknesses = [float(t) for t in opt_t]
                    best_score = score
        except Exception:
            pass

    # Final validation
    if not validate(best_materials, best_thicknesses):
        # Fallback to simple design
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * nL)]

    return best_materials, [float(t) for t in best_thicknesses]
