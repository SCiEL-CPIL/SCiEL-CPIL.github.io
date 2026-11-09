"""Broadband antireflection coating design with local optimization."""
import time
import math

def _reflectance_spectrum(materials, thicknesses, n0, ns, wl_min, wl_max, n_pts):
    """Compute mean reflectance (percent) over n_pts wavelengths using
    the characteristic-matrix method at normal incidence."""
    nL = materials['L']
    nH = materials['H']
    n_layers = len(thicknesses)
    total_nm = sum(thicknesses)
    if n_layers == 0:
        # bare substrate
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    # Build wavelength grid
    if n_pts == 1:
        wavelengths = [(wl_min + wl_max) / 2.0]
    else:
        step = (wl_max - wl_min) / (n_pts - 1)
        wavelengths = [wl_min + i * step for i in range(n_pts)]

    total_ref = 0.0
    k2pi = 2.0 * math.pi

    for wl in wavelengths:
        # Stack matrices from substrate side up (M = M1 * M2 * ... * MNL)
        M = 1.0 + 0.0j
        for i in range(n_layers - 1, -1, -1):
            ni = nH if materials_labels[i] == 'H' else nL
            phase = k2pi * ni * thicknesses[i] / wl
            c = math.cos(phase)
            s = math.sin(phase)
            # Layer matrix: [[c, i*s/ni], [i*ni*s, c]]
            M11 = c
            M12 = 1j * s / ni
            M21 = 1j * ni * s
            M22 = c
            # M = layer_matrix * M
            new_M11 = M11 * M + M12 * (M.imag if False else 0)
            # Do proper 2x2 complex multiply
            # M is currently a scalar? No, need full 2x2.
            pass

    # Redo with proper 2x2 matrix handling
    total_ref = 0.0
    for wl in wavelengths:
        # Start with identity 2x2
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j

        for i in range(n_layers - 1, -1, -1):
            ni = nH if materials_labels[i] == 'H' else nL
            phase = k2pi * ni * thicknesses[i] / wl
            c = math.cos(phase)
            s = math.sin(phase)
            L11 = c
            L12 = 1j * s / ni
            L21 = 1j * ni * s
            L22 = c
            # new = L * M
            n11 = L11 * m11 + L12 * m21
            n12 = L11 * m12 + L12 * m22
            n21 = L21 * m11 + L22 * m21
            n22 = L21 * m12 + L22 * m22
            m11, m12, m21, m22 = n11, n12, n21, n22

        # Effective admittance Y = (n0 * m22 + m21) / (n0 * m12 + m11) ... 
        # Actually Y = (m11 * ns + m12 * ns * n0) / (m21 * ns + m22 * n0)
        # Standard: Y = (n0*m11 + m12) / (n0*m21 + m22) * ns ... 
        # Let me use: Y = (m11 + m12/ns) / (m21 + m22/ns) * ... 
        # Correct formula: Y = (m11*ns + m12) / (m21*ns + m22)
        Y = (m11 * ns + m12) / (m21 * ns + m22)
        r = (n0 - Y) / (n0 + Y)
        total_ref += abs(r) ** 2

    return 100.0 * total_ref / n_pts


def design(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    nL = materials['L']
    nH = materials['H']
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre_wl = 0.5 * (wl_min + wl_max)

    deadline = time.time() + 50.0  # safety margin before 60s CPU / 120s wall

    def eval_design(labels, thicknesses):
        """Evaluate mean reflectance for a given design."""
        global materials_labels
        materials_labels = labels
        return _reflectance_spectrum(materials, thicknesses, n0, ns, wl_min, wl_max, 200)

    # Start with a quarter-wave stack: L H L H ... or similar
    # Classic broadband AR: multiple quarter-wave pairs
    # For 400-800nm, centre=600nm
    # Quarter-wave thickness at centre: L: 600/(4*1.38)=108.7, H: 600/(4*2.35)=63.8
    
    # Try a 4-layer quarter-wave stack: L H H L (or L H L H)
    # Actually for broadband, alternating layers work well
    
    # Initial guess: 4 layers, quarter-wave at centre
    qL = centre_wl / (4.0 * nL)
    qH = centre_wl / (4.0 * nH)
    
    # Try several initial patterns
    candidates = []
    
    # Pattern 1: L H L H
    candidates.append((['L','H','L','H'], [qL, qH, qL, qH]))
    # Pattern 2: H L H L
    candidates.append((['H','L','H','L'], [qH, qL, qH, qL]))
    # Pattern 3: L H H L
    candidates.append((['L','H','H','L'], [qL, qH, qH, qL]))
    # Pattern 4: 6 layers L H L H L H
    candidates.append((['L','H','L','H','L','H'], [qL, qH, qL, qH, qL, qH]))
    # Pattern 5: 8 layers
    candidates.append((['L','H','L','H','L','H','L','H'], 
                       [qL, qH, qL, qH, qL, qH, qL, qH]))
    
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    for labels, thick in candidates:
        total = sum(thick)
        if total > max_total_nm or len(labels) > max_layers:
            continue
        if any(t < 0 or t > max_layer_nm for t in thick):
            continue
        score = eval_design(labels, thick)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thick = thick[:]
    
    if best_labels is None:
        # Fallback: single quarter-wave L
        return ["L"], [centre_wl / (4.0 * nL)]
    
    # Local optimization using simple coordinate descent / Nelder-Mead style
    # Use a simple gradient-free approach: perturb each thickness
    
    labels = best_labels
    thick = best_thick[:]
    current_score = best_score
    
    # Try to add more layers if we have budget
    # First, do local optimization on current design
    step_size = 5.0  # nm
    n_iterations = 0
    max_iter = 2000
    
    while n_iterations < max_iter and time.time() < deadline:
        improved = False
        for i in range(len(thick)):
            # Try increasing
            new_thick = thick[:]
            new_thick[i] = min(max_layer_nm, thick[i] + step_size)
            if sum(new_thick) <= max_total_nm:
                s = eval_design(labels, new_thick)
                if s < current_score:
                    thick = new_thick
                    current_score = s
                    improved = True
            
            # Try decreasing
            new_thick = thick[:]
            new_thick[i] = max(0.0, thick[i] - step_size)
            s = eval_design(labels, new_thick)
            if s < current_score:
                thick = new_thick
                current_score = s
                improved = True
        
        n_iterations += 1
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 0.5:
                break
    
    # Check if we can improve by adding layers
    if len(labels) < max_layers and time.time() < deadline:
        # Try adding a layer at various positions
        for pos in range(len(labels) + 1):
            for mat in ['L', 'H']:
                new_labels = labels[:]
                new_thick = thick[:]
                new_labels.insert(pos, mat)
                nm = qL if mat == 'L' else qH
                new_thick.insert(pos, nm)
                if sum(new_thick) <= max_total_nm and len(new_labels) <= max_layers:
                    s = eval_design(new_labels, new_thick)
                    if s < current_score:
                        labels = new_labels
                        thick = new_thick
                        current_score = s
                        # Re-optimize
                        step_size = 5.0
                        n_iterations = 0
                        while n_iterations < 500 and time.time() < deadline:
                            improved = False
                            for i in range(len(thick)):
                                new_thick = thick[:]
                                new_thick[i] = min(max_layer_nm, thick[i] + step_size)
                                if sum(new_thick) <= max_total_nm:
                                    s2 = eval_design(labels, new_thick)
                                    if s2 < current_score:
                                        thick = new_thick
                                        current_score = s2
                                        improved = True
                                new_thick = thick[:]
                                new_thick[i] = max(0.0, thick[i] - step_size)
                                s2 = eval_design(labels, new_thick)
                                if s2 < current_score:
                                    thick = new_thick
                                    current_score = s2
                                    improved = True
                            n_iterations += 1
                            if not improved:
                                step_size *= 0.5
                                if step_size < 0.5:
                                    break
                        break
    
    return labels, thick
