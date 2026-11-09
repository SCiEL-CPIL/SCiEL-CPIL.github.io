"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    ambient_index = problem["ambient_index"]  # 1.0
    substrate_index = problem["substrate_index"]  # 1.52
    materials = problem["materials"]  # {'L': 1.38, 'H': 2.35}
    nL = materials["L"]
    nH = materials["H"]
    band_nm = problem["band_nm"]  # [400, 800]
    max_layers = problem["max_layers"]  # 20
    max_total_nm = problem["max_total_nm"]  # 1000
    max_layer_nm = problem["max_layer_nm"]  # 400

    # Wavelengths for evaluation: 401 equally spaced points from 400 to 800 nm
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)

    def compute_mean_reflectance(labels, thicknesses):
        """Compute mean reflectance (%) using characteristic matrix method."""
        n_layers = len(labels)
        if n_layers == 0:
            # Bare substrate
            r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
            return 100.0 * r0 * r0

        # Build characteristic matrices for each wavelength
        total_reflectance = 0.0
        for wl in wavelengths:
            # Optical thicknesses (delta = 2*pi*n*d / lambda)
            # For each layer, build the characteristic matrix
            M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

            for i in range(n_layers):
                n_i = nL if labels[i] == 'L' else nH
                d_i = thicknesses[i]
                # Phase delta = 2*pi*n*d/lambda
                delta = 2.0 * math.pi * n_i * d_i / wl
                # Characteristic matrix for a single layer:
                # [[cos(delta), (i/sin(delta))/n_i],
                #  [n_i*sin(delta), cos(delta)]]
                c = math.cos(delta)
                s = math.sin(delta)
                if abs(s) < 1e-15:
                    # Handle sin(delta) ~ 0 case
                    M_layer = np.array([[c, 0.0], [n_i * s, c]], dtype=complex)
                else:
                    M_layer = np.array([[c, 1.0j / (n_i * s)], [n_i * s, c]], dtype=complex)
                M_total = M_total @ M_layer

            # Admittance at the front surface
            # Y = M[1,0] / M[0,0] (in units of the substrate admittance... actually)
            # For normal incidence, the input admittance is:
            # Y_in = M[1,0] / M[0,0]
            # Reflectance: R = |(Y_in - n_ambient) / (Y_in + n_ambient)|^2
            M00, M01 = M_total[0, 0], M_total[0, 1]
            M10, M11 = M_total[1, 0], M_total[1, 1]

            # The characteristic matrix relates:
            # [E, H]_front = M * [E, H]_substrate_interface
            # For a plane wave in the substrate: H_sub = E_sub / n_sub
            # So [E_sub, E_sub/n_sub]
            # At the front: E_front = M00*E_sub + M01*E_sub/n_sub
            #              H_front = M10*E_sub + M11*E_sub/n_sub
            # Admittance Y = E_front / H_front (normalized)
            # Actually, let's use the standard formula:
            # B = M00 + M01 / n_sub
            # C = M10 + M11 / n_sub
            # Y_in = B / C
            # R = |(Y_in - n_ambient) / (Y_in + n_ambient)|^2

            B = M00 + M01 / substrate_index
            C = M10 + M11 / substrate_index
            Y_in = B / C

            r = (Y_in - ambient_index) / (Y_in + ambient_index)
            R = abs(r) ** 2
            total_reflectance += R

        mean_R = total_reflectance / len(wavelengths)
        return 100.0 * mean_R

    def check_constraints(labels, thicknesses):
        """Check if the design satisfies all constraints."""
        if len(labels) > max_layers:
            return False
        if len(labels) != len(thicknesses):
            return False
        total = sum(thicknesses)
        if total > max_total_nm + 1e-9:
            return False
        for t in thicknesses:
            if t < -1e-9 or t > max_layer_nm + 1e-9:
                return False
            if labels[labels.index(t) if t in labels else 0] not in ['L', 'H']:
                return False
        for lab in labels:
            if lab not in ['L', 'H']:
                return False
        return True

    # Strategy: Use a heuristic-based search with local optimization
    # Start with a known good AR coating structure and optimize

    # Known good starting points for broadband AR coatings:
    # 1. Quarter-wave stack: L/H/L/H/... alternating
    # 2. Multi-layer with optimized thicknesses

    # Let's try several initial structures and optimize each

    def optimize_thicknesses(labels, thicknesses, max_time=10.0):
        """Local optimization of thicknesses using Nelder-Mead or gradient-free method."""
        opt_start = time.time()
        n_params = len(thicknesses)
        if n_params == 0:
            return labels, thicknesses

        # Use scipy's minimize if available, else manual Nelder-Mead
        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fallback: simple coordinate descent
            best_thick = list(thicknesses)
            best_score = compute_mean_reflectance(labels, best_thick)
            step = 5.0  # nm
            for _ in range(50):
                if time.time() - opt_start > max_time:
                    break
                improved = False
                for i in range(n_params):
                    for direction in [-step, step]:
                        new_thick = list(best_thick)
                        new_thick[i] = max(0.0, min(max_layer_nm, new_thick[i] + direction))
                        if sum(new_thick) > max_total_nm:
                            continue
                        score = compute_mean_reflectance(labels, new_thick)
                        if score < best_score:
                            best_score = score
                            best_thick = new_thick
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 0.1:
                        break
            return labels, best_thick

        def objective(x):
            x = np.clip(x, 0, max_layer_nm)
            if np.sum(x) > max_total_nm:
                return 100.0
            return compute_mean_reflectance(labels, list(x))

        x0 = np.array(thicknesses, dtype=float)
        bounds = [(0.0, max_layer_nm) for _ in range(n_params)]

        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 0.1, 'fatol': 1e-4,
                                  'adaptive': True},
                         bounds=bounds)

        if result.success or result.fun < compute_mean_reflectance(labels, thicknesses):
            best_thick = list(np.clip(result.x, 0, max_layer_nm))
            # Ensure total thickness constraint
            total = sum(best_thick)
            if total > max_total_nm:
                scale = max_total_nm / total
                best_thick = [t * scale for t in best_thick]
            return labels, best_thick
        else:
            return labels, thicknesses

    # Generate candidate structures
    candidates = []

    # Candidate 1: Single quarter-wave L layer
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    d_qw_L = center_wl / (4.0 * nL)
    candidates.append((['L'], [d_qw_L]))

    # Candidate 2: Quarter-wave L/H pair
    d_qw_H = center_wl / (4.0 * nH)
    candidates.append((['L', 'H'], [d_qw_L, d_qw_H]))

    # Candidate 3: L/H/L
    candidates.append((['L', 'H', 'L'], [d_qw_L, d_qw_H, d_qw_L]))

    # Candidate 4: H/L/H
    candidates.append((['H', 'L', 'H'], [d_qw_H, d_qw_L, d_qw_H]))

    # Candidate 5: L/H/L/H
    candidates.append((['L', 'H', 'L', 'H'], [d_qw_L, d_qw_H, d_qw_L, d_qw_H]))

    # Candidate 6: H/L/H/L
    candidates.append((['H', 'L', 'H', 'L'], [d_qw_H, d_qw_L, d_qw_H, d_qw_L]))

    # Candidate 7: 5-layer alternating starting with L
    candidates.append((['L', 'H', 'L', 'H', 'L'], [d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L]))

    # Candidate 8: 5-layer alternating starting with H
    candidates.append((['H', 'L', 'H', 'L', 'H'], [d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H]))

    # Candidate 9: 6-layer
    candidates.append((['L', 'H', 'L', 'H', 'L', 'H'],
                      [d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H]))

    # Candidate 10: 7-layer
    candidates.append((['L', 'H', 'L', 'H', 'L', 'H', 'L'],
                      [d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L]))

    # Candidate 11: 8-layer
    candidates.append((['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],
                      [d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H]))

    # Candidate 12: 9-layer
    candidates.append((['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'],
                      [d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L]))

    # Candidate 13: 10-layer
    candidates.append((['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],
                      [d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H, d_qw_L, d_qw_H]))

    # Also try some non-quarter-wave initial guesses
    # Broadband AR often benefits from non-uniform thicknesses
    # Try a symmetric 3-layer with different thicknesses
    d1 = center_wl / (4.0 * nL) * 0.8
    d2 = center_wl / (4.0 * nH) * 1.2
    candidates.append((['L', 'H', 'L'], [d1, d2, d1]))

    d1 = center_wl / (4.0 * nL) * 1.2
    d2 = center_wl / (4.0 * nH) * 0.8
    candidates.append((['L', 'H', 'L'], [d1, d2, d1]))

    # Evaluate and optimize each candidate
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')

    for labels, thicknesses in candidates:
        if time.time() > deadline:
            break

        # Check constraints
        if not check_constraints(labels, thicknesses):
            continue

        score = compute_mean_reflectance(labels, thicknesses)
        if score < best_score:
            best_score = score
            best_labels = list(labels)
            best_thicknesses = list(thicknesses)

        # Optimize this candidate
        opt_labels, opt_thick = optimize_thicknesses(labels, thicknesses, max_time=8.0)
        if check_constraints(opt_labels, opt_thick):
            opt_score = compute_mean_reflectance(opt_labels, opt_thick)
            if opt_score < best_score:
                best_score = opt_score
                best_labels = list(opt_labels)
                best_thicknesses = list(opt_thick)

    # Try additional random perturbations of the best solution
    if best_labels is not None and time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            perturbed = [max(0.0, min(max_layer_nm, t + np.random.uniform(-20, 20)))
                        for t in best_thicknesses]
            total = sum(perturbed)
            if total > max_total_nm:
                scale = max_total_nm / total
                perturbed = [t * scale for t in perturbed]
            if check_constraints(best_labels, perturbed):
                score = compute_mean_reflectance(best_labels, perturbed)
                if score < best_score:
                    best_score = score
                    best_thicknesses = list(perturbed)

    # Final check
    if best_labels is None:
        # Fallback to single quarter-wave L
        return ['L'], [d_qw_L]

    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
