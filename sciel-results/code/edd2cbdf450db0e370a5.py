"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """Compute reflectance (0-1) for a stack using the characteristic matrix method."""
    n = len(n_layers)
    if n == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.abs(r) ** 2

    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)

    for i in range(n):
        ni = n_layers[i]
        di = d_layers[i]
        delta = 2.0 * np.pi * ni * di / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        L11 = cos_d
        L12 = 1j * sin_d / ni
        L21 = 1j * ni * sin_d
        L22 = cos_d
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11 = new_M11; M12 = new_M12; M21 = new_M21; M22 = new_M22

    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    return np.abs(r) ** 2


def _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers)
    return float(np.mean(R) * 100.0)


def _is_valid(materials, thicknesses, problem):
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    total = sum(thicknesses)
    if total > max_total_nm + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    for m in materials:
        if m not in problem["materials"]:
            return False
    return True


def _evaluate_design(materials, thicknesses, problem, wavelengths):
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_layers = np.array([problem["materials"][m] for m in materials], dtype=float)
    d_layers = np.array(thicknesses, dtype=float)
    return _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers)


def _optimize_center_wavelength(mats, problem, lambda0, deadline, max_iter=2000):
    """
    Optimize thicknesses to minimize reflectance at a single wavelength (center).
    This provides a high-quality initial guess for broadband optimization.
    """
    n = len(mats)
    if n == 0:
        return []
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    n_layers = np.array([problem["materials"][m] for m in mats], dtype=float)
    d_layers = np.zeros(n, dtype=float)
    wavelengths = np.array([lambda0])
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]

    def objective(x):
        x = np.clip(x, 0.0, max_layer_nm)
        penalty = 0.0
        total = x.sum()
        if total > max_total_nm:
            penalty += (total - max_total_nm) * 10.0
        d_layers[:] = x
        R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers)
        return float(R[0]) + penalty

    # Initial guess: Quarter wave
    x0 = np.array([lambda0 / (4.0 * ni) for ni in n_layers])
    x0 = np.clip(x0, 0.0, max_layer_nm)
    if x0.sum() > max_total_nm:
        x0 *= max_total_nm / x0.sum()

    # Simple Nelder-Mead for 1D objective
    step = 10.0
    simplex = [x0.copy() for _ in range(n + 1)]
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1][i] = min(x0[i] + step, max_layer_nm)
    fvals = [objective(s) for s in simplex]

    alpha = 1.0; gamma = 2.0; rho = 0.5; sigma = 0.5
    for it in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if max(fvals) - min(fvals) < 1e-12:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

    best_x = np.clip(simplex[0], 0.0, max_layer_nm)
    if best_x.sum() > max_total_nm:
        best_x *= max_total_nm / best_x.sum()
    return list(best_x)


def _nelder_mead(mats, x0, problem, wavelengths, deadline, max_iter=3000):
    """Nelder-Mead optimization with constraint handling."""
    n = len(mats)
    if n == 0:
        return list(x0), _evaluate_design(mats, list(x0), problem, wavelengths)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    n_layers = np.array([problem["materials"][m] for m in mats], dtype=float)
    d_layers = np.zeros(n, dtype=float)

    def objective(x):
        x = np.clip(x, 0.0, max_layer_nm)
        penalty = 0.0
        total = x.sum()
        if total > max_total_nm:
            penalty += (total - max_total_nm) * 0.5
        d_layers[:] = x
        R = _char_matrix_method(wavelengths, problem["ambient_index"], problem["substrate_index"], n_layers, d_layers)
        return float(np.mean(R) * 100.0) + penalty

    x0 = np.clip(np.asarray(x0, dtype=float), 0.0, max_layer_nm)
    step = 20.0
    simplex = [x0.copy() for _ in range(n + 1)]
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1][i] = min(x0[i] + step, max_layer_nm)
    fvals = [objective(s) for s in simplex]

    alpha = 1.0; gamma = 2.0; rho = 0.5; sigma = 0.5
    for it in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if max(fvals) - min(fvals) < 1e-12:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

    best_x = np.clip(simplex[0], 0.0, max_layer_nm)
    if best_x.sum() > max_total_nm:
        scale = max_total_nm / best_x.sum()
        best_x *= scale
    best_f = _evaluate_design(mats, list(best_x), problem, wavelengths)
    return list(best_x), best_f


def _polish_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=3000):
    """High-precision coordinate descent polish."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    step_sizes = [1.0] * n
    min_step = 0.001
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            for factor in [1.0, 0.5, 0.25, 0.1]:
                step = step_sizes[i] * factor
                for sign in (1, -1):
                    new_thick = current_thicks[i] + sign * step
                    if new_thick < 0:
                        new_thick = 0.0
                    if new_thick > max_layer_nm:
                        new_thick = max_layer_nm
                    new_thicks = list(current_thicks)
                    new_thicks[i] = new_thick
                    if _is_valid(mats, new_thicks, problem):
                        new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_thick = new_thick
            if best_local_score < current_score - 1e-10:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        if not improved:
            step_sizes = [s * 0.5 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.2, 5.0) for s in step_sizes]
    return mats, current_thicks, current_score


def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s wall-clock kill

    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    centre = 0.5 * (band_nm[0] + band_nm[1])

    def qw(m):
        return centre / (4.0 * materials_dict[m])

    # --- Generate diverse index sequences ---
    sequences = []
    # Alternating sequences
    for n in range(1, min(15, max_layers + 1)):
        sequences.append(["H" if i % 2 == 0 else "L" for i in range(n)])
        sequences.append(["L" if i % 2 == 0 else "H" for i in range(n)])
    # Some non-alternating patterns that might be beneficial
    sequences.append(["H", "L", "L", "H", "L", "L", "H"])
    sequences.append(["L", "H", "H", "L", "H", "H", "L"])
    sequences.append(["H", "L", "H", "L", "L", "H", "L", "H"])
    sequences.append(["L", "H", "L", "H", "H", "L", "H", "L"])
    # Classic 10-layer
    inner = ["L", "H", "L", "H", "L", "H", "L", "H", "L"]
    sequences.append(["H"] + inner + ["H"])
    
    # Remove duplicates
    unique_sequences = []
    seen = set()
    for seq in sequences:
        key = tuple(seq)
        if key not in seen and len(seq) <= max_layers:
            seen.add(key)
            unique_sequences.append(seq)

    # --- Phase 1: Center Wavelength Optimization for each sequence ---
    candidates = []
    for seq in unique_sequences:
        if time.time() > deadline - 30.0:
            break
        # Optimize thicknesses for center wavelength
        thicks = _optimize_center_wavelength(seq, problem, centre, deadline - 20.0, max_iter=1000)
        if not thicks:
            continue
        # Check validity
        if not _is_valid(seq, thicks, problem):
            continue
        # Evaluate broadband reflectance
        score = _evaluate_design(seq, thicks, problem, wavelengths)
        candidates.append((score, seq, thicks))

    if not candidates:
        # Fallback
        best_materials = ["L"]
        best_thicknesses = [qw("L")]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    else:
        # Sort by score
        candidates.sort(key=lambda x: x[0])
        best_score, best_materials, best_thicknesses = candidates[0]

    # --- Phase 2: Broadband Optimization on the best candidates ---
    # Try top 3 candidates with full Nelder-Mead
    for score, mats, thicks in candidates[:3]:
        if time.time() > deadline - 10.0:
            break
        if mats == best_materials:
            continue
        # Optimize this candidate
        new_thicks, new_score = _nelder_mead(mats, thicks, problem, wavelengths, deadline - 5.0, max_iter=2000)
        if time.time() < deadline - 4.0:
            _, new_thicks, new_score = _polish_thicknesses(mats, new_thicks, problem, wavelengths, deadline - 3.0, max_iter=1000)
        if new_score < best_score:
            best_score = new_score
            best_materials = list(mats)
            best_thicknesses = list(new_thicks)

    # --- Phase 3: Final Polish on the global best ---
    if time.time() < deadline - 5.0:
        best_thicknesses, best_score = _nelder_mead(best_materials, best_thicknesses, problem, wavelengths, deadline - 2.0, max_iter=3000)
    if time.time() < deadline - 2.0:
        _, best_thicknesses, best_score = _polish_thicknesses(best_materials, best_thicknesses, problem, wavelengths, deadline - 0.5, max_iter=4000)

    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [qw("L")]

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
