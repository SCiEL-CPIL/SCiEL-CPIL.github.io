"""Broadband antireflection coating design: tapered alternating stack + Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Precompute evaluation wavelengths
_WL = np.linspace(400.0, 800.0, 401)
_K = 2.0 * np.pi / _WL  # 1/nm, shape (401,)

N_AMB = 1.0
N_SUB = 1.52
N_L = 1.38
N_H = 2.35
MAX_LAYERS = 20
MAX_TOTAL_NM = 1000.0
MAX_LAYER_NM = 400.0


def _reflectance(labels, thicknesses):
    """Vectorized mean reflectance (%) over 401 wavelengths using characteristic matrix."""
    n = len(labels)
    if n == 0:
        r0 = (N_SUB - N_AMB) / (N_SUB + N_AMB)
        return 100.0 * (r0 * r0)

    N = len(_WL)
    # Start with identity matrix components (vectorized over wavelengths)
    M00 = np.ones(N, dtype=np.complex128)
    M01 = np.zeros(N, dtype=np.complex128)
    M10 = np.zeros(N, dtype=np.complex128)
    M11 = np.ones(N, dtype=np.complex128)

    for j in range(n):
        n_j = N_L if labels[j] == 'L' else N_H
        d_j = thicknesses[j]
        delta = _K * n_j * d_j  # (N,)
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        L00 = cos_d
        L01 = 1j * sin_d / n_j
        L10 = 1j * n_j * sin_d
        L11 = cos_d

        new_M00 = M00 * L00 + M01 * L10
        new_M01 = M00 * L01 + M01 * L11
        new_M10 = M10 * L00 + M11 * L10
        new_M11 = M10 * L01 + M11 * L11
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11

    # Admittance at front surface
    Y = M01 / M00
    R = np.abs((N_AMB - Y) / (N_AMB + Y)) ** 2
    return float(np.mean(R) * 100.0)


def _valid(labels, thicknesses):
    if len(labels) != len(thicknesses):
        return False
    if len(labels) > MAX_LAYERS or len(labels) == 0:
        return False
    total = 0.0
    for t in thicknesses:
        if t < -1e-9 or t > MAX_LAYER_NM + 1e-9:
            return False
        total += t
    if total > MAX_TOTAL_NM + 1e-9:
        return False
    for m in labels:
        if m not in ('L', 'H'):
            return False
    return True


def _clamp_thicknesses(thicknesses):
    return [min(max(t, 0.0), MAX_LAYER_NM) for t in thicknesses]


def _tapered_stack(n_layers, base_th, grad, start_with_L=True):
    """Generate a tapered alternating stack.
    Layer i thickness = base_th * (1.0 + grad * i), clamped to [0, MAX_LAYER_NM].
    If total exceeds MAX_TOTAL_NM, scale all down.
    """
    labels = []
    thicknesses = []
    for i in range(n_layers):
        labels.append('L' if (i % 2 == 0) == start_with_L else 'H')
        t = base_th * (1.0 + grad * i)
        t = min(max(t, 0.0), MAX_LAYER_NM)
        thicknesses.append(t)
    total = sum(thicknesses)
    if total > MAX_TOTAL_NM and total > 0:
        scale = MAX_TOTAL_NM / total
        thicknesses = [t * scale for t in thicknesses]
    return labels, thicknesses


def _nelder_mead(objective, x0, max_iter=300, tol=1e-7, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Nelder-Mead with convergence check."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = 0.25 * abs(xi[i]) if abs(xi[i]) > 1e-6 else 0.25
        xi[i] += step
        simplex.append(xi)
    fvals = [objective(s) for s in simplex]

    for it in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if abs(fvals[-1] - fvals[0]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    return simplex[0], fvals[0]


def _objective_tapered(n_layers, start_with_L, params):
    """Objective for tapered stack: params = [base_th, grad]."""
    base_th, grad = params
    if base_th < 0 or grad < -1.0 or grad > 1.0:
        return 100.0
    labels, thicknesses = _tapered_stack(n_layers, base_th, grad, start_with_L)
    if not _valid(labels, thicknesses):
        return 100.0
    return _reflectance(labels, thicknesses)


def _optimize_tapered(n_layers, start_with_L, time_budget):
    """Multi-start Nelder-Mead on (base_th, grad) for a given stack topology."""
    deadline = time.time() + time_budget
    best_score = 100.0
    best_params = None
    best_labels = None
    best_thicks = None

    # Multi-start points
    starts = []
    # Quarter-wave based starts
    centre = 600.0
    d_L = centre / (4.0 * N_L)  # ~108.7
    d_H = centre / (4.0 * N_H)  # ~63.8
    # base_th near the average of d_L and d_H, with various gradients
    avg_th = 0.5 * (d_L + d_H)
    for base in [avg_th * 0.5, avg_th * 0.75, avg_th, avg_th * 1.25, avg_th * 1.5]:
        for grad in [-0.3, -0.15, 0.0, 0.15, 0.3]:
            starts.append([base, grad])
    # Additional random starts
    rng = np.random.RandomState(42 + n_layers)
    for _ in range(20):
        base = rng.uniform(30.0, 150.0)
        grad = rng.uniform(-0.5, 0.5)
        starts.append([base, grad])

    for x0 in starts:
        if time.time() > deadline:
            break
        x_opt, f_opt = _nelder_mead(
            lambda p, nl=n_layers, swl=start_with_L: _objective_tapered(nl, swl, p),
            x0, max_iter=150
        )
        if f_opt < best_score:
            best_score = f_opt
            best_params = x_opt
            labels, thicknesses = _tapered_stack(n_layers, x_opt[0], x_opt[1], start_with_L)
            if _valid(labels, thicknesses):
                best_labels = labels
                best_thicks = thicknesses

    return best_score, best_labels, best_thicks


def _optimize_full_thicknesses(labels, thicknesses, time_budget):
    """Nelder-Mead on all thicknesses directly (for smaller stacks)."""
    deadline = time.time() + time_budget
    n = len(thicknesses)
    if n > 12:
        return 100.0, None, None

    def objective(x):
        thicks = list(x)
        if any(t < -1e-9 or t > MAX_LAYER_NM + 1e-9 for t in thicks):
            return 100.0
        if sum(thicks) > MAX_TOTAL_NM + 1e-9:
            return 100.0
        return _reflectance(labels, thicks)

    x0 = np.array(thicknesses, dtype=float)
    # Multi-start
    best_score = 100.0
    best_x = None
    rng = np.random.RandomState(123)
    starts = [x0.copy()]
    for _ in range(5):
        pert = x0 + rng.uniform(-0.2, 0.2, size=n) * np.maximum(x0, 10.0)
        starts.append(np.clip(pert, 0.0, MAX_LAYER_NM))

    for s in starts:
        if time.time() > deadline:
            break
        x_opt, f_opt = _nelder_mead(objective, s, max_iter=100)
        if f_opt < best_score:
            best_score = f_opt
            best_x = x_opt

    if best_x is not None:
        thicks = _clamp_thicknesses(list(best_x))
        # Final scale to fit total
        total = sum(thicks)
        if total > MAX_TOTAL_NM:
            scale = MAX_TOTAL_NM / total
            thicks = [t * scale for t in thicks]
        if _valid(labels, thicks):
            return best_score, labels, thicks
    return 100.0, None, None


def design(problem):
    """Design a broadband antireflection coating."""
    deadline = time.time() + 140.0  # Safety margin under 150s wall-clock

    best_score = 100.0
    best_labels = ['L']
    best_thicks = [600.0 / (4.0 * N_L)]

    # Try tapered alternating stacks for various n_layers
    n_layers_options = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    for n_layers in n_layers_options:
        if time.time() > deadline - 15.0:
            break
        # Try both starting with L and H
        for start_with_L in [True, False]:
            if time.time() > deadline - 10.0:
                break
            time_budget = min(12.0, max(2.0, (deadline - time.time()) - 10.0))
            score, labels, thicks = _optimize_tapered(n_layers, start_with_L, time_budget)
            if score < best_score:
                best_score = score
                best_labels = labels
                best_thicks = thicks

    # Refine the best design with full thickness optimization (if small enough)
    if len(best_thicks) <= 12 and time.time() < deadline - 15.0:
        time_budget = min(20.0, max(3.0, (deadline - time.time()) - 10.0))
        score, labels, thicks = _optimize_full_thicknesses(best_labels, best_thicks, time_budget)
        if score < best_score:
            best_score = score
            best_labels = labels
            best_thicks = thicks

    # Final validity check
    if not _valid(best_labels, best_thicks):
        centre = 600.0
        best_labels = ['L']
        best_thicks = [centre / (4.0 * N_L)]

    return best_labels, best_thicks
# EVOLVE-BLOCK-END
