# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 30.0


def _score(points: np.ndarray) -> float:
    """Compute (d_min / d_max)^2, with safeguards for degenerate cases."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if not (np.isfinite(d_min) and np.isfinite(d_max)):
        return 0.0
    if d_max <= 1e-15:
        return 0.0
    if d_min <= 1e-15:
        return 0.0
    ratio = d_min / d_max
    return float(ratio * ratio)


def _simulated_annealing(points: np.ndarray, deadline: float, seed: int = 42) -> np.ndarray:
    """
    Simulated annealing on the unit sphere for 14 points in 3D.
    Uses random perturbations and acceptance based on score improvement.
    """
    rng = np.random.default_rng(seed)
    best_pts = points.copy()
    best_score = _score(best_pts)
    current_pts = best_pts.copy()
    current_score = best_score

    T0 = 0.1
    T_end = 1e-8
    cooling_rate = 0.9997
    max_steps = 200000

    T = T0
    step = 0

    while step < max_steps:
        if time.time() > deadline:
            break

        # Random perturbation: pick a random point and move it slightly
        i = rng.integers(0, 14)
        # Random direction on sphere
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)

        # Perturbation size
        perturbation_size = T * 0.5
        new_pt = current_pts[i] + perturbation_size * direction
        new_pt = new_pt / np.linalg.norm(new_pt)

        trial_pts = current_pts.copy()
        trial_pts[i] = new_pt
        trial_score = _score(trial_pts)

        # Acceptance criterion
        delta = trial_score - current_score
        if delta > 0 or rng.uniform() < np.exp(delta / T):
            current_pts = trial_pts
            current_score = trial_score

            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()

        T *= cooling_rate
        if T < T_end:
            T = T_end
            break

        step += 1

    return best_pts


def _coordinate_descent(points: np.ndarray, deadline: float, seed: int = 42) -> np.ndarray:
    """
    Coordinate descent: optimize each point one at a time using local search.
    """
    rng = np.random.default_rng(seed)
    best_pts = points.copy()
    best_score = _score(best_pts)

    n_iterations = 0
    max_iterations = 50000

    while n_iterations < max_iterations:
        if time.time() > deadline:
            break

        improved = False
        for i in range(14):
            if time.time() > deadline:
                break

            # Try to improve point i by local search
            current_pt = best_pts[i]
            best_local_pt = current_pt
            best_local_score = best_score

            # Local search around current point
            for _ in range(200):
                if time.time() > deadline:
                    break

                # Random perturbation
                direction = rng.normal(size=3)
                direction = direction / np.linalg.norm(direction)
                step_size = 0.05 * (0.5 ** (n_iterations // 100))

                candidate = current_pt + step_size * direction
                candidate = candidate / np.linalg.norm(candidate)

                trial_pts = best_pts.copy()
                trial_pts[i] = candidate
                trial_score = _score(trial_pts)

                if trial_score > best_local_score:
                    best_local_score = trial_score
                    best_local_pt = candidate
                    current_pt = candidate

            if best_local_score > best_score + 1e-12:
                best_pts[i] = best_local_pt
                best_score = best_local_score
                improved = True

        n_iterations += 1

        if not improved and n_iterations > 10:
            break

    return best_pts


def _refine_with_gradient(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    Use gradient-based optimization to refine the configuration.
    """
    def objective(x):
        pts = x.reshape(14, 3)
        # Normalize to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms_safe = np.where(norms > 1e-15, norms, 1.0)
        pts = pts / norms_safe
        return -_score(pts)

    # Initial point
    x0 = points.flatten()

    # Use Nelder-Mead for robustness
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        if result.fun < 0:
            refined_pts = result.x.reshape(14, 3)
            norms = np.linalg.norm(refined_pts, axis=1, keepdims=True)
            norms_safe = np.where(norms > 1e-15, norms, 1.0)
            refined_pts = refined_pts / norms_safe
            if _score(refined_pts) > _score(points):
                return refined_pts
    except Exception:
        pass

    return points


def _generate_structured_guesses() -> list:
    """
    Generate structured initial guesses based on symmetry and known good configurations.
    """
    guesses = []
    rng = np.random.default_rng(12345)

    # 1. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    ico_pts = np.array(ico_verts, dtype=np.float64)
    norms = np.linalg.norm(ico_pts, axis=1, keepdims=True)
    ico_pts = ico_pts / norms

    # Add 2 points to maximize minimum distance
    pts = ico_pts.copy()
    for _ in range(2):
        best_candidate = None
        best_min_dist = -1
        for _ in range(10000):
            candidate = rng.normal(size=3)
            candidate = candidate / np.linalg.norm(candidate)
            dists = np.linalg.norm(pts - candidate, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate.copy()
        if best_candidate is not None:
            pts = np.vstack([pts, best_candidate])
    guesses.append(pts)

    # 2. Cube + Tetrahedron + 2 more
    cube_verts = [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ]
    cube_pts = np.array(cube_verts, dtype=np.float64)
    norms = np.linalg.norm(cube_pts, axis=1, keepdims=True)
    cube_pts = cube_pts / norms

    tetra_verts = [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ]
    tetra_pts = np.array(tetra_verts, dtype=np.float64)
    norms = np.linalg.norm(tetra_pts, axis=1, keepdims=True)
    tetra_pts = tetra_pts / norms

    combined = np.vstack([cube_pts, tetra_pts])
    for _ in range(2):
        best_candidate = None
        best_min_dist = -1
        for _ in range(10000):
            candidate = rng.normal(size=3)
            candidate = candidate / np.linalg.norm(candidate)
            dists = np.linalg.norm(combined - candidate, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate.copy()
        if best_candidate is not None:
            combined = np.vstack([combined, best_candidate])
    guesses.append(combined)

    # 3. Octahedron + 8 cube vertices
    octa_verts = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
    octa_pts = np.array(octa_verts, dtype=np.float64)
    guesses.append(np.vstack([octa_pts, cube_pts]))

    # 4. 2 poles + 12 equatorial
    pts = np.zeros((14, 3))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    for i in range(12):
        angle = 2 * np.pi * i / 12
        pts[i + 2] = [np.cos(angle), np.sin(angle), 0]
    guesses.append(pts)

    # 5. 2 poles + 6 at z=0.5 + 6 at z=-0.5
    pts = np.zeros((14, 3))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    r = np.sqrt(1 - 0.25)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts[i + 2] = [r * np.cos(angle), r * np.sin(angle), 0.5]
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        pts[i + 8] = [r * np.cos(angle), r * np.sin(angle), -0.5]
    guesses.append(pts)

    # 6. Fibonacci sphere (14 points)
    n_pts = 14
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = np.zeros((n_pts, 3))
    for i in range(n_pts):
        z = 1 - 2 * (i + 0.5) / n_pts
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    guesses.append(fib_pts)

    # 7. Random
    random_pts = np.zeros((14, 3))
    for i in range(14):
        pt = rng.normal(size=3)
        random_pts[i] = pt / np.linalg.norm(pt)
    guesses.append(random_pts)

    return guesses


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN

    # Generate initial guesses
    guesses = _generate_structured_guesses()

    # Initialize best
    best_pts = guesses[0].copy()
    best_score = _score(best_pts)

    # For each initial guess, run simulated annealing + coordinate descent + gradient refinement
    for i, guess in enumerate(guesses):
        if time.time() > deadline:
            break

        # Simulated annealing
        sa_pts = _simulated_annealing(guess, deadline, seed=1000 + i)
        sa_score = _score(sa_pts)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()

        # Coordinate descent
        cd_pts = _coordinate_descent(best_pts, deadline, seed=2000 + i)
        cd_score = _score(cd_pts)
        if cd_score > best_score:
            best_score = cd_score
            best_pts = cd_pts.copy()

        # Gradient refinement
        grad_pts = _refine_with_gradient(best_pts, deadline)
        grad_score = _score(grad_pts)
        if grad_score > best_score:
            best_score = grad_score
            best_pts = grad_pts.copy()

        # If we have time, do another round of SA + CD
        if time.time() < deadline - 60:
            sa_pts2 = _simulated_annealing(best_pts, deadline, seed=3000 + i)
            sa_score2 = _score(sa_pts2)
            if sa_score2 > best_score:
                best_score = sa_score2
                best_pts = sa_pts2.copy()

            cd_pts2 = _coordinate_descent(best_pts, deadline, seed=4000 + i)
            cd_score2 = _score(cd_pts2)
            if cd_score2 > best_score:
                best_score = cd_score2
                best_pts = cd_pts2.copy()

    # Final polish with gradient refinement
    if time.time() < deadline:
        final_pts = _refine_with_gradient(best_pts, deadline)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()

    # Validate and return
    best_pts = np.asarray(best_pts, dtype=np.float64)

    if best_pts.shape != (14, 3):
        best_pts = guesses[0]

    if not np.all(np.isfinite(best_pts)):
        best_pts = guesses[0]

    # Ensure points are on or near the unit sphere (renormalize)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-15, norms, 1.0)
    best_pts = best_pts / norms_safe

    return best_pts


# EVOLVE-BLOCK-END
