import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 148.0

    # Unit hexagon vertices (flat-top, vertex at angle 0)
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Outer hexagon face normals (flat-top)
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    def get_hex_verts(center, angle_deg):
        """Get vertices of a single unit hexagon."""
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex_verts @ R.T
        verts += np.array(center)
        return verts

    def calc_side(centers, angles):
        """Calculate required outer hexagon side length."""
        max_apothem = 0.0
        for k in range(len(centers)):
            verts = get_hex_verts(centers[k], angles[k])
            projs = verts @ normals.T
            max_apothem = max(max_apothem, np.max(projs))
        return 2.0 * max_apothem / np.sqrt(3)

    def check_non_overlap(centers, min_dist=np.sqrt(3) * 0.999):
        """Check if hexagons are non-overlapping."""
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    return False
        return True

    def objective(centers, angles, penalty_weight=10000.0):
        """Objective: side length + overlap penalty."""
        side = calc_side(centers, angles)
        penalty = 0.0
        min_dist = np.sqrt(3) * 0.999
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * penalty_weight
        return side + penalty

    d = np.sqrt(3)  # Adjacent center distance in hex lattice

    best_side = float('inf')
    best_centers = None
    best_angles = None

    # Generate candidate initial configurations
    initial_configs = []

    # Config 1: 7-hex cluster + 4 outer
    centers1 = np.array([
        [0, 0],
        [d, 0], [d/2, 1.5], [-d/2, 1.5],
        [-d, 0], [-d/2, -1.5], [d/2, -1.5],
        [2*d, 0], [1.5*d, 1.5*1.5], [-1.5*d, 1.5*1.5],
        [-2*d, 0]
    ])
    initial_configs.append((centers1, np.zeros(N)))

    # Config 2: Compact triangular (rows of 3,4,3,1)
    centers2 = []
    for i in range(3):
        centers2.append([i*d - d, 0])
    for i in range(4):
        centers2.append([i*d - 1.5*d + d/2, 1.5])
    for i in range(3):
        centers2.append([i*d - d, 3.0])
    centers2.append([0, 4.5])
    centers2 = np.array(centers2[:N])
    initial_configs.append((centers2, np.zeros(N)))

    # Config 3: Two rows (5+6)
    centers3 = []
    for i in range(5):
        centers3.append([i*d - 2*d, 0])
    for i in range(6):
        centers3.append([i*d - 2.5*d + d/2, 1.5])
    centers3 = np.array(centers3[:N])
    initial_configs.append((centers3, np.zeros(N)))

    # Config 4: 4-3-4 arrangement
    centers4 = []
    for i in range(4):
        centers4.append([i*d - 1.5*d, 0])
    for i in range(3):
        centers4.append([i*d - d + d/2, 1.5])
    for i in range(4):
        centers4.append([i*d - 1.5*d, 3.0])
    centers4 = np.array(centers4[:N])
    initial_configs.append((centers4, np.zeros(N)))

    # Config 5: Random scatter
    np.random.seed(42)
    centers5 = np.random.uniform(-3*d, 3*d, (N, 2))
    initial_configs.append((centers5, np.zeros(N)))

    # Local search with coordinate descent
    def local_search(init_centers, init_angles, max_iters=5000, step_size=0.05):
        current_centers = init_centers.copy().astype(float)
        current_angles = init_angles.copy().astype(float)
        current_obj = objective(current_centers, current_angles)

        best_centers = current_centers.copy()
        best_angles = current_angles.copy()
        best_obj = current_obj

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dangle = np.random.normal(0, step_size * 5)

            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle

            new_obj = objective(new_centers, new_angles)

            if new_obj < current_obj:
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj

                if current_obj < best_obj:
                    best_obj = current_obj
                    best_centers = current_centers.copy()
                    best_angles = current_angles.copy()

            # Decay step size
            step_size *= 0.9998

        return best_centers, best_angles, best_obj

    # Run local search from each configuration
    for config_idx, (init_centers, init_angles) in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break

        try:
            result_centers, result_angles, result_obj = local_search(
                init_centers, init_angles, max_iters=8000, step_size=0.08
            )

            if check_non_overlap(result_centers):
                side = calc_side(result_centers, result_angles)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
        except Exception:
            pass

    # High-precision polish pass
    if best_centers is not None and time.time() < deadline - 5:
        current_centers = best_centers.copy()
        current_angles = best_angles.copy()
        current_obj = objective(current_centers, current_angles)

        step_size = 0.02
        for it in range(10000):
            if time.time() > deadline - 3:
                break

            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dangle = np.random.normal(0, step_size * 3)

            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle

            new_obj = objective(new_centers, new_angles)

            if new_obj < current_obj:
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj

            step_size *= 0.9997

        if current_obj < objective(best_centers, best_angles):
            best_centers = current_centers
            best_angles = current_angles
            best_side = calc_side(best_centers, best_angles)

    # Final validation and output
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Verify no overlaps
    if not check_non_overlap(best_centers):
        # Fallback to a safe configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x, y = best_centers[i, 0], best_centers[i, 1]
        ang = best_angles[i]
        inner.append([float(x), float(y), float(ang)])

    return inner, (0.0, 0.0), float(best_side), 0.0
