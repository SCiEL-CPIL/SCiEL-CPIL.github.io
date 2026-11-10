# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 580  # Safety margin before 600s limit
    n = 14

    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0 or np.any(~np.isfinite(dists)):
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return points / norms

    def fibonacci_sphere(n_pts):
        golden = (1 + np.sqrt(5)) / 2
        i = np.arange(n_pts)
        z = 1 - (2 * i + 1) / n_pts
        r = np.sqrt(np.maximum(0, 1 - z**2))
        theta = 2 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack([x, y, z])
        return pts

    def dodecahedron_based(n_pts):
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
            [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
            [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        
        selected = [0]
        while len(selected) < n_pts:
            best_idx = -1
            best_min_dist = -1
            for idx in range(len(verts)):
                if idx in selected:
                    continue
                min_d = min(np.linalg.norm(verts[idx] - verts[s]) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(best_idx)
        
        return verts[selected]

    def icosahedron_based(n_pts):
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        
        if n_pts <= 12:
            return verts[:n_pts]
        
        pts = verts.copy()
        remaining = n_pts - 12
        
        for _ in range(remaining):
            best_pos = None
            best_min_dist = -1
            np.random.seed(42 + len(pts))
            for _ in range(3000):
                candidate = np.random.randn(3)
                candidate = candidate / np.linalg.norm(candidate)
                dists = np.linalg.norm(pts - candidate, axis=1)
                min_d = np.min(dists)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = candidate
            if best_pos is not None:
                pts = np.vstack([pts, best_pos])
        
        return pts[:n_pts]

    def local_polish(points, max_iter=5000, tol=1e-10):
        """
        Local polish using gradient ascent on the ratio objective.
        Projects points back to unit sphere after each step.
        """
        points = project_to_sphere(points.copy())
        current_score = calculate_score(points)
        
        for it in range(max_iter):
            if time.time() - start_time > timeout - 10:
                break
                
            # Compute pairwise distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            dist_matrix = np.sqrt(np.maximum(dist_sq, 0))
            
            mask = ~np.eye(n, dtype=bool)
            dists = dist_matrix[mask]
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max < 1e-12:
                break
            
            # Find min and max pairs
            min_pairs = np.argwhere(dist_matrix <= d_min + 1e-8)
            max_pairs = np.argwhere(dist_matrix >= d_max - 1e-8)
            
            # Build gradient
            grad = np.zeros_like(points)
            
            # For min distance: push apart (increase d_min)
            min_weight = 1.0 / d_max
            for idx in min_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad[i_idx] += min_weight * unit
                    grad[j_idx] -= min_weight * unit
            
            # For max distance: push together (decrease d_max)
            max_weight = d_min / (d_max ** 2)
            for idx in max_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad[i_idx] -= max_weight * unit
                    grad[j_idx] += max_weight * unit
            
            # Project to tangent space
            for k in range(n):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            grad = grad / grad_norm
            
            # Adaptive step size with backtracking
            step_lr = 0.005
            improved = False
            for attempt in range(15):
                new_points = project_to_sphere(points + step_lr * grad)
                new_score = calculate_score(new_points)
                if new_score > current_score + 1e-12:
                    points = new_points
                    current_score = new_score
                    improved = True
                    break
                step_lr *= 0.5
            
            if not improved:
                break
        
        return points, current_score

    def simulated_annealing(init_points, n_iter=50000, t_start=1.0, t_end=0.01):
        """
        Discrete Simulated Annealing on the sphere.
        Moves: pick a random point and perturb it by a random small rotation.
        Accept if score improves or with probability exp(-delta/T).
        """
        points = project_to_sphere(init_points.copy())
        current_score = calculate_score(points)
        best_points = points.copy()
        best_score = current_score
        
        n_iter = min(n_iter, max(1000, int((timeout - (time.time() - start_time) - 20) * 100)))
        
        for it in range(n_iter):
            if time.time() - start_time > timeout - 15:
                break
            
            # Temperature schedule (geometric cooling)
            progress = it / max(n_iter - 1, 1)
            T = t_start * (t_end / t_start) ** progress
            
            # Generate move: pick random point, apply random small rotation
            idx = np.random.randint(n)
            # Random rotation axis
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            # Random small angle
            angle = np.random.randn() * 0.15
            
            # Apply rotation to selected point using Rodrigues' formula
            p = points[idx]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            cross = np.cross(axis, p)
            dot = np.dot(axis, p)
            p_new = p * cos_a + cross * sin_a + axis * dot * (1 - cos_a)
            
            new_points = points.copy()
            new_points[idx] = p_new
            new_points = project_to_sphere(new_points)
            
            new_score = calculate_score(new_points)
            delta = new_score - current_score
            
            # Accept if better or with probability exp(-delta/T)
            if delta > 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
        
        return best_points, best_score

    best_score = 0.0
    best_points = None

    # Generate multiple initializations
    inits = []
    inits.append(fibonacci_sphere(n))
    inits.append(dodecahedron_based(n))
    inits.append(icosahedron_based(n))

    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))

    for seed in [777, 888, 999, 1111, 2222, 3333]:
        np.random.seed(seed)
        inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.3)

    # Phase 1: Run SA on each initialization
    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout - 30:
            break

        pts = project_to_sphere(init.copy())
        s_init = calculate_score(pts)
        if s_init > best_score:
            best_score = s_init
            best_points = pts.copy()

        # Run SA with time budget
        sa_time_budget = min(20, (timeout - (time.time() - start_time) - 30) / max(len(inits) - init_idx, 1))
        n_sa_iter = int(sa_time_budget * 5000)  # ~5000 iters per second
        
        pts_sa, s_sa = simulated_annealing(pts, n_iter=max(n_sa_iter, 1000))
        if s_sa > best_score:
            best_score = s_sa
            best_points = pts_sa.copy()

    # Phase 2: Local polish on best points found
    if best_points is not None and time.time() - start_time < timeout - 10:
        pts_polished, s_polished = local_polish(best_points, max_iter=10000)
        if s_polished > best_score:
            best_score = s_polished
            best_points = pts_polished.copy()

    # Phase 3: More SA rounds with the polished result as seed
    if time.time() - start_time < timeout - 20:
        for round in range(3):
            if time.time() - start_time > timeout - 15:
                break
            n_sa_iter = int(min(30, (timeout - (time.time() - start_time) - 15)) * 3000)
            pts_sa, s_sa = simulated_annealing(best_points, n_iter=max(n_sa_iter, 2000))
            if s_sa > best_score:
                best_score = s_sa
                best_points = pts_sa.copy()
            
            # Polish again
            if time.time() - start_time < timeout - 10:
                pts_polished, s_polished = local_polish(best_points, max_iter=5000)
                if s_polished > best_score:
                    best_score = s_polished
                    best_points = pts_polished.copy()

    if best_points is None:
        best_points = fibonacci_sphere(n)

    best_points = project_to_sphere(best_points)
    return best_points

# EVOLVE-BLOCK-END
