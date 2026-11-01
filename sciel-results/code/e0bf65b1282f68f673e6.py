import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def objective_smooth(x, tau=100.0):
    """
    Smoothed objective function using log-sum-exp for max and min.
    Minimizes log(max_sq_dist) - log(min_sq_dist).
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max of squared distances
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min of squared distances
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    # Objective is log(max) - log(min)
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def objective_epigraph(x, tau=100.0):
    """
    Epigraph-style objective: minimize t such that all sq_dists are in [1/t, t].
    We parameterize with log(t) and use soft constraints.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft max and min
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    # We want to minimize log(max/min) = log(max) - log(min)
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def fibonacci_sphere(n):
    """Generate n points on a sphere using the Fibonacci sphere method."""
    points = []
    golden_angle = np.pi * (3. - np.sqrt(5.))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # --- Build diverse initial candidates ---
    coord_candidates = []
    
    # 1. Fibonacci sphere
    fib = fibonacci_sphere(N)
    coord_candidates.append(fib.flatten())
    
    # 2. Random points on a sphere (multiple)
    for i in range(15):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        coord_candidates.append(pts.flatten())
        
    # 3. Random points in a cube
    for i in range(15):
        pts = rng.uniform(-1, 1, size=(N, D))
        coord_candidates.append(pts.flatten())

    # 4. Icosahedron + 2 extra points (symmetric)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 points at poles of a slightly larger sphere
    extra = np.array([[0, 0, 1.3], [0, 0, -1.3]])
    ico_extra = np.vstack([ico, extra])
    coord_candidates.append(ico_extra.flatten())
    
    # 5. Cube vertices + face centers (14 points: 8 vertices + 6 face centers)
    cube_verts = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], dtype=float)
    face_centers = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    cube_face = np.vstack([cube_verts, face_centers])
    coord_candidates.append(cube_face.flatten())
    
    # 6. Truncated tetrahedron approximation (12 vertices) + 2
    # Truncated tetrahedron vertices
    tt = []
    # 4 original tetrahedron vertices truncated
    tetra = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    # Add midpoints of edges
    for i in range(4):
        for j in range(i+1, 4):
            mid = (tetra[i] + tetra[j]) / 2
            mid = mid / np.linalg.norm(mid)
            tt.append(mid)
    tt = np.array(tt)
    # That gives us 6 edge midpoints + 4 truncated vertices = 10, need 14
    # Add 2 more
    extra_tt = rng.normal(size=(4, D)) * 0.3
    extra_tt = extra_tt / np.linalg.norm(extra_tt, axis=1, keepdims=True)
    tt_full = np.vstack([tetra, tt, extra_tt[:2]])
    coord_candidates.append(tt_full.flatten())
    
    # 7. Octahedron (6) + cube vertices (8) = 14
    octa = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
    octa_cube = np.vstack([octa, cube_verts])
    coord_candidates.append(octa_cube.flatten())
    
    # 8. Scaled icosahedron with 2 points inside
    ico_inner = ico * 0.7
    inner_pts = rng.normal(size=(2, D)) * 0.5
    ico_mixed = np.vstack([ico_inner, inner_pts])
    coord_candidates.append(ico_mixed.flatten())
    
    # --- Multi-stage optimization with many restarts ---
    n_coord_restarts = 60
    tau_schedule = [3.0, 10.0, 50.0, 200.0, 1000.0]
    
    for i in range(n_coord_restarts):
        if time.time() - start_time > timeout - 25:
            break
            
        if i < len(coord_candidates):
            x0 = coord_candidates[i]
        else:
            # Mix of random sphere and random cube
            if i % 2 == 0:
                x0 = rng.normal(size=(N, D)).flatten()
                x0 = x0 / np.linalg.norm(x0.reshape(N, D), axis=1, keepdims=True).flatten()
            else:
                x0 = rng.uniform(-1, 1, size=(N * D))
            
        try:
            current_x = x0.copy()
            
            for tau in tau_schedule:
                if time.time() - start_time > timeout - 15:
                    break
                    
                res = minimize(
                    objective_smooth, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    current_x = res.x
                    pts = current_x.reshape(N, D)
                    update_best(pts)
                    
        except Exception:
            pass

    # --- Final polish: high-precision optimization from best points ---
    if best_points is not None and time.time() - start_time < timeout - 3:
        try:
            # Try several high-tau polishes
            for tau_polish in [500.0, 2000.0, 10000.0]:
                if time.time() - start_time > timeout - 2:
                    break
                res_final = minimize(
                    objective_smooth, 
                    best_points.flatten(), 
                    method='L-BFGS-B', 
                    args=(tau_polish,),
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_final.x is not None:
                    pts_final = res_final.x.reshape(N, D)
                    update_best(pts_final)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = fib
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
