"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

def _validate(packing):
    """Check if packing is valid: disjoint interiors, inside unit square."""
    if len(packing) != N:
        return False
    for x, y, r in packing:
        if r <= 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ri = packing[i]
            xj, yj, rj = packing[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True

def _score(packing):
    """Sum of radii."""
    return sum(packing[:, 2])

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges as index pairs."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a = simplex[i]
                b = simplex[(i+1) % 3]
                if a != b:
                    edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        return []

def _spring_forces(centers, radii, edges, k_spring=10.0, k_repel=100.0, k_wall=50.0):
    """Compute forces on each circle."""
    n = len(centers)
    forces = np.zeros((n, 2))
    
    # Spring forces along Delaunay edges (attract to tangency)
    for i, j in edges:
        dx = centers[j][0] - centers[i][0]
        dy = centers[j][1] - centers[i][1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 1e-10:
            continue
        target_dist = radii[i] + radii[j]
        # Force to push/pull to target distance
        f = k_spring * (dist - target_dist)
        fx = f * dx / dist
        fy = f * dy / dist
        forces[i] += [fx, fy]
        forces[j] -= [fx, fy]
    
    # Repulsive forces for all pairs (prevent overlap)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-10:
                continue
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                # Strong repulsion when overlapping
                overlap = min_dist - dist
                f = k_repel * overlap
                fx = f * dx / dist
                fy = f * dy / dist
                forces[i] -= [fx, fy]
                forces[j] += [fx, fy]
    
    # Wall forces
    for i in range(n):
        r = radii[i]
        # Left wall
        if centers[i][0] - r < 0:
            forces[i][0] += k_wall * (r - centers[i][0])
        # Right wall
        if centers[i][0] + r > 1:
            forces[i][0] -= k_wall * (centers[i][0] + r - 1)
        # Bottom wall
        if centers[i][1] - r < 0:
            forces[i][1] += k_wall * (r - centers[i][1])
        # Top wall
        if centers[i][1] + r > 1:
            forces[i][1] -= k_wall * (centers[i][1] + r - 1)
    
    return forces

def _optimize_from_initial(centers, radii, max_iter=5000, deadline=None):
    """Optimize packing using spring-based force simulation."""
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score = _score(np.column_stack([centers, radii]))
    
    # Adaptive step size
    dt = 0.01
    damping = 0.9
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Get Delaunay edges
        edges = _get_delaunay_edges(centers)
        
        # Compute forces
        forces = _spring_forces(centers, radii, edges)
        
        # Update positions with damping
        centers += dt * forces
        centers *= damping + (1-damping)  # This is wrong, let's just use velocity-based
        
        # Actually, let's use a simpler momentum-based approach
        # Reset: use simple gradient descent with adaptive step
        # Break out and use a different approach
    
    return best_centers, best_radii

def _optimize_gradient(centers, radii, max_iter=20000, deadline=None):
    """Gradient-based optimization with Delaunay-aware repulsion."""
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score = _score(np.column_stack([centers, radii]))
    
    lr_pos = 0.005
    lr_rad = 0.001
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Get Delaunay edges
        edges = _get_delaunay_edges(centers)
        edge_set = set((min(i,j), max(i,j)) for i,j in edges)
        
        # Compute gradient for positions
        grad_pos = np.zeros((n, 2))
        grad_rad = np.zeros(n)
        
        # Repulsion forces (all pairs)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dx, dy = 1e-8, 0
                    dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                
                # Penalty for overlap
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Force proportional to overlap
                    f = overlap
                    fx = f * dx / dist
                    fy = f * dy / dist
                    grad_pos[i] -= [fx, fy]
                    grad_pos[j] += [fx, fy]
                    # Gradient for radii: increasing radius increases overlap
                    grad_rad[i] -= 1.0
                    grad_rad[j] -= 1.0
                else:
                    # Weak attraction for nearby circles (Delaunay neighbors)
                    if (min(i,j), max(i,j)) in edge_set and dist < min_dist * 1.5:
                        # Mild attraction to bring them closer
                        f = 0.1 * (dist - min_dist)
                        fx = f * dx / dist
                        fy = f * dy / dist
                        grad_pos[i] -= [fx, fy]
                        grad_pos[j] += [fx, fy]
        
        # Wall constraints
        for i in range(n):
            r = radii[i]
            # Left
            if centers[i][0] - r < 0:
                grad_pos[i][0] -= 10.0 * (r - centers[i][0])
            # Right
            if centers[i][0] + r > 1:
                grad_pos[i][0] += 10.0 * (centers[i][0] + r - 1)
            # Bottom
            if centers[i][1] - r < 0:
                grad_pos[i][1] -= 10.0 * (r - centers[i][1])
            # Top
            if centers[i][1] + r > 1:
                grad_pos[i][1] += 10.0 * (centers[i][1] + r - 1)
        
        # Update positions
        centers += lr_pos * grad_pos
        
        # Update radii: try to increase, but constrained by neighbors
        for i in range(n):
            # Max radius allowed by walls
            max_r_wall = min(centers[i][0], 1-centers[i][0], centers[i][1], 1-centers[i][1])
            if max_r_wall < 1e-6:
                max_r_wall = 1e-6
            
            # Max radius allowed by neighbors
            max_r_neighbor = max_r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-10:
                    max_r_neighbor = min(max_r_neighbor, dist - radii[j])
            
            max_r_neighbor = max(1e-6, max_r_neighbor)
            
            # Move radius toward max allowed
            target_r = max_r_neighbor * 0.99
            radii[i] += lr_rad * (target_r - radii[i])
            radii[i] = max(1e-6, min(radii[i], max_r_neighbor))
        
        # Clamp positions to valid range
        for i in range(n):
            r = radii[i]
            centers[i][0] = max(r, min(1-r, centers[i][0]))
            centers[i][1] = max(r, min(1-r, centers[i][1]))
        
        # Evaluate
        packing = np.column_stack([centers, radii])
        if _validate(packing):
            s = _score(packing)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
                best_radii = radii.copy()
        
        # Adaptive learning rate
        if it % 1000 == 0:
            lr_pos *= 0.8
            lr_rad *= 0.9
        
        # Check convergence
        if it > 1000 and abs(best_score - _score(packing)) < 1e-8:
            if it > 5000:
                break
    
    return best_centers, best_radii

def construct():
    deadline = time.time() + 120  # Safety margin before 150s limit
    
    best_packing = None
    best_score = 0.0
    
    # Multiple random initializations
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Random initial placement
        centers = np.random.uniform(0.1, 0.9, (N, 2))
        radii = np.full(N, 0.02)
        
        # Optimize
        c, r = _optimize_gradient(centers, radii, max_iter=10000, deadline=deadline - 5)
        
        packing = np.column_stack([c, r])
        if _validate(packing):
            s = _score(packing)
            if s > best_score:
                best_score = s
                best_packing = packing
    
    if best_packing is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_packing
# EVOLVE-BLOCK-END
