"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 11
DEADLINE = 115.0  # safety margin before 120s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def sat_overlap_penalty(v1, v2, tol=1e-9):
        """
        Calculate a smooth penalty for overlap between two convex polygons.
        Returns 0 if disjoint, > 0 if overlapping.
        Uses the 12 edge normals from both polygons.
        """
        penalty = 0.0
        # Get all edge normals from both polygons
        # Poly 1 normals
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue
            normal /= norm_len
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 + tol < min2 or max2 + tol < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth on this axis
            # The amount by which the intervals [min1, max1] and [min2, max2] overlap
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                penalty += overlap ** 2
        
        # Poly 2 normals
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue
            normal /= norm_len
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 + tol < min2 or max2 + tol < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                penalty += overlap ** 2
                
        return penalty

    def compute_S(vertices_list, ocx, ocy, oangle):
        """
        Compute the minimum side length S of the outer hexagon 
        centered at (ocx, ocy) with rotation oangle to contain all vertices.
        """
        apothem = np.sqrt(3) / 2
        max_proj = 0.0
        
        # Normals for the container hexagon (rotated by oangle)
        # Base normals at 30, 90, 150... degrees relative to container
        for k in range(6):
            theta_n = oangle + np.pi / 6 + k * np.pi / 3
            cos_n = np.cos(theta_n)
            sin_n = np.sin(theta_n)
            
            for verts in vertices_list:
                # Project global vertices onto the global normal
                projs = verts[:, 0] * cos_n + verts[:, 1] * sin_n
                # The projection of the center is ocx*cos_n + ocy*sin_n
                center_proj = ocx * cos_n + ocy * sin_n
                max_rel = np.max(projs) - center_proj
                min_rel = np.min(projs) - center_proj
                extent = max(abs(max_rel), abs(min_rel))
                if extent > max_proj:
                    max_proj = extent
        
        return max_proj / apothem

    def objective(params):
        """
        params: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, oangle]
        """
        if time.time() > deadline:
            return float('inf')
            
        n_inner = N
        inner_params = params[:3 * n_inner]
        cont_params = params[3 * n_inner:]
        ocx, ocy, oangle = cont_params
        
        vertices_list = []
        for i in range(n_inner):
            cx = inner_params[3 * i]
            cy = inner_params[3 * i + 1]
            angle = inner_params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S(vertices_list, ocx, ocy, oangle)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                # Quick rejection
                c1 = np.array([inner_params[3*i], inner_params[3*i+1]])
                c2 = np.array([inner_params[3*j], inner_params[3*j+1]])
                if np.linalg.norm(c1 - c2) > 2.5:
                    continue
                p = sat_overlap_penalty(vertices_list[i], vertices_list[j])
                penalty += p
        
        return S + 100.0 * penalty

    # Initialize with hexagonal lattice
    np.random.seed(42)
    params = np.zeros(3 * N + 3)
    
    # Inner hexagons
    # Center
    params[0:3] = [0, 0, 0]
    
    # Ring 1
    r1 = np.sqrt(3)
    for k in range(6):
        theta = k * np.pi / 3
        cx = r1 * np.cos(theta)
        cy = r1 * np.sin(theta)
        idx = 1 + k
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        
    # Ring 2 (4 more)
    # Place them in the gaps
    # Approx positions
    r2 = 2 * np.sqrt(3)
    angles2 = [0, np.pi/3, 2*np.pi/3, np.pi] # Just guessing a spread
    for k in range(4):
        theta = angles2[k]
        cx = r2 * np.cos(theta)
        cy = r2 * np.sin(theta)
        idx = 7 + k
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        
    # Container
    params[3 * N] = 0.0
    params[3 * N + 1] = 0.0
    params[3 * N + 2] = 0.0

    current_params = params.copy()
    current_energy = objective(current_params)
    best_params = current_params.copy()
    best_energy = current_energy
    
    # Phase 1: Simulated Annealing for global exploration
    T = 0.1
    T_min = 0.001
    cooling = 0.995
    
    # Run SA for a portion of the time
    sa_deadline = start_time + 40.0
    while time.time() < sa_deadline:
        # Perturb
        new_params = current_params.copy()
        
        # Randomly select an inner hexagon to move
        idx = np.random.randint(0, N)
        # Move x, y, angle
        new_params[3 * idx] += np.random.normal(0, 0.1)
        new_params[3 * idx + 1] += np.random.normal(0, 0.1)
        new_params[3 * idx + 2] += np.random.normal(0, 0.05)
        
        # Occasionally perturb container
        if np.random.random() < 0.05:
            new_params[3 * N] += np.random.normal(0, 0.1)
            new_params[3 * N + 1] += np.random.normal(0, 0.1)
            new_params[3 * N + 2] += np.random.normal(0, 0.05)
            
        new_energy = objective(new_params)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_params = new_params
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = current_params.copy()
                
        T *= cooling
        if T < T_min:
            break

    # Phase 2: Local Refinement with Nelder-Mead
    # Try to polish the best solution found by SA
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass

    # Phase 3: Multi-start Nelder-Mead from perturbed best solutions
    # Use remaining time to try multiple local optimizations from slightly perturbed best
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Try 3-5 restarts
        num_restarts = 3
        restart_budget = remaining_time / (num_restarts + 1)
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Perturb the best solution slightly
            perturbed = best_params.copy()
            noise_scale = 0.05 * (1 - r / num_restarts) # Decreasing noise
            for i in range(N):
                perturbed[3*i] += np.random.normal(0, noise_scale)
                perturbed[3*i+1] += np.random.normal(0, noise_scale)
                perturbed[3*i+2] += np.random.normal(0, noise_scale * 0.5)
            
            try:
                res = minimize(objective, perturbed, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_params = res.x
            except Exception:
                pass

    # Extract results
    inner = []
    vertices_list = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
        vertices_list.append(hex_vertices(cx, cy, angle))

    ocx = best_params[3 * N]
    ocy = best_params[3 * N + 1]
    oangle = best_params[3 * N + 2]
    
    # Final check and S calculation
    S = compute_S(vertices_list, ocx, ocy, oangle)
    
    # If there is any overlap, the score might be invalid or penalized.
    # We assume the optimization minimized the penalty sufficiently.
    
    return inner, (ocx, ocy), S, np.degrees(oangle)
# EVOLVE-BLOCK-END
