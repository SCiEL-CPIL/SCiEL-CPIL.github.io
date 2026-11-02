"""Rewritten program for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT (Separating Axis Theorem)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint
        if dist_sq > 4.0:
            return True
        
        # If dist < 1.7, definitely overlap (conservative)
        if dist_sq < 2.8: 
            return False
        
        # Precise SAT check
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        all_edges = []
        for i in range(6):
            all_edges.append(v1[i] - v1[(i+1) % 6])
            all_edges.append(v2[i] - v2[(i+1) % 6])
        
        for edge in all_edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            normal = normal / norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def smooth_objective(params, beta=1.0):
        """
        Smooth surrogate objective.
        Returns (soft_side + penalty, exact_side, valid).
        """
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Compute all vertices
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        sqrt3 = np.sqrt(3)
        
        # Soft containment
        normal_projections = []
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            extents = np.dot(offsets, normal)
            
            # Soft max via log-sum-exp
            max_ext = np.max(extents)
            if max_ext > 0:
                soft_max = max_ext + np.log(np.sum(np.exp((extents - max_ext) / beta))) * beta
            else:
                soft_max = max_ext
            normal_projections.append(soft_max)
        
        max_proj = np.max(normal_projections)
        if max_proj > 0:
            soft_side = max_proj + np.log(np.sum(np.exp((np.array(normal_projections) - max_proj) / beta))) * beta
            soft_side = soft_side * 2.0 / sqrt3
        else:
            soft_side = 2.0 / sqrt3
        
        # Soft overlap penalty
        overlap_penalty = 0.0
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                h1 = inner[i]
                h2 = inner[j]
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                
                if dist < 2.0:
                    # Quadratic penalty
                    overlap_penalty += (2.0 - dist) ** 2
                    # Check validity with SAT
                    if not hexagons_disjoint_fast(h1, h2):
                        valid = False
        
        return soft_side + 10.0 * overlap_penalty

    def objective_exact(params):
        """Exact objective: returns side length if valid, else large penalty."""
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return 1e6
        
        side = compute_min_side_fast(inner, center, angle)
        return side

    def initial_guess():
        params = np.zeros(3*N + 3)
        positions = []
        # Hexagonal packing: 1 center, 6 ring 1, 5 ring 2
        positions.append((0, 0, 0))
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    best_params = initial_guess()
    best_score = objective_exact(best_params)
    
    # Local search using Nelder-Mead on the smooth objective
    # We minimize the smooth objective, then check exact validity.
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb initial guess
        current_params = best_params.copy()
        if restart > 0:
            current_params += np.random.normal(0, 0.1, size=len(current_params))
        
        # Minimize smooth objective
        # Nelder-Mead is robust for this
        res = minimize(
            smooth_objective, 
            current_params, 
            method='Nelder-Mead', 
            options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000, 'maxfev': 2000}
        )
        
        if res.success or res.fun < best_score:
            candidate_params = res.x
            exact_score = objective_exact(candidate_params)
            if exact_score < best_score:
                best_score = exact_score
                best_params = candidate_params

    # Final polish on the best solution
    if time.time() < deadline - 5.0:
        res = minimize(
            smooth_objective, 
            best_params, 
            method='Nelder-Mead', 
            options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 5000, 'maxfev': 5000}
        )
        if res.fun < best_score:
            candidate_params = res.x
            exact_score = objective_exact(candidate_params)
            if exact_score < best_score:
                best_score = exact_score
                best_params = candidate_params

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to initial guess if local search broke validity
        inner = []
        for i in range(N):
            pos = initial_guess()
            inner.append([float(pos[3*i]), float(pos[3*i+1]), float(pos[3*i+2])])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side_fast(inner, center, angle)

    return inner, center, side, angle
