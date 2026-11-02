"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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
        
        # If dist > 2.0, definitely disjoint (max distance between any two points on unit hexagons)
        if dist_sq > 4.0:
            return True
        
        # Precise SAT check for all other cases
        a1 = math.radians(h1[2])
        a2 = math.radians(h2[2])
        
        # Precompute vertices using math for speed
        v1 = np.array([[h1[0] + math.cos(a1 + i*math.pi/3), h1[1] + math.sin(a1 + i*math.pi/3)] for i in range(6)])
        v2 = np.array([[h2[0] + math.cos(a2 + i*math.pi/3), h2[1] + math.sin(a2 + i*math.pi/3)] for i in range(6)])
        
        # Check all 12 edge normals (6 from each hexagon)
        for i in range(6):
            # Edge normal from h1
            e1 = v1[(i+1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            nn1 = math.sqrt(n1[0]**2 + n1[1]**2)
            if nn1 > 1e-12:
                n1 = n1 / nn1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                if proj1.max() < proj2.min() - 1e-9 or proj2.max() < proj1.min() - 1e-9:
                    return True
            
            # Edge normal from h2
            e2 = v2[(i+1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            nn2 = math.sqrt(n2[0]**2 + n2[1]**2)
            if nn2 > 1e-12:
                n2 = n2 / nn2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                if proj1.max() < proj2.min() - 1e-9 or proj2.max() < proj1.min() - 1e-9:
                    return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        sqrt3 = math.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = []
        for i in range(N):
            a = math.radians(inner[i][2])
            verts = [[inner[i][0] + math.cos(a + j*math.pi/3), inner[i][1] + math.sin(a + j*math.pi/3)] for j in range(6)]
            all_verts.extend(verts)
        all_verts = np.array(all_verts)
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        base_angle = math.radians(angle) + math.radians(30)
        for k in range(6):
            normal_angle = base_angle + k * math.pi / 3
            normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = offsets @ normal
            max_e = extents.max()
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def is_valid(inner):
        """Check if all hexagons are pairwise disjoint."""
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return False
        return True

    def objective(params):
        """Objective: minimize outer side length while ensuring disjointness."""
        # Unpack
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
        
        # Compute side
        side = compute_min_side_fast(inner, center, angle)
        
        return side

    def hexagonal_grid_init():
        """Create a honeycomb-like initialization that is valid and compact."""
        params = np.zeros(3*N + 3)
        
        # For flat-top hexagons of side 1:
        # Width = 2, Height = sqrt(3)
        # In a honeycomb pattern:
        # Horizontal spacing between centers in same row: 3 (they touch at vertices)
        # Wait, let me recalculate properly.
        # Flat-top hexagon vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
        # Two hexagons side by side: if A is at (0,0), its rightmost point is (1,0).
        # Hexagon B to the right touching at vertex: leftmost point of B is (1,0), so center of B is (2,0).
        # But in a honeycomb, hexagons are offset. Let me use a known valid configuration.
        
        # Use a 2x6 grid with honeycomb offset for better packing
        # Row 0: 6 hexagons at y=0, x = 0, 3, 6, 9, 12, 15 (spacing 3 for flat-top touching)
        # Wait, that's too wide. Let me think about the actual honeycomb geometry.
        
        # For flat-top hexagons of side s=1:
        # - Horizontal distance between centers in same row: 3s = 3 (touching at left/right vertices)
        #   Actually no. The rightmost vertex is at x=s=1. The next hexagon's leftmost is at x=-s=-1.
        #   If they touch, center distance = s - (-s) = 2s = 2.
        #   Wait, I'm confusing myself. Let me just use spacing > 2 to be safe.
        
        # Let's use a proper honeycomb with spacing slightly > 2.0
        # Flat-top hexagons in honeycomb:
        # Column spacing: 3/2 * 2 = 3? No.
        # Let me just use a simple valid grid with spacing 2.1
        
        spacing = 2.1
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * spacing
                y = row * spacing
                positions.append((x, y, 0))
        
        # Center the grid
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        for i in range(N):
            params[3*i] = positions[i][0] - cx
            params[3*i+1] = positions[i][1] - cy
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    def nelder_mead(params, f, max_iter=2000, tol=1e-8, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
        """Nelder-Mead optimization."""
        n = len(params)
        simplex = [params.copy() for _ in range(n + 1)]
        for i in range(n):
            simplex[i+1] = params.copy()
            simplex[i+1][i] += 0.1 * (1.0 if abs(params[i]) < 1e-10 else params[i])
        
        fvals = [f(s) for s in simplex]
        
        for _ in range(max_iter):
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            # Centroid of all but worst point
            centroid = np.zeros(n)
            for i in range(n):
                centroid += simplex[i]
            centroid /= n
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[n])
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[n])
                fe = f(xe)
                if fe < fr:
                    simplex[n] = xe
                    fvals[n] = fe
                else:
                    simplex[n] = xr
                    fvals[n] = fr
            elif fr < fvals[n-1]:
                simplex[n] = xr
                fvals[n] = fr
            else:
                # Contraction
                if fr < fvals[n]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    fc = f(xc)
                    if fc <= fr:
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = f(simplex[i])
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[n])
                    fc = f(xc)
                    if fc <= fvals[n]:
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = f(simplex[i])
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    # Start with a valid hexagonal grid initialization
    best_params = hexagonal_grid_init()
    best_score = objective(best_params)
    
    # Run Nelder-Mead optimization with more iterations
    if time.time() < deadline - 15.0:
        best_params, best_score = nelder_mead(best_params, objective, max_iter=8000)
    
    # Polish with simulated annealing - longer run
    if time.time() < deadline - 5.0:
        current_params = best_params.copy()
        current_score = best_score
        
        T = 0.3
        T_min = 0.0001
        cooling_rate = 0.9999
        step_size = 0.05
        
        polish_start = time.time()
        while time.time() - polish_start < 12.0:
            new_params = current_params.copy()
            idx = random.randint(0, 3*N + 2)
            new_params[idx] += random.gauss(0, step_size * (0.5 + 0.5 * T))
            
            new_score = objective(new_params)
            
            if new_score < current_score or (T > T_min and random.random() < math.exp(-min((new_score - current_score) / T, 500))):
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
    
    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final check: ensure validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback: use the hexagonal grid
        inner = []
        spacing = 2.1
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * spacing
                y = row * spacing
                positions.append((x, y, 0))
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        for pos in positions:
            inner.append([float(pos[0] - cx), float(pos[1] - cy), float(pos[2])])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side_fast(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
