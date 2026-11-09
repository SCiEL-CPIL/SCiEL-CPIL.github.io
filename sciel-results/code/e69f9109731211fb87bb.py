import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    # Initial configuration: 7-hex flower + 4 corners
    # Centers in Cartesian
    # 7-hex flower
    centers = [(0, 0)]
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
    
    # 4 corners at 30, 90, 150, 210 degrees (distance 3 from center)
    # Let's add them at 30, 90, 150, 270 to be somewhat symmetric?
    # Or just pick 4.
    corner_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 3*np.pi/2]
    for angle in corner_angles:
        centers.append((3 * np.cos(angle), 3 * np.sin(angle)))
        
    # Flatten to vector for optimizer
    # Variables: x1, y1, ..., x11, y11, phi
    x0 = []
    for c in centers:
        x0.append(c[0])
        x0.append(c[1])
    x0.append(0.0) # phi
    
    def objective(vars):
        xs = vars[0::2]
        ys = vars[1::2]
        phi = vars[-1]
        
        # Calculate vertices for each hexagon
        # Hexagon vertices relative to center, side 1
        # Angles: 0, 60, 120, 180, 240, 300
        # Distance 1
        hex_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
        rel_x = np.cos(hex_angles)
        rel_y = np.sin(hex_angles)
        
        max_val = 0
        for i in range(N):
            cx = xs[i]
            cy = ys[i]
            # Vertices
            vx = cx + rel_x
            vy = cy + rel_y
            
            # Check constraint for all 6 normals of outer hexagon
            # Normals at phi + k*pi/3
            for k in range(6):
                theta = phi + k * np.pi / 3
                nx = np.cos(theta)
                ny = np.sin(theta)
                # Dot product
                dots = vx * nx + vy * ny
                # We want max abs dot product
                m = np.max(np.abs(dots))
                if m > max_val:
                    max_val = m
        # S = max_val * 2 / sqrt(3)
        return max_val * 2 / np.sqrt(3)
        
    # Optimize
    res = minimize(objective, x0, method='Powell', options={'maxiter': 200})
    
    # Extract best positions
    best_vars = res.x
    xs = best_vars[0::2]
    ys = best_vars[1::2]
    phi = best_vars[-1]
    
    # Check for overlaps? 
    # The objective doesn't penalize overlaps.
    # We need to ensure they are disjoint.
    # If they overlap, the result is invalid.
    # We should add a penalty for overlaps.
    
    # Let's refine the objective to include overlap penalty.
    # Distance between centers must be >= sqrt(3) (for edge-to-edge).
    # Actually, min distance between any two points of two hexagons.
    # Approximation: Distance between centers >= sqrt(3) is sufficient if orientations are similar.
    # With rotations, it's harder.
    # Let's assume no rotation for inner hexagons to simplify overlap check.
    # If inner hexagons are all aligned, min center distance is sqrt(3).
    
    # Let's re-run with overlap penalty.
    
    def objective_with_penalty(vars):
        xs = vars[0::2]
        ys = vars[1::2]
        phi = vars[-1]
        
        # Boundary penalty
        hex_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
        rel_x = np.cos(hex_angles)
        rel_y = np.sin(hex_angles)
        
        max_val = 0
        for i in range(N):
            cx = xs[i]
            cy = ys[i]
            vx = cx + rel_x
            vy = cy + rel_y
            for k in range(6):
                theta = phi + k * np.pi / 3
                nx = np.cos(theta)
                ny = np.sin(theta)
                dots = vx * nx + vy * ny
                m = np.max(np.abs(dots))
                if m > max_val:
                    max_val = m
        S = max_val * 2 / np.sqrt(3)
        
        # Overlap penalty
        overlap_pen = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < np.sqrt(3):
                    overlap_pen += (np.sqrt(3) - dist)**2
                    
        return S + 100 * overlap_pen
        
    res2 = minimize(objective_with_penalty, x0, method='Powell', options={'maxiter': 500})
    
    best_vars = res2.x
    xs = best_vars[0::2]
    ys = best_vars[1::2]
    phi = best_vars[-1]
    
    # Calculate final S
    hex_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    rel_x = np.cos(hex_angles)
    rel_y = np.sin(hex_angles)
    max_val = 0
    for i in range(N):
        cx = xs[i]
        cy = ys[i]
        vx = cx + rel_x
        vy = cy + rel_y
        for k in range(6):
            theta = phi + k * np.pi / 3
            nx = np.cos(theta)
            ny = np.sin(theta)
            dots = vx * nx + vy * ny
            m = np.max(np.abs(dots))
            if m > max_val:
                max_val = m
    S = max_val * 2 / np.sqrt(3)
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([xs[i], ys[i], 0.0]) # Angle 0 for all inner hexagons
        
    return inner, (0.0, 0.0), S, phi

