"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    r = np.sqrt(3) / 2  # inradius of unit hexagon
    d = np.sqrt(3)      # center-to-center distance for edge-to-edge hexagons
    
    # Generate hex lattice positions
    positions = []
    for q in range(-5, 6):
        for rr in range(-5, 6):
            x = d * (q + rr / 2.0)
            y = d * (np.sqrt(3) / 2.0 * rr)
            positions.append((x, y))
    
    # Candidate centers for outer hexagon
    center_candidates = set()
    for cx in np.linspace(-4, 4, 17):
        for cy in np.linspace(-4, 4, 17):
            center_candidates.add((round(cx, 6), round(cy, 6)))
    
    # Also add lattice points and midpoints
    for q in range(-4, 5):
        for rr in range(-4, 5):
            x = d * (q + rr / 2.0)
            y = d * (np.sqrt(3) / 2.0 * rr)
            center_candidates.add((round(x, 6), round(y, 6)))
            center_candidates.add((round(x + d/2, 6), round(y + d*np.sqrt(3)/4, 6)))
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_rot = 0.0
    
    for outer_cx, outer_cy in center_candidates:
        # Find 12 nearest lattice points
        dists = sorted([(np.hypot(px - outer_cx, py - outer_cy), px, py) for px, py in positions])
        selected = [(px, py) for _, px, py in dists[:12]]
        
        # Try outer hexagon rotations from 0 to 30 degrees
        for outer_rot in np.linspace(0, 30, 7):
            angles = (np.array([0, 60, 120, 180, 240, 300]) + outer_rot) * np.pi / 180
            dirs = np.column_stack([np.cos(angles), np.sin(angles)])
            
            max_inrad = 0
            for px, py in selected:
                dx, dy = px - outer_cx, py - outer_cy
                projs = dx * dirs[:, 0] + dy * dirs[:, 1]
                max_inrad = max(max_inrad, np.max(projs) + r)
            
            s = max_inrad * 2 / np.sqrt(3)
            if s < best_side:
                best_side = s
                best_inner = [[px, py, 0.0] for px, py in selected]
                best_center = (outer_cx, outer_cy)
                best_rot = outer_rot
    
    if best_inner is None:
        best_inner = [[d * k, 0.0, 0.0] for k in range(-5, 7)]
        best_center = (0.0, 0.0)
        best_side = 40.0
        best_rot = 0.0
    
    return best_inner, best_center, best_side, best_rot
# EVOLVE-BLOCK-END
