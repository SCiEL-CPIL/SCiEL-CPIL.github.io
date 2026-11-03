"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Use a hexagonal lattice arrangement for compactness
    # Axial coordinates for a compact 12-hexagon cluster
    # Try a 3-4-3-2 row pattern
    axial_coords = []
    # Row 0: 3 hexagons
    for i in range(3):
        axial_coords.append((i - 1, 0))
    # Row 1: 4 hexagons  
    for i in range(4):
        axial_coords.append((i - 1.5, 1))
    # Row 2: 3 hexagons
    for i in range(3):
        axial_coords.append((i - 1, 2))
    # Row 3: 2 hexagons
    for i in range(2):
        axial_coords.append((i - 0.5, 3))
    
    # Convert axial to Cartesian (for flat-top hexagons)
    # x = sqrt(3) * (q + r/2), y = 3/2 * r
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the arrangement
    center_offset = np.mean(centers, axis=0)
    centers = centers - center_offset
    
    # Compute the minimal enclosing regular hexagon
    # For a regular hexagon with side length S and rotation angle theta,
    # the constraint is that all points lie within the hexagon
    
    # The half-width in each of 6 directions must be <= S * sqrt(3) / 2
    # Direction vectors for a regular hexagon at rotation theta:
    # angle_k = theta + k * pi/3 for k=0..5
    
    def hexagon_side_length(theta, points):
        """Compute minimum side length of regular hexagon with rotation theta enclosing all points."""
        min_side = 0.0
        for k in range(6):
            angle = theta + k * np.pi / 3
            dx = np.cos(angle)
            dy = np.sin(angle)
            # Project all points onto this direction
            projections = points @ np.array([dx, dy])
            # The hexagon extends from -S*sqrt(3)/2 to S*sqrt(3)/2 in this direction
            # So max projection - min projection <= S * sqrt(3)
            # But we need max projection <= S*sqrt(3)/2 and min projection >= -S*sqrt(3)/2
            # So S >= max(|max_proj|, |min_proj|) * 2 / sqrt(3)
            max_proj = np.max(projections)
            min_proj = np.min(projections)
            needed = 2.0 / np.sqrt(3) * max(abs(max_proj), abs(min_proj))
            min_side = max(min_side, needed)
        return min_side
    
    # Try different rotation angles
    best_side = float('inf')
    best_theta = 0.0
    for theta in np.linspace(0, np.pi / 3, 61):
        side = hexagon_side_length(theta, centers)
        if side < best_side:
            best_side = side
            best_theta = theta
    
    # Create inner hexagons (all axis-aligned, angle 0)
    inner = []
    for cx, cy in centers:
        inner.append([float(cx), float(cy), 0.0])
    
    outer_center = [0.0, 0.0]
    outer_side = float(best_side)
    outer_angle = float(best_theta * 180 / np.pi)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
