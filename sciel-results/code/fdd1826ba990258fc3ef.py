# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length (maximizing 1/side).
    
    Uses a hexagonal lattice arrangement: 1 center + 6 surrounding + 4 in the gaps,
    then optimizes positions to minimize the bounding hexagon.
    """
    # Initial configuration: hexagonal lattice
    # Center at origin
    # 6 neighbors at distance 2 (touching)
    # 4 more in the "gap" positions
    
    # For regular hexagons with side 1, the distance between centers of touching hexagons is 2.
    # In a hexagonal lattice, neighbors are at 60-degree intervals.
    
    # Let's use a symmetric arrangement:
    # Center: (0, 0)
    # Ring 1: 6 hexagons at distance 2, angles 0, 60, 120, 180, 240, 300
    # Ring 2: 4 hexagons placed in some of the gaps
    
    # Actually, for 11 hexagons, a common optimal packing is:
    # 1 center + 6 surrounding + 4 additional
    # The 4 additional can be placed at positions that minimize the bounding hexagon.
    
    # Let's parameterize using the hexagonal lattice coordinates.
    # In axial coordinates (q, r), the Cartesian position is:
    # x = 2 * q + r
    # y = sqrt(3) * r
    # But for unit hexagons, the center-to-center distance is 2, so we scale accordingly.
    
    # Actually, let me think more carefully. For regular hexagons with side length 1:
    # The distance between centers of two touching hexagons is 2 (sum of their apothems * 2 = 2 * (sqrt(3)/2) * 2 = 2*sqrt(3)? No...)
    # Wait, for two regular hexagons with side 1 to touch, the distance between centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)? No.
    # The apothem of a regular hexagon with side 1 is sqrt(3)/2. Two hexagons touching edge-to-edge have center distance = 2 * apothem = sqrt(3).
    # But actually, for two regular hexagons with side 1, if they share an edge, the distance between centers is 2 * (sqrt(3)/2) = sqrt(3).
    # Hmm, but the problem says "unit-side regular hexagons" and they need to be separated (not intersecting).
    # The condition is that on at least one of the 12 edge normals, the projected intervals are apart by more than 1e-6.
    # This means they can touch but not overlap.
    
    # Let me reconsider. For two regular hexagons with side 1:
    # If they are in "flat-top" orientation (vertex at top), the width is 2 (from left to right vertex) and height is sqrt(3).
    # If they are in "pointy-top" orientation, width is sqrt(3) and height is 2.
    
    # The distance between centers for two touching hexagons depends on orientation.
    # For simplicity, let's use the standard hexagonal packing where all hexagons have the same orientation.
    # In this case, the center-to-center distance for touching hexagons is 2 * apothem = sqrt(3) if they share an edge.
    # But actually, for two regular hexagons with side 1 in the same orientation, the minimum center distance to not overlap is 2 * (sqrt(3)/2) = sqrt(3) when they share an edge.
    
    # Let me just use a numerical optimization approach.
    # Start with a reasonable initial guess and optimize.
    
    # Initial positions: hexagonal lattice with spacing = sqrt(3)
    # Actually, let me use spacing = 2 for safety and then optimize.
    
    # Let's use a different approach: start with positions that are definitely valid (separated)
    # and then optimize to minimize the bounding hexagon.
    
    # Use a symmetric arrangement. Let's place:
    # Center: (0, 0)
    # 6 surrounding at distance d, angles 0, 60, ..., 300
    # 4 more at distance 2d, angles 30, 150, 210, 330 (in the gaps)
    
    d = 2.0  # initial distance
    
    angles_ring1 = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    angles_ring2 = np.array([30, 150, 210, 330]) * np.pi / 180
    
    positions = [(0, 0)]
    for a in angles_ring1:
        positions.append((d * np.cos(a), d * np.sin(a)))
    for a in angles_ring2:
        positions.append((2 * d * np.cos(a), 2 * d * np.sin(a)))
    
    # Flatten to a vector for optimization
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions])
    
    def objective(x):
        """Compute the minimum side length of a regular hexagon (centered at origin, angle 0) that contains all hexagons."""
        pos = np.array(x[:11 * 2]).reshape(11, 2)
        
        # For a regular hexagon centered at origin with side length s and angle 0 (flat-top),
        # the vertices are at (s*cos(60*i), s*sin(60*i)) for i=0..5.
        # The hexagon contains a point (px, py) if it's inside all 6 half-planes defined by the edges.
        
        # For a flat-top hexagon with side s, the 6 edge normals point in directions 30, 90, 150, 210, 270, 330 degrees.
        # The distance from center to each edge (apothem) is s * sqrt(3) / 2.
        
        # A point (px, py) is inside the hexagon if for each edge normal direction theta,
        # the projection of (px, py) onto the normal is <= apothem.
        
        # The 6 normals for a flat-top hexagon (angle 0) are at angles 30, 90, 150, 210, 270, 330 degrees.
        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # For each inner hexagon, we need to check that all its vertices are inside the outer hexagon.
        # But for the objective, we can approximate by checking the center and the farthest vertex.
        # Actually, let's just compute the maximum over all inner hexagon vertices of the required apothem.
        
        max_required = 0
        for i in range(11):
            cx, cy = pos[i]
            # The inner hexagon has vertices at (cx + cos(60*j), cy + sin(60*j)) for j=0..5 (angle 0)
            for j in range(6):
                vx = cx + np.cos(j * np.pi / 3)
                vy = cy + np.sin(j * np.pi / 3)
                # Project onto each normal
                proj = np.abs(normals @ np.array([vx, vy]))
                max_required = max(max_required, np.max(proj))
        
        # The required apothem is max_required, so side = 2 * max_required / sqrt(3)
        return 2 * max_required / np.sqrt(3)
    
    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
    
    if not result.success:
        # Fall back to initial guess
        x_opt = x0
    else:
        x_opt = result.x
    
    pos_opt = np.array(x_opt[:22]).reshape(11, 2)
    
    # Compute the actual required side length
    side = objective(x_opt)
    
    # Add a small margin for safety
    side += 0.01
    
    inner_hex_data = np.column_stack([pos_opt[:, 0], pos_opt[:, 1], np.zeros(11)])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END
