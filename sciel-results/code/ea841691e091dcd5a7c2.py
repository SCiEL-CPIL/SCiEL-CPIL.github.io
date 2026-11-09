# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60.0) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_gap_hex(hex1, hex2):
    """Compute minimum separation between two unit hexagons.
    Returns the minimum distance between any edge of hex1 and any edge of hex2,
    or a negative value if they overlap.
    """
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    # Get all 12 edge normals (6 from each hexagon)
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            normals.append(n)
    
    min_gap = np.inf
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        gap_forward = proj2.min() - proj1.max()
        gap_backward = proj1.min() - proj2.max()
        sep = max(gap_forward, gap_backward)
        if sep < min_gap:
            min_gap = sep
    
    return min_gap

def _required_side(inner_hexes):
    """Calculate the minimum side length of a centered, axis-aligned hexagon to contain all inner hexes."""
    max_r = 0.0
    for i in range(len(inner_hexes)):
        cx, cy, a_deg = inner_hexes[i]
        verts = _hex_vertices(cx, cy, 1.0, a_deg)
        for v in verts:
            x, y = v
            # Distance from center to boundary of a regular hexagon with side s, centered at origin, angle 0
            # The boundary is defined by |x| <= s, |y| <= s*sqrt(3)/2? No.
            # Vertices at (s, 0), (s/2, s*sqrt(3)/2), etc.
            # The lines are:
            # x = s
            # x = -s
            # sqrt(3)*y + x = s*sqrt(3)  -> y + x/sqrt(3) = s
            # sqrt(3)*y - x = s*sqrt(3)  -> y - x/sqrt(3) = s
            # y = -s*sqrt(3)/2 ? No.
            # Let's check the vertices of the outer hexagon with side S, center (0,0), angle 0.
            # V0: (S, 0)
            # V1: (S/2, S*sqrt(3)/2)
            # V2: (-S/2, S*sqrt(3)/2)
            # V3: (-S, 0)
            # V4: (-S/2, -S*sqrt(3)/2)
            # V5: (S/2, -S*sqrt(3)/2)
            
            # Edge 0-1: Line from (S,0) to (S/2, S*sqrt(3)/2).
            # Slope: (S*sqrt(3)/2 - 0) / (S/2 - S) = (S*sqrt(3)/2) / (-S/2) = -sqrt(3).
            # Eq: y - 0 = -sqrt(3)(x - S) => y = -sqrt(3)x + S*sqrt(3) => sqrt(3)x + y = S*sqrt(3).
            # Constraint: sqrt(3)x + y <= S*sqrt(3) => (sqrt(3)x + y)/sqrt(3) <= S => x + y/sqrt(3) <= S.
            
            # Edge 1-2: Line from (S/2, S*sqrt(3)/2) to (-S/2, S*sqrt(3)/2).
            # Eq: y = S*sqrt(3)/2.
            # Constraint: y <= S*sqrt(3)/2 => y / (sqrt(3)/2) <= S => 2y/sqrt(3) <= S.
            
            # Edge 2-3: Line from (-S/2, S*sqrt(3)/2) to (-S, 0).
            # Slope: (0 - S*sqrt(3)/2) / (-S - (-S/2)) = (-S*sqrt(3)/2) / (-S/2) = sqrt(3).
            # Eq: y - 0 = sqrt(3)(x + S) => y = sqrt(3)x + S*sqrt(3) => -sqrt(3)x + y = S*sqrt(3).
            # Constraint: -sqrt(3)x + y <= S*sqrt(3) => -x + y/sqrt(3) <= S.
            
            # By symmetry, the constraints are:
            # 1. x + y/sqrt(3) <= S
            # 2. -x + y/sqrt(3) <= S
            # 3. -x - y/sqrt(3) <= S
            # 4. x - y/sqrt(3) <= S
            # 5. y <= S*sqrt(3)/2  => 2y/sqrt(3) <= S
            # 6. -y <= S*sqrt(3)/2 => -2y/sqrt(3) <= S
            
            # So S >= max( 
            #   |x| + |y|/sqrt(3),  # Combining 1,2,3,4? 
            #   Let's check:
            #   If x>0, y>0: Max is x + y/sqrt(3) (from 1) or 2y/sqrt(3) (from 5).
            #   If x>0, y<0: Max is x - y/sqrt(3) (from 4) or -2y/sqrt(3) (from 6).
            #   Basically, S = max( |x| + |y|/sqrt(3), 2|y|/sqrt(3) )?
            #   Let's test a vertex (S, 0). S = max(S + 0, 0) = S. Correct.
            #   Let's test a vertex (S/2, S*sqrt(3)/2).
            #   |x| + |y|/sqrt(3) = S/2 + (S*sqrt(3)/2)/sqrt(3) = S/2 + S/2 = S.
            #   2|y|/sqrt(3) = 2*(S*sqrt(3)/2)/sqrt(3) = S.
            #   So S = max( |x| + |y|/sqrt(3), 2|y|/sqrt(3) ) works?
            #   Wait, what about x component?
            #   The constraints are linear.
            #   S >= x + y/sqrt(3)
            #   S >= -x + y/sqrt(3)
            #   S >= -x - y/sqrt(3)
            #   S >= x - y/sqrt(3)
            #   S >= 2y/sqrt(3)
            #   S >= -2y/sqrt(3)
            
            #   The first four imply S >= |x| + |y|/sqrt(3).
            #   The last two imply S >= 2|y|/sqrt(3).
            
            #   So required S for point (x,y) is max( |x| + |y|/sqrt(3), 2|y|/sqrt(3) ).
            
            r1 = abs(x) + abs(y) / np.sqrt(3)
            r2 = 2 * abs(y) / np.sqrt(3)
            r = max(r1, r2)
            if r > max_r:
                max_r = r
    return max_r

def _objective(params):
    """
    params: 11*2 (x,y) + 1 (side) is not needed if we calculate required side.
    Let's just optimize x,y.
    params: 22 floats.
    """
    inner_hexes = np.zeros((11, 3))
    inner_hexes[:, 0] = params[0::2]
    inner_hexes[:, 1] = params[1::2]
    inner_hexes[:, 2] = 0.0 # Fixed angle
    
    side = _required_side(inner_hexes)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            gap = _min_gap_hex(inner_hexes[i], inner_hexes[j])
            if gap < 1e-6:
                penalty += (1e-6 - gap) ** 2 * 1000
    
    return side + penalty

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    sqrt3 = np.sqrt(3)
    
    # Initial guess: 1 center, 6 ring, 4 extra
    # Center
    xs = [0.0]
    ys = [0.0]
    
    # Ring 1
    xs.append(sqrt3)
    ys.append(0.0)
    xs.append(sqrt3/2)
    ys.append(1.5)
    xs.append(-sqrt3/2)
    ys.append(1.5)
    xs.append(-sqrt3)
    ys.append(0.0)
    xs.append(-sqrt3/2)
    ys.append(-1.5)
    xs.append(sqrt3/2)
    ys.append(-1.5)
    
    # Ring 2 (Pick 4)
    # Let's pick the ones that are "corners" of the hexagonal lattice
    # (2*sqrt3, 0), (sqrt3, 3), (-sqrt3, 3), (-2*sqrt3, 0)
    xs.append(2*sqrt3)
    ys.append(0.0)
    xs.append(sqrt3)
    ys.append(3.0)
    xs.append(-sqrt3)
    ys.append(3.0)
    xs.append(-2*sqrt3)
    ys.append(0.0)
    
    params_init = np.array(xs + ys)
    
    # Optimize
    best_params = params_init.copy()
    best_val = _objective(params_init)
    
    # Use Powell method
    try:
        result = minimize(_objective, params_init, method='Powell', 
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except:
        pass
    
    # Try Nelder-Mead for robustness
    try:
        result2 = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result2.fun < best_val:
            best_val = result2.fun
            best_params = result2.x
    except:
        pass
    
    # Extract results
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = best_params[0::2]
    inner_hex_data[:, 1] = best_params[1::2]
    inner_hex_data[:, 2] = 0.0
    
    outer_side = _required_side(inner_hex_data)
    
    # Add small buffer to ensure validity
    outer_side += 1e-5
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(outer_side)

# EVOLVE-BLOCK-END
