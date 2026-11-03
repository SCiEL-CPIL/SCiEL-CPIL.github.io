"""Improved program for packing 11 unit hexagons in a regular hexagon."""
import math
import time
import numpy as np

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Regular hexagon with side length 1, flat-top orientation
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_vertices = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])

    def hex_support_function(angle_deg):
        """Support function of unit regular hexagon in direction angle_deg."""
        a = math.radians(angle_deg)
        ux, uy = math.cos(a), math.sin(a)
        projections = hex_vertices @ np.array([ux, uy])
        return np.max(projections)

    def hex_radius(angle_deg):
        """Distance from center to boundary in direction angle_deg."""
        return hex_support_function(angle_deg)

    def check_packing(inner_positions, outer_center, outer_side, outer_angle_deg):
        """Check if all inner hexagons fit inside the outer hexagon and don't overlap."""
        # Check containment: for each inner hex, check all 6 directions
        for ix, iy, iangle in inner_positions:
            # For containment, we need: for all directions phi,
            # |inner_center - outer_center| * cos(phi - direction_to_center) + hex_support(inner, phi) <= outer_support(outer, phi)
            # Simplified: check 6 face normals of outer hexagon
            for k in range(6):
                phi = (outer_angle_deg + 30 + 60 * k) % 360  # face normals
                phi_rad = math.radians(phi)
                ux, uy = math.cos(phi_rad), math.sin(phi_rad)
                
                # Distance from outer center to inner center projected on normal
                dx = ix - outer_center[0]
                dy = iy - outer_center[1]
                proj = dx * ux + dy * uy
                
                # Inner hex support in this direction
                inner_support = hex_support_function(iangle + phi)
                
                # Outer hex support in this direction
                outer_support = outer_side * hex_support_function(outer_angle_deg + phi)
                
                if proj + inner_support > outer_support + 1e-9:
                    return False
        
        # Check non-overlap between inner hexagons
        for i in range(len(inner_positions)):
            for j in range(i + 1, len(inner_positions)):
                x1, y1, a1 = inner_positions[i]
                x2, y2, a2 = inner_positions[j]
                dx = x2 - x1
                dy = y2 - y1
                dist = math.sqrt(dx * dx + dy * dy)
                
                # Minimum distance between two unit hexagons depends on relative angle
                # Conservative check: if dist < 2*sqrt(3)/2 = sqrt(3) ~ 1.732, they might overlap
                # More accurate: check if hexagons intersect
                # For simplicity, use: two hexagons of side 1 don't overlap if center distance > sqrt(3)
                # Actually, the minimum distance is sqrt(3) when aligned
                if dist < math.sqrt(3) - 1e-9:
                    # More careful check needed
                    # Check if any vertex of one is inside the other
                    for va in range(6):
                        vangle = a1 + 60 * va
                        vx = x1 + math.cos(math.radians(vangle))
                        vy = y1 + math.sin(math.radians(vangle))
                        # Check if (vx, vy) is inside hexagon centered at (x2, y2) with angle a2
                        if _point_in_hex(vx, vy, x2, y2, a2):
                            return False
                    for va in range(6):
                        vangle = a2 + 60 * va
                        vx = x2 + math.cos(math.radians(vangle))
                        vy = y2 + math.sin(math.radians(vangle))
                        if _point_in_hex(vx, vy, x1, y1, a1):
                            return False
        
        return True

    def _point_in_hex(px, py, hx, hy, hangle_deg):
        """Check if point (px,py) is inside hexagon centered at (hx,hy) with rotation hangle_deg."""
        dx = px - hx
        dy = py - hy
        # Rotate point by -hangle to align hex with standard orientation
        a = math.radians(-hangle_deg)
        rx = dx * math.cos(a) - dy * math.sin(a)
        ry = dx * math.sin(a) + dy * math.cos(a)
        
        # Check against 6 half-planes of standard hexagon
        for k in range(6):
            angle = math.radians(30 + 60 * k)
            ux = math.cos(angle)
            uy = math.sin(angle)
            if rx * ux + ry * uy > math.sqrt(3) / 2 + 1e-9:
                return False
        return True

    # Initial guess: hexagonal cluster of 11 hexagons
    # One center, 6 around it, 4 more in second ring
    # For unit hexagons, adjacent centers are sqrt(3) apart
    
    # Hexagonal lattice: centers at distance sqrt(3)
    d = math.sqrt(3)
    
    # 7 hexagons in center cluster (1 + 6)
    positions = []
    # Center
    positions.append([0.0, 0.0, 0.0])
    # 6 around center
    for k in range(6):
        angle = math.radians(60 * k)
        positions.append([d * math.cos(angle), d * math.sin(angle), 0.0])
    
    # Add 4 more in second ring
    # Second ring positions at distance 2*d from center, but only some fit
    for k in range(6):
        angle = math.radians(60 * k + 30)
        positions.append([2 * d * math.cos(angle), 2 * d * math.sin(angle), 0.0])
        if len(positions) >= N:
            break
    
    # Try to find a good outer hexagon
    best_side = float('inf')
    best_result = None
    
    # Try different outer angles
    for outer_angle in [0, 30, 15, -15, 45, -45, 60, 120]:
        # Estimate required side length
        # For each direction, compute required outer side
        max_required = 0
        
        for ix, iy, iangle in positions:
            for k in range(6):
                phi = (outer_angle + 30 + 60 * k) % 360
                phi_rad = math.radians(phi)
                ux, uy = math.cos(phi_rad), math.sin(phi_rad)
                
                dx = ix
                dy = iy
                proj = dx * ux + dy * uy
                
                inner_support = hex_support_function(iangle + phi)
                
                # Required: proj + inner_support <= S * hex_support(outer_angle + phi)
                outer_support_unit = hex_support_function(outer_angle + phi)
                if outer_support_unit > 1e-9:
                    required = (proj + inner_support) / outer_support_unit
                    max_required = max(max_required, required)
        
        candidate_side = max_required + 0.01  # small margin
        
        if candidate_side < best_side:
            # Verify packing
            if check_packing(positions, (0.0, 0.0), candidate_side, outer_angle):
                best_side = candidate_side
                best_result = (positions, (0.0, 0.0), candidate_side, outer_angle)
    
    # If no valid packing found, use a large outer hexagon
    if best_result is None:
        # Fallback: place all in a row with large outer hex
        inner = [[d * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    inner, center, side, angle = best_result
    return [[float(x), float(y), float(a)] for x, y, a in inner], center, float(side), float(angle)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
