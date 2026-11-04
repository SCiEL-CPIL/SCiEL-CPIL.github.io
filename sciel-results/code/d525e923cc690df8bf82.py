"""Packing 12 unit hexagons in a regular hexagon - hexagonal lattice arrangement."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.pi * np.arange(6) / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    # Transform to hexagon's local frame
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    dx, dy = px - cx, py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # Regular hexagon with side s has vertices at distance s from center
    # In local frame, the hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes
    # For a hexagon with circumradius = side (for regular hexagon), 
    # the distance from center to each side (inradius) = side * sqrt(3)/2
    # The half-plane constraints: for each edge, the point must be on the interior side
    inradius = side * np.sqrt(3) / 2
    # The 6 edges are at angles 30, 90, 150, 210, 270, 330 degrees from center
    # Each edge is at distance inradius from center
    for k in range(6):
        edge_angle = np.radians(30 + 60 * k)
        ex, ey = np.cos(edge_angle), np.sin(edge_angle)
        # The edge normal points outward
        dist = lx * ex + ly * ey
        if dist > inradius + 1e-9:
            return False
    return True

def hexagon_distance(p, cx, cy, side, angle_deg):
    """Return signed distance from point p to hexagon (negative if inside)."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    dx, dy = p[0] - cx, p[1] - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    inradius = side * np.sqrt(3) / 2
    max_dist = 0
    for k in range(6):
        edge_angle = np.radians(30 + 60 * k)
        ex, ey = np.cos(edge_angle), np.sin(edge_angle)
        dist = lx * ex + ly * ey - inradius
        max_dist = max(max_dist, dist)
    return max_dist

def hex_hex_overlap(d, side1, side2):
    """Return overlap distance between two hexagons with center distance d."""
    # For two regular hexagons with sides s1, s2, the minimum distance between
    # their boundaries when centers are at distance d
    # The inradii are r1 = s1*sqrt(3)/2, r2 = s2*sqrt(3)/2
    # If d >= r1 + r2 + something, they don't overlap
    # This is complex; use vertex-based check
    return None  # We'll handle this differently

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Hexagonal lattice arrangement for 12 unit hexagons
    # Arrangement: 1 center, 6 in first ring, 5 in second ring
    # For unit hexagons, center-to-center distance for adjacent = sqrt(3)
    
    sqrt3 = np.sqrt(3)
    
    # Define positions in hexagonal coordinates
    # Hex grid: axial coordinates (q, r) -> cartesian
    # For flat-top hexagons: x = sqrt(3) * (q + r/2), y = 3/2 * r
    # For pointy-top: x = 3/2 * q, y = sqrt(3) * (r + q/2)
    
    # Let's use a standard hexagonal packing
    # Centers at positions forming a hexagonal pattern
    positions = []
    angles = []
    
    # Center hexagon
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # First ring: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        a = np.radians(60 * k)
        positions.append((sqrt3 * np.cos(a), sqrt3 * np.sin(a)))
        angles.append(0.0)
    
    # Second ring: 5 more hexagons (to make 12 total)
    # Place them at distance 2*sqrt(3) from center, at angles 0, 60, 120, 180, 240
    for k in range(5):
        a = np.radians(60 * k)
        positions.append((2 * sqrt3 * np.cos(a), 2 * sqrt3 * np.sin(a)))
        angles.append(0.0)
    
    # Now find the minimum outer hexagon that contains all these
    # The outer hexagon is regular, centered at origin, with some side length S and angle
    
    def outer_contains_all(S, angle_deg, pos_list):
        """Check if outer hexagon with side S, angle contains all inner hexagons."""
        for (cx, cy), inner_angle in zip(pos_list, angles):
            # Check all 6 vertices of inner hexagon
            inner_verts = hex_vertices(cx, cy, 1.0, inner_angle)
            for vx, vy in inner_verts:
                if not point_in_hexagon(vx, vy, 0.0, 0.0, S, angle_deg):
                    return False
        return True
    
    # Binary search on S for each angle
    def min_S_for_angle(angle_deg):
        lo, hi = 2.0, 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if outer_contains_all(mid, angle_deg, positions):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Try different outer hexagon angles
    best_S = float('inf')
    best_angle = 0.0
    
    for angle_deg in np.arange(0, 30, 1.0):  # Symmetry: 60 degree period, but let's check 0-30
        S = min_S_for_angle(angle_deg)
        if S < best_S:
            best_S = S
            best_angle = angle_deg
    
    # Refine around best angle
    if time.time() < deadline:
        for angle_deg in np.arange(best_angle - 5, best_angle + 5, 0.1):
            S = min_S_for_angle(angle_deg)
            if S < best_S:
                best_S = S
                best_angle = angle_deg
    
    # Further refinement
    if time.time() < deadline:
        for angle_deg in np.arange(best_angle - 0.5, best_angle + 0.5, 0.01):
            S = min_S_for_angle(angle_deg)
            if S < best_S:
                best_S = S
                best_angle = angle_deg
    
    # Build inner list
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    
    return inner, (0.0, 0.0), best_S, best_angle
