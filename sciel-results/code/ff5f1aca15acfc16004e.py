# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def hexagon_packing_12():
    # Define 12 hexagon centers using axial coordinates
    # (q, r) -> x = 1.5*q, y = sqrt(3)*(r + q/2)
    # This configuration is a known valid packing (no overlaps)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (-2, 1), (-2, 0)
    ]
    
    inner = np.zeros((12, 3))
    all_vertices = []
    
    for i, (q, r) in enumerate(axial_coords):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2)
        inner[i] = [x, y, 0.0]
        verts = _hex_vertices(x, y, 1.0, 0.0)
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    # Calculate minimal side length s for outer hexagon (center 0,0, angle 0)
    # to contain all vertices.
    # Outer hexagon edges normals (angle 0):
    # Normals point to midpoints of edges.
    # Edge midpoints angles: 30, 90, 150, 210, 270, 330 degrees.
    # Distance from center to edge (inradius) is s * sqrt(3) / 2.
    # Condition: dot(v, n) <= s * sqrt(3) / 2 for all normals n and vertices v.
    # s >= dot(v, n) * 2 / sqrt(3)
    
    normals = []
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    # Project all vertices onto all normals
    # dots shape: (12*6, 6)
    dots = all_vertices @ normals.T
    
    # For each normal, find the max projection
    max_projections = dots.max(axis=0)
    
    # Required s for each normal
    required_s = max_projections * 2 / np.sqrt(3)
    
    # The overall s must satisfy all
    s = required_s.max()
    
    # Add a tiny margin for safety against floating point errors
    s += 1e-9
    
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, s

# EVOLVE-BLOCK-END
