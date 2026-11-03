"""Program for packing 12 unit regular hexagons inside a regular hexagon."""
import numpy as np
from itertools import product

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Place 12 disjoint unit regular hexagons inside a regular hexagon,
    minimizing the outer side length.
    
    Approach: Grid-based discrete search on a triangular lattice with
    a single global rotation angle. We search over rotation angles and
    lattice configurations to find the minimal bounding hexagon.
    """
    
    def hex_vertices(cx, cy, side, angle_deg=0.0):
        """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)
    
    def hex_corner_distances(vertices, center):
        """Return squared distances from center to each vertex."""
        dx = vertices[:, 0] - center[0]
        dy = vertices[:, 1] - center[1]
        return dx**2 + dy**2
    
    def min_enclosing_hexagon_side(points, angle_deg=0.0):
        """
        Find the minimum side length of a regular hexagon (with given orientation)
        that encloses all given points.
        
        For a regular hexagon with side length s and orientation angle, the width
        in each of the 6 normal directions must be >= 2s.
        We compute the required s by checking projections onto the 6 normals.
        """
        angle_rad = np.radians(angle_deg)
        
        # The 6 outward normal directions for a regular hexagon
        # For a hexagon with flat top (vertex at angle 0), normals are at 0, 60, 120, 180, 240, 300 degrees
        # Actually, let's think about this differently.
        
        # A regular hexagon can be described as the intersection of 6 half-planes.
        # For a hexagon centered at origin with side length s and rotation angle alpha:
        # The vertices are at angles alpha + k*60 degrees, distance s from center.
        # The edges are perpendicular to directions alpha + k*60 + 30 degrees.
        # The distance from center to each edge (apothem) is s * cos(pi/6) = s * sqrt(3)/2.
        
        # So the hexagon is: for each of 6 directions d_k = alpha + k*60 + 30,
        # the projection of any point onto d_k must be in [-apothem, apothem].
        
        # To enclose all points, we need:
        # For each direction d_k, max over points of |dot(p - c, d_k)| <= apothem
        # where c is the hexagon center.
        
        # To minimize s (and thus apothem), we should center the hexagon appropriately.
        # For a fixed orientation, the optimal center is such that the range of projections
        # in each direction is minimized.
        
        # Actually, for a given set of points and a given hexagon orientation, the minimum
        # enclosing hexagon has its center at the point that minimizes the maximum projection.
        
        # Let's use a different approach: for a given orientation, the minimum s is:
        # s = max over all pairs of points and all 6 directions of the required constraint.
        
        # More precisely: for direction d (unit vector), if we center at c, we need
        # max_i dot(p_i - c, d) <= apothem and min_i dot(p_i - c, d) >= -apothem
        # So apothem >= (max_i dot(p_i, d) - min_i dot(p_i, d)) / 2
        # (the center c cancels out for the range)
        
        # Wait, that's not quite right. Let me reconsider.
        
        # If the hexagon is centered at c, then for direction d_k:
        # max_i dot(p_i - c, d_k) <= apothem
        # min_i dot(p_i - c, d_k) >= -apothem
        # => apothem >= max( max_i dot(p_i, d_k) - dot(c, d_k), dot(c, d_k) - min_i dot(p_i, d_k) )
        
        # To minimize apothem for this direction, we choose dot(c, d_k) = (max_i dot(p_i, d_k) + min_i dot(p_i, d_k)) / 2
        # Then apothem >= (max_i dot(p_i, d_k) - min_i dot(p_i, d_k)) / 2
        
        # But c must work for all 6 directions simultaneously. So we can't independently optimize each.
        
        # For simplicity, let's assume the hexagon center is at the centroid of the points,
        # or we can try a few candidate centers.
        
        # Actually, for the minimum enclosing regular hexagon with fixed orientation,
        # the optimal center is the one that minimizes the maximum over the 6 directions
        # of (max_i dot(p_i - c, d_k) - min_i dot(p_i - c, d_k)) / 2... no wait.
        
        # Let me just compute: for a given center c and orientation, the required apothem is:
        # apothem = max over k of max( max_i dot(p_i - c, d_k), -min_i dot(p_i - c, d_k) )
        # And s = apothem / cos(pi/6) = apothem * 2/sqrt(3)
        
        # To find the best c, we can note that the optimal c is such that the hexagon is "tight"
        # in at least 3 directions. This is a 2D optimization problem.
        
        # For simplicity and speed, let's use the centroid as the center, and also try
        # a few other centers.
        
        centroid = np.mean(points, axis=0)
        
        best_s = float('inf')
        
        # Try centroid and a few offsets
        candidate_centers = [centroid]
        
        # Also try the center of the bounding box
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        candidate_centers.append((bbox_min + bbox_max) / 2)
        
        for c in candidate_centers:
            apothem_required = 0.0
            for k in range(6):
                d_k = np.array([np.cos(angle_rad + k * np.pi / 3 + np.pi / 6),
                                np.sin(angle_rad + k * np.pi / 3 + np.pi / 6)])
                projections = np.dot(points - c, d_k)
                proj_max = np.max(projections)
                proj_min = np.min(projections)
                # Need: proj_max <= apothem and proj_min >= -apothem
                # So apothem >= max(proj_max, -proj_min)
                apothem_required = max(apothem_required, proj_max, -proj_min)
            
            s = apothem_required * 2 / np.sqrt(3)
            best_s = min(best_s, s)
        
        return best_s
    
    def get_lattice_points(radius, spacing):
        """
        Generate points on a triangular lattice within a given radius.
        Triangular lattice: points at (i*a + j*a/2, j*a*sqrt(3)/2) for integers i, j.
        """
        points = []
        # Determine range of i, j
        max_idx = int(np.ceil(radius / spacing)) + 1
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                x = i * spacing + j * spacing / 2
                y = j * spacing * np.sqrt(3) / 2
                if np.sqrt(x**2 + y**2) <= radius:
                    points.append((x, y))
        return np.array(points)
    
    def check_disjoint(hex_centers, hex_side=1.0):
        """
        Check if unit hexagons at given centers are disjoint (non-overlapping).
        Two unit hexagons are disjoint if the distance between their centers is >= 2.
        (The minimum distance between centers of two non-overlapping unit hexagons
        is 2, since the inradius of a unit hexagon is sqrt(3)/2, and they touch
        when center distance = 2 * inradius... wait, no.)
        
        Actually, for two regular hexagons of side 1, the minimum center-to-center
        distance for them to be disjoint depends on their relative orientation.
        If they have the same orientation, the minimum distance is 2 (they share an edge
        when distance = 2).
        
        For safety, let's use a slightly larger threshold.
        """
        n = len(hex_centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = hex_centers[i][0] - hex_centers[j][0]
                dy = hex_centers[i][1] - hex_centers[j][1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 2.0 - 1e-9:
                    return False
        return True
    
    # Search over rotation angles
    best_result = None
    best_side = float('inf')
    
    # Try various rotation angles
    angles = np.linspace(0, 60, 61)  # 61 angles from 0 to 60 degrees
    
    for angle_deg in angles:
        # Generate lattice points
        # The lattice spacing should be at least 2 to ensure disjointness
        # Try different spacings
        for spacing in [2.0, 2.05, 2.1, 2.2]:
            lattice_points = get_lattice_points(radius=15, spacing=spacing)
            
            if len(lattice_points) < N:
                continue
            
            # Try to select N points from the lattice
            # For simplicity, try contiguous blocks or random selections
            # Let's try a simple approach: take the first N points, or various subsets
            
            # Try selecting points that are close to the origin
            # Sort by distance from origin and take the first N
            dists = np.sqrt(lattice_points[:, 0]**2 + lattice_points[:, 1]**2)
            indices = np.argsort(dists)
            
            # Try different starting points in the sorted list
            for start in range(min(20, len(indices) - N + 1)):
                selected_indices = indices[start:start + N]
                hex_centers = lattice_points[selected_indices]
                
                if not check_disjoint(hex_centers):
                    continue
                
                # Compute the minimum enclosing hexagon side
                s = min_enclosing_hexagon_side(hex_centers, angle_deg)
                
                if s < best_side:
                    best_side = s
                    # Store the configuration
                    inner = []
                    for cx, cy in hex_centers:
                        inner.append([float(cx), float(cy), float(angle_deg)])
                    # The outer hexagon center is the centroid
                    outer_center = np.mean(hex_centers, axis=0)
                    best_result = (inner, (float(outer_center[0]), float(outer_center[1])), float(s), float(angle_deg))
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: return a valid but not optimal configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
