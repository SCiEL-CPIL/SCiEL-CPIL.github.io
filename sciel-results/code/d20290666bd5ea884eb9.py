<<<<<<< SEARCH
    def compute_side_from_angle(angle_deg):
        """Compute the minimum outer hexagon side length for a given global rotation angle."""
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate all vertices
        rotated = flat_verts @ R.T
        
        # For the outer hexagon with rotation angle phi, we need to find
        # the minimum side length such that all rotated vertices are inside.
        # 
        # A regular hexagon centered at origin with side length L and rotation phi:
        # The six facet normals are at angles phi, phi+60, phi+120, phi+180, phi+240, phi+300
        # The distance from center to each facet is L * sqrt(3)/2
        # A point (x,y) is inside iff for all k: dot(point, normal_k) <= L * sqrt(3)/2
        # So L >= 2 * max_k(dot(point, normal_k)) / sqrt(3)
        #
        # We can optimize over phi as well. But for a rigid cluster, let's first
        # try phi = 0 (outer hexagon aligned with one of the inner hexagons' orientations)
        # Actually, we should also optimize phi. But let's start simple.
        
        # Try several outer hexagon angles
        best_L = float('inf')
        for phi_deg in [0, 30]:
            phi_rad = np.deg2rad(phi_deg)
            normals = np.array([
                [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
                for k in range(6)
            ])
            projs = rotated @ normals.T  # (66, 6)
            max_proj = np.max(projs)
            L = 2.0 * max_proj / np.sqrt(3)
            if L < best_L:
                best_L = L
        
        return best_L
=======
    def compute_side_from_angles(angle_deg, phi_deg):
        """Compute the minimum outer hexagon side length for given cluster and outer rotation angles."""
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate all vertices
        rotated = flat_verts @ R.T
        
        # For the outer hexagon with rotation angle phi, we need to find
        # the minimum side length such that all rotated vertices are inside.
        # A regular hexagon centered at origin with side length L and rotation phi:
        # The six facet normals are at angles phi, phi+60, ..., phi+300
        # The distance from center to each facet is L * sqrt(3)/2
        # A point (x,y) is inside iff for all k: dot(point, normal_k) <= L * sqrt(3)/2
        # So L >= 2 * max_k(dot(point, normal_k)) / sqrt(3)
        
        phi_rad = np.deg2rad(phi_deg)
        normals = np.array([
            [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
            for k in range(6)
        ])
        projs = rotated @ normals.T  # (66, 6)
        max_proj = np.max(projs)
        L = 2.0 * max_proj / np.sqrt(3)
        
        return L
>>>>>>> REPLACE

