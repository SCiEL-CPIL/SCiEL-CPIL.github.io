<<<<<<< SEARCH
    def compute_min_outer_side(inner):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        We optimize the rotation angle of the outer hexagon.
        The outer hexagon is centered at the centroid of the inner configuration.
        """
        # 1. Get all vertices of inner hexagons
        all_verts = []
        for h in inner:
            all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
        all_verts = np.vstack(all_verts)
        
        # 2. Compute centroid (approximate center for outer hex)
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        
        # 3. Find optimal rotation angle phi for outer hexagon
        # Outer hexagon normals are at angles phi + 30 + k*60
        # Required side S = max_k ( 2/sqrt(3) * max_v | n_k . offset_v | )
        # We scan phi in [0, 60)
        
        sqrt3 = math.sqrt(3)
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            phi = np.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                # Project all offsets
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                # S = 2/sqrt(3) * max_proj
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg
                
        # Fine scan around best_phi
        for phi_deg in np.arange(best_phi - 2, best_phi + 2, 0.1):
            if phi_deg < 0 or phi_deg >= 60: continue
            phi = np.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg

        # Polish: Use scipy minimize on the phi angle to refine
        # The objective function for a given phi is the max extent
        def obj_phi(phi_rad):
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi_rad + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            return max_extent

        # Start from best_phi found in scan
        phi_init = np.radians(best_phi)
        # Try a few local optimizations from nearby starts to avoid local minima
        for start_offset in [0, 0.1, -0.1, 0.2, -0.2]:
            start_phi = phi_init + start_offset
            try:
                res = minimize(obj_phi, start_phi, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = np.degrees(res.x[0]) % 60.0
            except:
                pass

        return best_side, best_phi, center
=======
    def compute_min_outer_side(inner):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Fast vectorized implementation.
        """
        # 1. Get all vertices of inner hexagons
        all_verts = []
        for h in inner:
            all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
        all_verts = np.vstack(all_verts)
        
        sqrt3 = math.sqrt(3)
        best_side = float('inf')
        best_phi = 0.0
        
        # Scan phi in [0, 60)
        # We only need 3 unique normals for the width calculation
        # Normals are at phi + 30, phi + 90, phi + 150
        # Or phi + pi/6 + k*pi/3 for k=0,1,2
        
        # Use a reasonably fine scan for speed
        phi_steps = np.arange(0, 60, 0.5)
        
        for phi_deg in phi_steps:
            phi = np.radians(phi_deg)
            # 3 unique normal directions
            angles = phi + np.pi/6 + np.array([0, np.pi/3, 2*np.pi/3])
            cos_angles = np.cos(angles)
            sin_angles = np.sin(angles)
            
            # Project all vertices onto these 3 directions
            # all_verts is (72, 2)
            # cos_angles is (3,)
            # proj is (72, 3)
            proj = all_verts[:, 0:1] * cos_angles[None, :] + all_verts[:, 1:2] * sin_angles[None, :]
            
            # Compute range for each direction
            ranges = np.max(proj, axis=0) - np.min(proj, axis=0)
            
            # Side length is max range / sqrt(3)
            side = np.max(ranges) / sqrt3
            
            if side < best_side:
                best_side = side
                best_phi = phi_deg
        
        # Refine around best_phi
        if best_phi > 0 and best_phi < 60:
            refine_steps = np.arange(best_phi - 0.5, best_phi + 0.5, 0.05)
            for phi_deg in refine_steps:
                if phi_deg < 0 or phi_deg >= 60: continue
                phi = np.radians(phi_deg)
                angles = phi + np.pi/6 + np.array([0, np.pi/3, 2*np.pi/3])
                cos_angles = np.cos(angles)
                sin_angles = np.sin(angles)
                
                proj = all_verts[:, 0:1] * cos_angles[None, :] + all_verts[:, 1:2] * sin_angles[None, :]
                ranges = np.max(proj, axis=0) - np.min(proj, axis=0)
                side = np.max(ranges) / sqrt3
                
                if side < best_side:
                    best_side = side
                    best_phi = phi_deg

        # Compute center for the best_phi
        # The center should be the midpoint of the projections for the optimal orientation
        phi = np.radians(best_phi)
        angles = phi + np.pi/6 + np.array([0, np.pi/3, 2*np.pi/3])
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        proj = all_verts[:, 0:1] * cos_angles[None, :] + all_verts[:, 1:2] * sin_angles[None, :]
        
        # mid_proj[k] = (max(proj[:,k]) + min(proj[:,k])) / 2
        # We want c such that n_k . c = mid_proj[k]
        # This is an overdetermined system (3 eqs, 2 unknowns). Use least squares.
        # A c = b
        # A is 3x2, b is 3x1
        A = np.column_stack((cos_angles, sin_angles))
        b = (np.max(proj, axis=0) + np.min(proj, axis=0)) / 2.0
        
        center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        return best_side, best_phi, center
>>>>>>> REPLACE

This change makes `compute_min_outer_side` significantly faster (no inner loops over normals, no scipy calls). This allows the SA to run many more iterations.

Now I need to adjust the SA loop to take advantage of this speed. I can increase `max_steps` and potentially adjust the cooling schedule.

