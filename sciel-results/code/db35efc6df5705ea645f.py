<<<<<<< SEARCH
    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)
=======
    def soft_max(values, temperature=0.05):
        """
        Soft maximum of values using Log-Sum-Exp.
        If temperature is very small, it approximates the max.
        """
        if temperature < 1e-9:
            return np.max(values)
        # Shift for numerical stability
        max_val = np.max(values)
        # If max_val is very large positive, exp might overflow, but we subtract max_val
        # If max_val is very large negative, exp(0) is fine.
        # The range of projections is small, so this is safe.
        return max_val + temperature * math.log(np.sum(np.exp((values - max_val) / temperature)))

    def calc_min_outer_side(positions, angles, outer_angle=0.0, temp=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        Uses soft max if temp > 0.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        # Collect all max projections for each hexagon against each normal
        # We need the global max over all vertices of all hexagons.
        # projs[i, k] is max projection of hex i on normal k.
        # We want max_{i, k} projs[i, k].
        
        if temp < 1e-9:
            max_proj = 0.0
            for i in range(N):
                cx, cy = positions[i]
                ang = angles[i]
                verts = get_hex_verts(cx, cy, 1.0, ang)
                projs = verts @ normals.T # (6, 6)
                max_proj_i = np.max(projs)
                if max_proj_i > max_proj:
                    max_proj = max_proj_i
            return max_proj * 2.0 / math.sqrt(3)
        else:
            all_max_projs = []
            for i in range(N):
                cx, cy = positions[i]
                ang = angles[i]
                verts = get_hex_verts(cx, cy, 1.0, ang)
                projs = verts @ normals.T # (6, 6)
                # For each normal, find max projection for this hex
                hex_max_projs = np.max(projs, axis=0) # (6,)
                all_max_projs.extend(hex_max_projs.tolist())
            
            all_max_projs = np.array(all_max_projs)
            # Soft max over all these values
            smooth_max_proj = soft_max(all_max_projs, temp)
            return smooth_max_proj * 2.0 / math.sqrt(3)
>>>>>>> REPLACE

