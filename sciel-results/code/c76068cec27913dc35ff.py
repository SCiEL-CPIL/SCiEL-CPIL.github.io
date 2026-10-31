    def calc_S_angle(axial_list, inner_angle_deg):
        inner_angle_rad = math.radians(inner_angle_deg)
        centers = []
        for q, r in axial_list:
            x = 1.5 * q
            y = math.sqrt(3) * (r + q / 2.0)
            centers.append((x, y))
            
        normals = []
        for i in range(6):
            theta = math.pi / 6.0 + i * math.pi / 3.0
            normals.append((math.cos(theta), math.sin(theta)))
            
        max_proj = 0.0
        for cx, cy in centers:
            for i in range(6):
                theta = inner_angle_rad + i * math.pi / 3.0
                vx = cx + math.cos(theta)
                vy = cy + math.sin(theta)
                for nx, ny in normals:
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
                        
        S = 2.0 * max_proj / math.sqrt(3.0)
        return S, centers
