    # Check inner hexagons are separated
    for i in range(12):
        for j in range(i + 1, 12):
            cx1, cy1, ca1 = inner_data[i]
            cx2, cy2, ca2 = inner_data[j]
            v1 = _hex_vertices(cx1, cy1, 1.0, ca1)
            v2 = _hex_vertices(cx2, cy2, 1.0, ca2)
            
            max_gap = -np.inf
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    norm = np.linalg.norm(edge)
                    if norm < 1e-15:
                        continue
                    normal = np.array([-edge[1], edge[0]]) / norm
                    
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    gap = max(min1 - max2, min2 - max1)
                    if gap > max_gap:
                        max_gap = gap
            
            if max_gap <= 1e-6:
                return False
