    def objective_exact(params):
        inner_hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            angle = params[i*3+2]
            inner_hexes.append((x, y, angle))
        
        outer_cx = params[N*3]
        outer_cy = params[N*3+1]
        outer_angle = params[N*3+2]
        
        penalty = 0.0
        verts_list = [hex_vertices(*inner_hexes[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += 100.0
        
        side = compute_min_side(inner_hexes, (outer_cx, outer_cy), outer_angle)
        return side + penalty
