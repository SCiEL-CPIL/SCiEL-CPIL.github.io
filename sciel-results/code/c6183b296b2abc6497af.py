    def get_side_and_penalty(params):
        p_inner = params[:33]
        cx = params[33]
        cy = params[34]
        o_ang = params[35]
        
        curr_inner = []
        verts_list = []
        for i in range(N):
            x = p_inner[3*i]
            y = p_inner[3*i+1]
            a = p_inner[3*i+2]
            curr_inner.append([x, y, a])
            verts_list.append(hex_vertices(x, y, 1.0, a))
            
        # Compute outer side
        side = compute_outer_side(curr_inner, (cx, cy), o_ang)
        
        penalty = 0.0
        for i in range(N):
            x1, y1, a1 = curr_inner[i]
            v1 = verts_list[i]
            for j in range(i+1, N):
                x2, y2, a2 = curr_inner[j]
                v2 = verts_list[j]
                
                dx = x1 - x2
                dy = y1 - y2
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 1.75:
                    penalty += (1.75 - dist) ** 2 * 100.0
                    
        return side + penalty
