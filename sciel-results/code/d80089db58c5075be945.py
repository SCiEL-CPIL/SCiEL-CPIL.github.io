        # Precompute vertices
        verts_list = [hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
        all_verts = np.vstack(verts_list)
        
        # ... containment ...
        
        # Separation
        for i in range(N):
            for j in range(i+1, N):
                margin = sat_margin_v(verts_list[i], verts_list[j])
                # ...
