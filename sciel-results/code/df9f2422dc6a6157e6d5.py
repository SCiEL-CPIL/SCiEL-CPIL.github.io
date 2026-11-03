        if valid:
            new_S = S - 0.005
            if new_S < 1.0:
                new_S = 1.0
            
            # Check if current inner is valid for new_S
            new_outer_verts = get_hexagon_vertices(0.0, 0.0, new_S, 0.0)
            is_valid_new = True
            for iv in inner_verts_list:
                if not is_inside_hexagon(iv, new_outer_verts):
                    is_valid_new = False
                    break
            
            if is_valid_new:
                if new_S < best_S:
                    best_S = new_S
                    best_inner = [row[:] for row in inner]
                S = new_S
                outer_verts = new_outer_verts
            else:
                # Cannot shrink, try to move hexagons to allow shrinking?
                # Or just stop shrinking and move hexagons to reduce "slack"
                pass
