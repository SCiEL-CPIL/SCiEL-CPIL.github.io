        delta_violations = 0
        for ap_id in ap_indices[idx]:
            ap = aps[ap_id]
            # Get colors
            c0 = new_color if ap[0] == idx else coloring[ap[0]]
            c1 = new_color if ap[1] == idx else coloring[ap[1]]
            c2 = new_color if ap[2] == idx else coloring[ap[2]]
            c3 = new_color if ap[3] == idx else coloring[ap[3]]
            
            new_is_violating = len({c0, c1, c2, c3}) < 3
            old_is_violating = ap_violating[ap_id]
            
            if old_is_violating and not new_is_violating:
                delta_violations -= 1
            elif not old_is_violating and new_is_violating:
                delta_violations += 1
                
        delta_distinct = 0
        if color_counts[old_color] == 1:
            delta_distinct -= 1
        if color_counts[new_color] == 0:
            delta_distinct += 1
