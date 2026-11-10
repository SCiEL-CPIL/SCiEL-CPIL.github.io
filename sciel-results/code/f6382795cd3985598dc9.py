            # Check if we can add pt
            # 1. Must have at least one line L with k[L]=0
            has_witness = False
            for L in point_to_lines[pt]:
                if k[L] == 0:
                    has_witness = True
                    break
            if not has_witness:
                continue

            valid = True
            for L in point_to_lines[pt]:
                if k[L] == 0:
                    # Check other points on L
                    for r in lines[L]:
                        if r == pt:
                            continue
                        if safe[r] == 1:
                            valid = False
                            break
                    if not valid:
                        break
                elif k[L] == 1:
                    q = line_point_in_C[L]
                    if safe[q] == 1:
                        valid = False
                        break
                        
            if valid:
                # Commit
                ...
