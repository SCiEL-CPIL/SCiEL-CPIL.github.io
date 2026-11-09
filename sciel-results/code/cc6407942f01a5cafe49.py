            for q, r_coord in ring_axial:
                if len(positions) >= n_hex:
                    break
                x, y = axial_to_cartesian(q, r_coord, spacing=2.0)
                positions.append([x, y, 0.0])
