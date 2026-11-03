<<<<<<< SEARCH
                for outer_angle_deg in np.arange(lo, hi + 0.001, 0.001):
                    if time.time() > _DEADLINE - 20.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_local_side:
                            best_local_side = side
                            best_local_angle = outer_angle_dec if False else outer_angle_deg
=======
                for outer_angle_deg in np.arange(lo, hi + 0.001, 0.001):
                    if time.time() > _DEADLINE - 20.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_local_side:
                            best_local_side = side
                            best_local_angle = outer_angle_deg
>>>>>>> REPLACE
