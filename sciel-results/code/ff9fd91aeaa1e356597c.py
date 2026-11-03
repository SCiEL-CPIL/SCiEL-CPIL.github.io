<<<<<<< SEARCH
            # Refine
            if time.time() < _DEADLINE - 10.0:
                refined_angle, refined_side, refined_center = _refine_outer_angle(
                    inner_hexes, math.radians(best_coarse_angle), tol=1e-5)
                
                if refined_side < best_side:
                    best_side = refined_side
                    best_inner = [[x, y, 0.0] for x, y in full_xy]
                    best_center = refined_center
                    best_outer_angle = refined_angle * 180.0 / math.pi
=======
            # Refine
            if time.time() < _DEADLINE - 10.0:
                refined_angle, refined_side, refined_center = _refine_outer_angle(
                    inner_hexes, math.radians(best_coarse_angle), tol=1e-5)
                
                # High-precision polish
                if time.time() < _DEADLINE - 8.0:
                    polished_angle, polished_side, polished_center = _polish_outer_hex(
                        inner_hexes, refined_angle, refined_center, refined_side, max_iter=200)
                    if polished_side < refined_side - 1e-9:
                        refined_side = polished_side
                        refined_angle = polished_angle
                        refined_center = polished_center
                
                if refined_side < best_side:
                    best_side = refined_side
                    best_inner = [[x, y, 0.0] for x, y in full_xy]
                    best_center = refined_center
                    best_outer_angle = refined_angle * 180.0 / math.pi
>>>>>>> REPLACE
