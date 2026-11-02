<<<<<<< SEARCH
    # Final validation and return
    inner, oc, os, oa = parse_params(best_params)

    if not is_valid(inner, oc, os, oa):
        # Try to fix by increasing outer side
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.05
            if os > 20:
                break
        # Check disjointness
        if not pairwise_disjoint(inner):
            # Last resort: use fallback config
            positions_a = []
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * 3.0
                    y = (row - 1) * 2.598
                    if row % 2 == 1:
                        x += 1.5
                    positions_a.append([x, y, 0.0])
            inner = [(p[0], p[1], p[2]) for p in positions_a]
            oc = (0.0, 0.0)
            oa = 30.0
            os = 8.0
            while not all_inside_outer(inner, oc, os, oa):
                os += 0.5

    return inner, oc, os, oa
=======
    # Final validation and return
    inner, oc, os, oa = parse_params(best_params)

    if not is_valid(inner, oc, os, oa):
        # Try to fix by increasing outer side
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.05
            if os > 20:
                break
        # Check disjointness
        if not pairwise_disjoint(inner):
            # Last resort: use fallback config
            positions_a = []
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * 3.0
                    y = (row - 1) * 2.598
                    if row % 2 == 1:
                        x += 1.5
                    positions_a.append([x, y, 0.0])
            inner = [(p[0], p[1], p[2]) for p in positions_a]
            oc = (0.0, 0.0)
            oa = 30.0
            os = 8.0
            while not all_inside_outer(inner, oc, os, oa):
                os += 0.5

    # Final high-precision polish: one more Nelder-Mead pass with tight tolerances
    if time.time() < deadline - 10:
        try:
            caging_params = make_params_no_outer(inner, oc, oa)
            res2 = minimize(objective_caging, caging_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 0.0005, 'fatol': 0.0002, 'adaptive': True})
            if res2.fun < os:
                inner2, oc2, oa2 = parse_params_no_outer(res2.x)
                os2 = required_outer_side_fast(inner2, oc2, oa2)
                # Verify validity
                inner_tuples2 = [(p[0], p[1], p[2]) for p in inner2]
                if pairwise_disjoint(inner_tuples2) and all_inside_outer(inner_tuples2, oc2, os2, oa2):
                    inner = inner_tuples2
                    oc = oc2
                    os = os2
                    oa = oa2
        except Exception:
            pass

    return inner, oc, os, oa
>>>>>>> REPLACE
