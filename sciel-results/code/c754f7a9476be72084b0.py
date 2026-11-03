<<<<<<< SEARCH
def _polish_center_angle(verts, init_cx, init_cy, init_angle, time_limit):
    """
    Jointly optimize outer hexagon center (cx, cy) and angle using the smooth
    surrogate with correct analytic gradients, then verify with exact LP.
    Returns (cx, cy, angle_deg, side) or None.
    """
    try:
        def obj_and_grad(x):
            cx, cy, ang = x
            s, gcx, gcy, gdeg = _smooth_objective_2d(verts, cx, cy, ang, beta=200.0)
            return s, np.array([gcx, gcy, gdeg])

        res = minimize(
            obj_and_grad,
            x0=[init_cx, init_cy, init_angle],
            method='L-BFGS-B',
            bounds=[(-20.0, 20.0), (-20.0, 20.0), (0.0, 30.0)],
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16}
        )
        if res.success:
            cx, cy, ang = res.x
            ang = float(np.clip(ang, 0.0, 30.0))
            exact = _min_enclosing_hex_side(verts, ang)
            if exact is not None:
                return exact[0], exact[1], ang, exact[2]
    except Exception:
        pass
    return None
=======
def _polish_center_angle(verts, init_cx, init_cy, init_angle, time_limit):
    """
    Jointly optimize outer hexagon center (cx, cy) and angle using the smooth
    surrogate with correct analytic gradients, then verify with exact LP.
    Returns (cx, cy, angle_deg, side) or None.
    """
    best_result = None
    best_side = 1e10
    
    # Try multiple beta values for the smooth surrogate
    for beta in [100.0, 200.0, 500.0]:
        if time.time() > time_limit:
            break
        try:
            def obj_and_grad(x, b=beta):
                cx, cy, ang = x
                s, gcx, gcy, gdeg = _smooth_objective_2d(verts, cx, cy, ang, beta=b)
                return s, np.array([gcx, gcy, gdeg])

            res = minimize(
                obj_and_grad,
                x0=[init_cx, init_cy, init_angle],
                method='L-BFGS-B',
                bounds=[(-20.0, 20.0), (-20.0, 20.0), (0.0, 30.0)],
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if res.success:
                cx, cy, ang = res.x
                ang = float(np.clip(ang, 0.0, 30.0))
                exact = _min_enclosing_hex_side(verts, ang)
                if exact is not None:
                    if exact[2] < best_side:
                        best_side = exact[2]
                        best_result = (exact[0], exact[1], ang, exact[2])
        except Exception:
            pass
    
    # Final golden-section polish on angle at the best center
    if best_result is not None and time.time() < time_limit:
        best_cx, best_cy, best_ang, _ = best_result
        for delta in [0.001, 0.0001]:
            if time.time() > time_limit:
                break
            lo = max(0.0, best_ang - delta)
            hi = min(30.0, best_ang + delta)
            gr = (math.sqrt(5) - 1) / 2
            a = lo
            b = hi
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = _min_enclosing_hex_side(verts, c)
            fd = _min_enclosing_hex_side(verts, d)
            fc = fc[2] if fc else 1e6
            fd = fd[2] if fd else 1e6
            while (b - a) > 1e-12 and time.time() < time_limit:
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = _min_enclosing_hex_side(verts, c)
                    fc = fc[2] if fc else 1e6
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = _min_enclosing_hex_side(verts, d)
                    fd = fd[2] if fd else 1e6
            mid = (a + b) / 2
            res = _min_enclosing_hex_side(verts, mid)
            if res:
                if res[2] < best_side:
                    best_side = res[2]
                    best_result = (res[0], res[1], float(mid), res[2])
    
    return best_result
>>>>>>> REPLACE
