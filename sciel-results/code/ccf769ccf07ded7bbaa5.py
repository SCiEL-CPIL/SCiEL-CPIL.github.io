<<<<<<< SEARCH
def _polish(pos, radii, deadline):
    """Multi-stage polish: L-BFGS-B, force-directed annealing, then SLSQP."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.001, pos[i, 0] - radii[i] * 0.5), min(0.999, pos[i, 0] + radii[i] * 0.5)))
        bounds.append((max(0.001, pos[i, 1] - radii[i] * 0.5), min(0.999, pos[i, 1] + radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    # Stage 1: L-BFGS-B with penalty objective
    if time.time() < deadline - 15:
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                s = np.sum(radii_pp)
                if s > best_score:
                    best_score = s
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass

    # Stage 2: Force-directed annealing to escape local optima
    if time.time() < deadline - 10:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, deadline)
            if _is_valid(pos_fd, radii_fd):
                s = np.sum(radii_fd)
                if s > best_score:
                    best_score = s
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
        except Exception:
            pass

    # Stage 3: SLSQP with hard constraints (if time allows)
    if time.time() < deadline - 3:
        try:
            pos_s, radii_s = _slsqp_polish(best_pos, best_radii, deadline)
            if _is_valid(pos_s, radii_s):
                s_new = np.sum(radii_s)
                if s_new >= best_score - 1e-10:
                    best_score = s_new
                    best_pos = pos_s.copy()
                    best_radii = radii_s.copy()
        except Exception:
            pass

    return best_pos, best_radii
=======
def _polish(pos, radii, deadline):
    """Multi-stage polish: L-BFGS-B, force-directed annealing, then SLSQP."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.001, pos[i, 0] - radii[i] * 0.5), min(0.999, pos[i, 0] + radii[i] * 0.5)))
        bounds.append((max(0.001, pos[i, 1] - radii[i] * 0.5), min(0.999, pos[i, 1] + radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    # Stage 1: L-BFGS-B with penalty objective
    if time.time() < deadline - 15:
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                s = np.sum(radii_pp)
                if s > best_score:
                    best_score = s
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass

    # Stage 2: Force-directed annealing to escape local optima
    if time.time() < deadline - 10:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, deadline)
            if _is_valid(pos_fd, radii_fd):
                s = np.sum(radii_fd)
                if s > best_score:
                    best_score = s
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
        except Exception:
            pass

    # Stage 3: SLSQP with hard constraints (if time allows)
    if time.time() < deadline - 3:
        try:
            pos_s, radii_s = _slsqp_polish(best_pos, best_radii, deadline)
            if _is_valid(pos_s, radii_s):
                s_new = np.sum(radii_s)
                if s_new >= best_score - 1e-10:
                    best_score = s_new
                    best_pos = pos_s.copy()
                    best_radii = radii_s.copy()
        except Exception:
            pass

    return best_pos, best_radii


def _polish_voronoi(centers, deadline):
    """Multi-stage polish in 26-dim center space using Voronoi radii.
    
    Stage 1: L-BFGS-B on Voronoi objective
    Stage 2: Voronoi force-directed annealing
    Stage 3: SLSQP on Voronoi objective
    """
    centers = centers.copy()
    best_pos = centers.copy()
    best_radii = _voronoi_radii(centers)
    best_score = np.sum(best_radii)
    
    # Stage 1: L-BFGS-B
    if time.time() < deadline - 20:
        try:
            x0 = centers.flatten()
            bounds = [(0.01, 0.99)] * (2 * N)
            result = minimize(_voronoi_objective, x0, args=(deadline,),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            centers_opt = result.x.reshape(N, 2)
            pos_pp, radii_pp = _post_process_voronoi(centers_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                s = np.sum(radii_pp)
                if s > best_score:
                    best_score = s
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass

    # Stage 2: Voronoi force-directed annealing
    if time.time() < deadline - 10:
        try:
            pos_fd, radii_fd = _voronoi_force_anneal(best_pos, deadline)
            if _is_valid(pos_fd, radii_fd):
                s = np.sum(radii_fd)
                if s > best_score:
                    best_score = s
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
        except Exception:
            pass

    # Stage 3: SLSQP on Voronoi objective
    if time.time() < deadline - 5:
        try:
            pos_s, radii_s = _slsqp_voronoi_polish(best_pos, deadline)
            if _is_valid(pos_s, radii_s):
                s_new = np.sum(radii_s)
                if s_new > best_score:
                    best_score = s_new
                    best_pos = pos_s.copy()
                    best_radii = radii_s.copy()
        except Exception:
            pass

    return best_pos, best_radii
>>>>>>> REPLACE
