<<<<<<< SEARCH
    # Gradient-based refinement as a final polish
    if time.time() < _DEADLINE - 5.0:
        r_est = best_score
        refined = _refine_with_gradient_descent(best_pts, r_est, steps=150, lr=0.003)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined
=======
    # Dynamics-based relaxation to self-organize dense packing
    if time.time() < _DEADLINE - 25.0:
        r_est = best_score
        dyn_pts = _dynamics_relax(best_pts, r_est, steps=400, dt=0.0015, damping=0.97)
        dyn_score = _score(dyn_pts)
        if dyn_score > best_score:
            best_score = dyn_score
            best_pts = dyn_pts.copy()

    # Gradient-based refinement as a final polish
    if time.time() < _DEADLINE - 5.0:
        r_est = best_score
        refined = _refine_with_gradient_descent(best_pts, r_est, steps=150, lr=0.003)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined
>>>>>>> REPLACE

