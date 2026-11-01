<<<<<<< SEARCH
    # Phase 3: Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        r_target = best_r * 1.0005
        try_improve(best_pts, r_target, 3000)
    
    # Phase 4: Very final polish
    if time.time() < deadline - 2.0:
        r_target = best_r * 1.0002
        try_improve(best_pts, r_target, 2000)
    
    return best_pts
=======
    # Phase 3: Final polish with tighter tolerance
    if time.time() < deadline - 10.0:
        r_target = best_r * 1.0005
        try_improve(best_pts, r_target, 3000)
    
    # Phase 4: Very final polish
    if time.time() < deadline - 5.0:
        r_target = best_r * 1.0002
        try_improve(best_pts, r_target, 2000)
    
    # Phase 5: Soft-min optimization for smooth gradient-based improvement
    if time.time() < deadline - 3.0:
        try:
            pts_soft, r_soft = _soft_min_optimize(best_pts, deadline - 1.0)
            r_soft = _compute_r(pts_soft)
            if r_soft > best_r:
                best_r = r_soft
                best_pts = pts_soft.copy()
        except Exception:
            pass
    
    # Phase 6: Final SLSQP polish on the best configuration
    if time.time() < deadline - 1.0:
        try:
            r_target = best_r * 1.0001
            pts_final, viol_final = _slsqp_optimize(best_pts, r_target, deadline - 0.5)
            r_final = _compute_r(pts_final)
            if r_final > best_r:
                best_r = r_final
                best_pts = pts_final.copy()
        except Exception:
            pass
    
    return best_pts
>>>>>>> REPLACE
