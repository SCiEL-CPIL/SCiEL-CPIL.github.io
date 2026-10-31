<<<<<<< SEARCH
    # ---------- Phase 8: Random-direction local search ----------
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _local_search_random(best_points, deadline, n_iter=100,
                                           sigma_start=0.015, sigma_end=0.0005)
            _update(pts)
        except Exception:
            pass
=======
    # ---------- Phase 8: Random-direction local search ----------
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _local_search_random(best_points, deadline, n_iter=150,
                                           sigma_start=0.02, sigma_end=0.0003)
            _update(pts)
        except Exception:
            pass
>>>>>>> REPLACE

