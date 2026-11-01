<<<<<<< SEARCH
    # 4. Final high-precision polish on the best solution found
    # This is where the exact objective + SLSQP/L-BFGS-B shines.
    if best_c is not None and time.time() < deadline - 5:
        remaining = max(5.0, deadline - time.time() - 5)
        c_final, r_final = _exact_polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
=======
    # 4. ADMM-style refinement to squeeze out more precision
    if best_c is not None and time.time() < deadline - 8:
        remaining = max(5.0, deadline - time.time() - 8)
        c_admm, r_admm = _admm_refine(best_c, best_radius, max_iters=100, max_time=min(8.0, remaining))
        if r_admm > best_radius:
            best_radius = r_admm
            best_c = c_admm
    
    # 5. KKT-based polish for final precision
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(3.0, deadline - time.time() - 3)
        c_kkt, r_kkt = _kkt_polish(best_c, max_time=min(5.0, remaining))
        if r_kkt > best_radius:
            best_radius = r_kkt
            best_c = c_kkt
>>>>>>> REPLACE
