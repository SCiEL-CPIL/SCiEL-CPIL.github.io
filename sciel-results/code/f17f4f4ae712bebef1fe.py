<<<<<<< SEARCH
    # Phase 3: Try different aspect ratios
    if best_result is not None and time.time() < deadline - 8:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w_orig = best_result[-1]
        h_orig = 2.0 - w_orig
        for w_try in [w_orig * 0.9, w_orig * 1.1, w_orig * 0.85, w_orig * 1.15]:
            if time.time() > deadline - 6:
                break
            if w_try < 0.3 or w_try > 1.7:
                continue
            h_try = 2.0 - w_try
            sx = w_try / w_orig
            sy = h_try / h_orig
            xs = x * sx
            ys = y * sy
            rs = r * min(sx, sy)
            xs, ys, rs = force_relax(xs, ys, rs, w_try, h_try, max_iter=500)
            xs, ys, rs = expansion_polish(xs, ys, rs, w_try, h_try, time_budget=min(3.0, deadline - time.time() - 4))
            if is_valid(xs, ys, rs, w_try, h_try):
                val = float(np.sum(rs))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = xs
                    z[1::3][:N] = ys
                    z[2::3][:N] = rs
                    z[-1] = w_try
                    best_result = z.copy()
=======
    # Phase 3: Try different aspect ratios
    if best_result is not None and time.time() < deadline - 8:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w_orig = best_result[-1]
        h_orig = 2.0 - w_orig
        for w_try in [w_orig * 0.9, w_orig * 1.1, w_orig * 0.85, w_orig * 1.15, w_orig * 0.95, w_orig * 1.05]:
            if time.time() > deadline - 5:
                break
            if w_try < 0.3 or w_try > 1.7:
                continue
            h_try = 2.0 - w_try
            sx = w_try / w_orig
            sy = h_try / h_orig
            xs = x * sx
            ys = y * sy
            rs = r * min(sx, sy)
            xs, ys, rs = force_relax(xs, ys, rs, w_try, h_try, max_iter=800)
            xs, ys, rs = expansion_polish(xs, ys, rs, w_try, h_try, time_budget=min(3.0, deadline - time.time() - 4))
            if is_valid(xs, ys, rs, w_try, h_try):
                val = float(np.sum(rs))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = xs
                    z[1::3][:N] = ys
                    z[2::3][:N] = rs
                    z[-1] = w_try
                    best_result = z.copy()
>>>>>>> REPLACE
