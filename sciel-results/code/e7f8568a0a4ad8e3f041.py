    # Strategy 4: Polish with swaps
    if best_pts_global is not None and time.time() < deadline - 10:
        polished_pts, polished_lines, polished_I = polish_solution(best_pts_global, best_lines_global, deadline - 10)
        if polished_I > best_I_global:
            best_I_global = polished_I
            best_lines_global = polished_lines
            best_pts_global = polished_pts
