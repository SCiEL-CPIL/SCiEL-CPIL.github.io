    # Polish phase
    if time.time() < deadline - 2.0:
        c_polish = best_c[:]
        score_polish = best_score
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for i in range(12):
                old_val = c_polish[i]
                # Try all valid values
                for v in range(-64, 65):
                    if v == 0: continue
                    if v == old_val: continue
                    c_polish[i] = v
                    s = get_score(c_polish)
                    if s < score_polish:
                        score_polish = s
                        best_score = s
                        best_c = c_polish[:]
                        changed = True
                        break # Restart loop for this index? Or just update and continue
                    # If not better, revert
                    c_polish[i] = old_val
                # After trying all, c_polish[i] is either the new best value or reverted to old_val
                # Actually, if we break, c_polish[i] is v.
                # If we didn't break, c_polish[i] is old_val.
