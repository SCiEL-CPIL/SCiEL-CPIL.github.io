        while it < max_iters:
            if time.time() > deadline:
                break

            viol = find_violation_fast(cur, n)
    