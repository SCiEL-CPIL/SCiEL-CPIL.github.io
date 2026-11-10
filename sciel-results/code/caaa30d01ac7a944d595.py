        if cost_3ap == 0:
            valid, viol = _is_valid_full(S)
            if valid:
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = list(S)
                    # ... grow ...
            else:
                # Break the violation
                a, k = viol
                # Remove a from S
                if a in S:
                    S.remove(a)
                # Reset stall counter
                stall_counter = 0
                T = 1.0
