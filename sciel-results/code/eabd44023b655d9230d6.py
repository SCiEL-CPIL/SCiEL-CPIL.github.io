            for var in pure_vars:
                if assignment[var] is None:
                    # Assign to satisfy the pure literal
                    # If var is in pos_vars, assign True. If in neg_vars, assign False.
                    if var in pos_vars:
                        assignment[var] = True
                    else:
                        assignment[var] = False
                    changed = True
