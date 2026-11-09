                for c_idx in var_to_clauses[var_idx]:
                    c = clauses[c_idx]
                    # Check if c is satisfied by exactly one literal, and that literal is var_idx
                    # We need to check all 3 literals in c
                    sat_count = 0
                    sole_sat_var = None
                    for l in c:
                        if assignment[abs(l)-1] == (l > 0):
                            sat_count += 1
                            sole_sat_var = abs(l) - 1
                            if sat_count > 1:
                                break
                    if sat_count == 1 and sole_sat_var == var_idx:
                        damage += 1
