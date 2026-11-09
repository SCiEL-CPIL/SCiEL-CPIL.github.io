                    # ... inside color loop ...
                    old_c = col[u][v]
                    col[u][v] = c
                    col[v][u] = c
                    
                    valid = False
                    new_bad_affected = set()
                    
                    for ac_id in affected_cycles:
                        ac_edges = edges_in_cycle[ac_id]
                        present, is_bad = _check_cycle_present_and_rainbow(adj, col, ac_edges)
                        if is_bad:
                            new_bad_affected.add(ac_id)
                    
                    # Calculate actual new bad set
                    # Remove affected from old, add new_bad_affected
                    # But we must be careful: some affected cycles might have been GOOD before and are now BAD?
                    # No, if they were good before, they are not in bad_cycles.
                    # If they are now bad, they are in new_bad_affected.
                    # So:
                    remaining_old_bad = bad_cycles - set(affected_cycles)
                    actual_new_bad = remaining_old_bad | new_bad_affected
                    
                    if len(actual_new_bad) < len(bad_cycles):
                        valid = True
                    elif len(actual_new_bad) == len(bad_cycles):
                        # Accept if the specific cycle we are trying to fix is fixed
                        if cid not in actual_new_bad:
                            valid = True
                    
                    if valid:
                        bad_cycles = actual_new_bad
                        fixed = True
                        break
                    else:
                        # Revert
                        col[u][v] = old_c
                        col[v][u] = old_c
