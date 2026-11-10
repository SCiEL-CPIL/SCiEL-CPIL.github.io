            if old_lit_val == False and new_lit_val == True:
                unsat_count[c_idx] -= 1
                if c_idx in unsat_set:
                    if unsat_count[c_idx] < len(clauses[c_idx]):
                        unsat_set.remove(c_idx)
            elif old_lit_val == True and new_lit_val == False:
                unsat_count[c_idx] += 1
                if c_idx not in unsat_set:
                    if unsat_count[c_idx] == len(clauses[c_idx]):
                        unsat_set.add(c_idx)
