    best_unsat = sum(1 for ci in range(len(clause_vars_list)) 
                     if not any(phase_assign[v] == p for v, p in zip(clause_vars_list[ci], clause_polarity_list[ci])))
