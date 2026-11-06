# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [True] * n
    
    # Precompute clause data
    pos_mask = [0] * n
    neg_mask = [0] * n
    var_to_clauses = [[] for _ in range(n)]
    
    for c, clause in enumerate(clauses):
        bit = 1 << c
        for lit in clause:
            var = abs(lit) - 1
            if lit > 0:
                pos_mask[var] |= bit
            else:
                neg_mask[var] |= bit
            var_to_clauses[var].append(c)

    def count_bits(mask):
        return mask.bit_count()

    rng = random.Random(42)
    
    # Multiple restarts to escape local optima
    max_restarts = 50
    max_steps_per_restart = 100000
    
    all_mask = (1 << m) - 1
    
    for restart in range(max_restarts):
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize sat_mask and unsat_list
        sat_mask = 0
        unsat_list = []
        
        # Determine initial state of each clause
        for c in range(m):
            is_sat = False
            for lit in clauses[c]:
                v = abs(lit) - 1
                if lit > 0:
                    if assignment[v]:
                        is_sat = True
                        break
                else:
                    if not assignment[v]:
                        is_sat = True
                        break
            if is_sat:
                sat_mask |= (1 << c)
            else:
                unsat_list.append(c)
                
        if not unsat_list:
            return assignment
        
        for step in range(max_steps_per_restart):
            if not unsat_list:
                return assignment
            
            # Pick a random unsatisfied clause
            c_idx = rng.choice(unsat_list)
            clause = clauses[c_idx]
            
            best_gain = -float('inf')
            best_vars = []
            
            for lit in clause:
                var = abs(lit) - 1
                
                # Calculate net gain of flipping var
                old_val = assignment[var]
                
                if old_val:
                    old_mask = pos_mask[var]
                    new_mask = neg_mask[var]
                else:
                    old_mask = neg_mask[var]
                    new_mask = pos_mask[var]
                
                # Gain: clauses satisfied by new value that are currently unsat
                gain_mask = new_mask & ~sat_mask
                gain = count_bits(gain_mask)
                
                # Loss: clauses satisfied by old value that are currently sat 
                # and will become unsat
                loss_candidates_mask = old_mask & sat_mask
                loss = 0
                
                # Iterate through loss candidates
                temp_mask = loss_candidates_mask
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    c_idx2 = lsb.bit_length() - 1
                    temp_mask ^= lsb
                    
                    # Check if clause c_idx2 is satisfied by any other literal
                    is_only_satisfied = True
                    for l2 in clauses[c_idx2]:
                        v2 = abs(l2) - 1
                        if v2 == var:
                            continue
                        if l2 > 0:
                            if assignment[v2]:
                                is_only_satisfied = False
                                break
                        else:
                            if not assignment[v2]:
                                is_only_satisfied = False
                                break
                    
                    if is_only_satisfied:
                        loss += 1
                
                net_gain = gain - loss
                
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_vars = [var]
                elif net_gain == best_gain:
                    best_vars.append(var)
            
            if not best_vars:
                var = rng.choice([abs(l) - 1 for l in clause])
            else:
                var = rng.choice(best_vars)
            
            # Apply flip
            assignment[var] = not assignment[var]
            
            # Update sat_mask and unsat_list incrementally
            affected_clauses = var_to_clauses[var]
            
            for c in affected_clauses:
                # Determine new status
                is_sat = False
                for l in clauses[c]:
                    v = abs(l) - 1
                    if l > 0:
                        if assignment[v]:
                            is_sat = True
                            break
                    else:
                        if not assignment[v]:
                            is_sat = True
                            break
                
                if is_sat:
                    if not (sat_mask & (1 << c)):
                        # Was unsat, now sat
                        try:
                            unsat_list.remove(c)
                        except ValueError:
                            pass
                        sat_mask |= (1 << c)
                else:
                    if sat_mask & (1 << c):
                        # Was sat, now unsat
                        unsat_list.append(c)
                        sat_mask &= ~(1 << c)
            
        # If we finish steps without solving, we restart (loop continues)
    
    # If all restarts fail, return last assignment
    return assignment
# EVOLVE-BLOCK-END
