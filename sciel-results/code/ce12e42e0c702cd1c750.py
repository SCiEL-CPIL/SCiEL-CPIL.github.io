"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import heapq
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    deadline = time.time() + 290
    
    cnf = [tuple(c) for c in clauses]
    m = len(cnf)
    
    def is_satisfied(assignment):
        for c in cnf:
            sat = False
            for lit in c:
                val = assignment[abs(lit) - 1]
                if val == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        count = 0
        for c in cnf:
            sat = False
            for lit in c:
                val = assignment[abs(lit) - 1]
                if val == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    var_to_clauses = [[] for _ in range(n)]
    clause_lit_pos = []
    for ci, c in enumerate(cnf):
        positions = []
        for lit in c:
            vi = abs(lit) - 1
            var_to_clauses[vi].append(ci)
            positions.append((vi, 1 if lit > 0 else -1))
        clause_lit_pos.append(positions)
    
    def preprocess(cnf_list):
        assignment = [None] * n
        cnf_work = [list(c) for c in cnf_list]
        changed = True
        while changed:
            changed = False
            new_cnf = []
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assignment[vi] is not None and assignment[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_cnf.append(c)
            cnf_work = new_cnf
            
            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assignment[vi] is None:
                        assignment[vi] = (lit > 0)
                        changed = True
                    elif assignment[vi] != (lit > 0):
                        return None, None, assignment
                    else:
                        cnf_work[i] = []
                        changed = True
                elif len(c) == 0:
                    return None, None, assignment
            
            lit_counts = [[0, 0] for _ in range(n)]
            for c in cnf_work:
                seen = set()
                for lit in c:
                    vi = abs(lit) - 1
                    if vi in seen:
                        continue
                    seen.add(vi)
                    if lit > 0:
                        lit_counts[vi][0] += 1
                    else:
                        lit_counts[vi][1] += 1
            
            for vi in range(n):
                if assignment[vi] is not None:
                    continue
                if lit_counts[vi][0] == 0 and lit_counts[vi][1] > 0:
                    assignment[vi] = False
                    changed = True
                elif lit_counts[vi][1] == 0 and lit_counts[vi][0] > 0:
                    assignment[vi] = True
                    changed = True
        
        if all(a is not None for a in assignment):
            final_assign = [bool(a) for a in assignment]
            if is_satisfied(final_assign):
                return final_assign, None, final_assign
            return None, None, assignment
        
        return None, cnf_work, assignment
    
    solved, reduced_cnf, partial_assign = preprocess(cnf)
    if solved is not None:
        return solved
    
    def make_full_assignment(partial):
        rng = random.Random()
        assign = []
        for i in range(n):
            if partial[i] is not None:
                assign.append(bool(partial[i]))
            else:
                assign.append(rng.random() < 0.5)
        return assign
    
    if reduced_cnf is not None:
        cnf = [tuple(c) for c in reduced_cnf if len(c) > 0]
        m = len(cnf)
        var_to_clauses = [[] for _ in range(n)]
        clause_lit_pos = []
        for ci, c in enumerate(cnf):
            positions = []
            for lit in c:
                vi = abs(lit) - 1
                var_to_clauses[vi].append(ci)
                positions.append((vi, 1 if lit > 0 else -1))
            clause_lit_pos.append(positions)
        start_assign = make_full_assignment(partial_assign)
    else:
        start_assign = [random.Random(0).random() < 0.5 for _ in range(n)]
    
    if is_satisfied(start_assign):
        return start_assign
    
    best_assign = start_assign[:]
    best_unsat = count_unsat(start_assign)
    
    def compute_clause_sat_counts(assignment):
        sat_counts = [0] * m
        for ci in range(m):
            sc = 0
            for vi, sign in clause_lit_pos[ci]:
                if assignment[vi] == (sign > 0):
                    sc += 1
            sat_counts[ci] = sc
        return sat_counts
    
    def walksat(assign, max_steps, p_bias, seed):
        """
        Optimized WalkSAT with:
        - Maintained unsat_clause_set for O(1) access to unsatisfied clauses
        - Precomputed clause lengths
        - Weighted clause selection based on frequency of being unsatisfied
        """
        if time.time() > deadline:
            return False, assign, count_unsat(assign)
        
        rng = random.Random(seed)
        cur = assign[:]
        
        # Precompute clause lengths
        clause_lens = [len(clause_lit_pos[ci]) for ci in range(m)]
        
        # Initialize clause satisfaction counts
        sat_counts = compute_clause_sat_counts(cur)
        cur_unsat = sum(1 for sc in sat_counts if sc == 0)
        min_unsat = cur_unsat
        min_assign = cur[:]
        
        # Maintain a list of unsatisfied clause indices
        unsat_list = [ci for ci in range(m) if sat_counts[ci] == 0]
        
        # Track how many times each clause has been selected while unsatisfied
        clause_select_count = [0] * m
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0
            
            if not unsat_list:
                return True, cur, 0
            
            # Select clause: weighted by selection frequency (more frequent = higher weight)
            # This creates a soft-min effect: clauses that persistently fail get more attention
            if len(unsat_list) == 1:
                ci = unsat_list[0]
            else:
                # Build weights based on selection count
                total_w = 0.0
                weights = []
                for uci in unsat_list:
                    w = 1.0 + 0.5 * min(clause_select_count[uci], 50)
                    weights.append(w)
                    total_w += w
                
                r = rng.random() * total_w
                cum = 0.0
                ci = unsat_list[-1]
                for idx, uci in enumerate(unsat_list):
                    cum += weights[idx]
                    if r <= cum:
                        ci = uci
                        break
            
            clause_select_count[ci] += 1
            
            # Select literal within clause
            clause = cnf[ci]
            # Weight literals by inverse degree (less connected variables are "cheaper" to flip)
            weights = []
            for lit in clause:
                vi = abs(lit) - 1
                w = 1.0 / (1.0 + len(var_to_clauses[vi]))
                weights.append(w)
            
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0.0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            
            literal = clause[chosen_idx]
            vi = abs(literal) - 1
            
            # With probability p_bias, set variable to satisfy the clause
            # Otherwise random flip
            if rng.random() < p_bias:
                cur[vi] = (literal > 0)
            else:
                cur[vi] = not cur[vi]
            
            # Incrementally update sat_counts for affected clauses
            delta_unsat = 0
            new_unsat_list = []
            
            for aci in var_to_clauses[vi]:
                old_sc = sat_counts[aci]
                new_sc = 0
                for vi2, sign2 in clause_lit_pos[aci]:
                    if cur[vi2] == (sign2 > 0):
                        new_sc += 1
                sat_counts[aci] = new_sc
                
                if old_sc == 0 and new_sc > 0:
                    delta_unsat -= 1
                elif old_sc > 0 and new_sc == 0:
                    delta_unsat += 1
            
            # Rebuild unsat_list (only for clauses affected by this flip)
            # Optimization: only check clauses that were unsat or are in var_to_clauses[vi]
            for aci in range(m):
                if sat_counts[aci] == 0:
                    new_unsat_list.append(aci)
            
            cur_unsat += delta_unsat
            unsat_list = new_unsat_list
            
            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]
        
        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat
    
    # Multi-restart strategy
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765]
    p_bias_values = [0.8, 0.75, 0.7, 0.85, 0.6, 0.9, 0.65, 0.78, 0.55, 0.82]
    
    while time.time() < deadline - 5:
        seed = seeds[attempt % len(seeds)]
        p_bias = p_bias_values[attempt % len(p_bias_values)]
        steps = min(10000 + attempt * 2000, 100000)
        
        success, result, min_unsat = walksat(best_assign, steps, p_bias, seed)
        
        if success:
            return result
        
        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]
        
        attempt += 1
        
        if attempt % 5 == 0:
            fresh = [random.Random(attempt).random() < 0.5 for _ in range(n)]
            fu = count_unsat(fresh)
            if fu < best_unsat:
                best_assign = fresh
                best_unsat = fu
    
    if is_satisfied(best_assign):
        return best_assign
    
    rng = random.Random(0)
    for _ in range(2000):
        if time.time() > deadline - 2:
            break
        vi = rng.randrange(n)
        best_assign[vi] = not best_assign[vi]
        if is_satisfied(best_assign):
            return best_assign
        u = count_unsat(best_assign)
        if u > best_unsat:
            best_assign[vi] = not best_assign[vi]
    
    return best_assign

# EVOLVE-BLOCK-END
