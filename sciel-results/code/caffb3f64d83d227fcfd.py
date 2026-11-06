"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import heapq
import math


def solve(instance):
    # Model-and-optimize style SAT solver:
    # 1) Deterministic preprocessing (unit propagation, pure literal elimination)
    # 2) WalkSAT-style local search with incremental clause tracking and smooth scoring
    # 3) Time-bounded multi-restart with increasingly aggressive parameters
    n, clauses = instance["n"], instance["clauses"]
    
    deadline = time.time() + 290  # safety margin before 330s kill
    
    # Build clause list as list of tuples of literals (signed 1-based)
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
    
    # Build variable-to-clause index and clause literal positions
    var_to_clauses = [[] for _ in range(n)]
    clause_lit_pos = []  # for each clause, list of (var_idx, sign)
    for ci, c in enumerate(cnf):
        positions = []
        for lit in c:
            vi = abs(lit) - 1
            var_to_clauses[vi].append(ci)
            positions.append((vi, 1 if lit > 0 else -1))
        clause_lit_pos.append(positions)
    
    # Unit propagation and pure literal elimination (one pass)
    def preprocess(cnf_list):
        """Apply unit propagation and pure literal elimination iteratively."""
        assignment = [None] * n  # None = unassigned
        cnf_work = [list(c) for c in cnf_list]
        changed = True
        while changed:
            changed = False
            # Remove satisfied clauses
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
            
            # Unit propagation
            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assignment[vi] is None:
                        assignment[vi] = (lit > 0)
                        changed = True
                    elif assignment[vi] != (lit > 0):
                        return None, None, assignment  # contradiction
                    else:
                        cnf_work[i] = []  # satisfied
                        changed = True
                elif len(c) == 0:
                    return None, None, assignment  # contradiction
            
            # Pure literal elimination
            lit_counts = [[0, 0] for _ in range(n)]  # [pos_count, neg_count]
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
                    assignment[vi] = False  # pure negative
                    changed = True
                elif lit_counts[vi][1] == 0 and lit_counts[vi][0] > 0:
                    assignment[vi] = True  # pure positive
                    changed = True
        
        # Check if all assigned
        if all(a is not None for a in assignment):
            final_assign = [bool(a) for a in assignment]
            if is_satisfied(final_assign):
                return final_assign, None, final_assign
            return None, None, assignment
        
        return None, cnf_work, assignment
    
    # Try preprocessing
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
        """For each clause, count how many literals are satisfied (0-3)."""
        sat_counts = [0] * m
        for ci in range(m):
            sc = 0
            for vi, sign in clause_lit_pos[ci]:
                if assignment[vi] == (sign > 0):
                    sc += 1
            sat_counts[ci] = sc
        return sat_counts
    
    def smooth_clause_score(sat_count, clause_len):
        """
        Smooth surrogate for the min objective: instead of hard 0/1,
        use a soft penalty that decreases smoothly as more literals are satisfied.
        score = exp(-alpha * sat_count / clause_len)
        This gives higher weight to clauses that are "closer" to being satisfied.
        """
        alpha = 3.0
        return math.exp(-alpha * sat_count / clause_len)
    
    def walksat(assign, max_steps, p_bias, seed):
        """Run WalkSAT with incremental clause tracking. Returns (success, final_assign, min_unsat)."""
        if time.time() > deadline:
            return False, assign, count_unsat(assign)
        
        rng = random.Random(seed)
        cur = assign[:]
        
        # Initialize clause satisfaction counts
        sat_counts = compute_clause_sat_counts(cur)
        cur_unsat = sum(1 for sc in sat_counts if sc == 0)
        min_unsat = cur_unsat
        min_assign = cur[:]
        
        # Smooth clause weights: higher for clauses closer to satisfied
        clause_weights = [smooth_clause_score(sc, len(clause_lit_pos[ci])) for ci, sc in enumerate(sat_counts)]
        
        # Track how many times each clause has been selected while unsatisfied
        clause_select_count = [0] * m
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0
            
            # Find unsatisfied clauses
            unsat_clauses = [ci for ci in range(m) if sat_counts[ci] == 0]
            
            if not unsat_clauses:
                return True, cur, 0
            
            # Update clause weights for unsatisfied clauses using smooth score
            # Also incorporate selection history as a secondary factor
            for uci in unsat_clauses:
                clause_select_count[uci] += 1
                # Combine smooth proximity score with selection frequency
                # Clauses that have been unsatisfied longer get slightly higher weight
                hist_factor = 1.0 + 0.1 * min(clause_select_count[uci], 20)
                clause_weights[uci] = smooth_clause_score(0, len(clause_lit_pos[uci])) * hist_factor
            
            # Select clause: weighted by smooth score
            total_weight = sum(clause_weights[ci] for ci in unsat_clauses)
            if total_weight <= 0:
                ci = rng.choice(unsat_clauses)
            else:
                r = rng.random() * total_weight
                cum = 0
                ci = unsat_clauses[-1]
                for uci in unsat_clauses:
                    cum += clause_weights[uci]
                    if r <= cum:
                        ci = uci
                        break
            
            # Select literal within clause
            clause = cnf[ci]
            weights = []
            for lit in clause:
                vi = abs(lit) - 1
                # Number of clauses this variable is in (degree)
                w = 1.0 / (1 + len(var_to_clauses[vi]))
                # Bias: if the variable is currently set to satisfy this literal, weight less
                if cur[vi] == (lit > 0):
                    w *= 0.3
                weights.append(w)
            
            # Weighted random selection
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            
            literal = clause[chosen_idx]
            vi = abs(literal) - 1
            
            # With probability p_bias, flip the literal that would satisfy the clause
            # Otherwise random flip
            if rng.random() < p_bias:
                cur[vi] = (literal > 0)
            else:
                cur[vi] = not cur[vi]
            
            # Incrementally update sat_counts for affected clauses
            delta_unsat = 0
            for aci in var_to_clauses[vi]:
                old_sc = sat_counts[aci]
                new_sc = 0
                for vi2, sign2 in clause_lit_pos[aci]:
                    if cur[vi2] == (sign2 > 0):
                        new_sc += 1
                sat_counts[aci] = new_sc
                if old_sc == 0 and new_sc > 0:
                    delta_unsat -= 1  # clause became satisfied
                elif old_sc > 0 and new_sc == 0:
                    delta_unsat += 1  # clause became unsatisfied
                # Update smooth weight
                clause_weights[aci] = smooth_clause_score(new_sc, len(clause_lit_pos[aci]))
            
            cur_unsat += delta_unsat
            
            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]
        
        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat
    
    # Multi-restart strategy with increasing effort
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765]
    base_steps = 10000
    p_bias_values = [0.8, 0.75, 0.7, 0.85, 0.6, 0.9, 0.65, 0.78, 0.55, 0.82]
    
    while time.time() < deadline - 5:
        seed = seeds[attempt % len(seeds)]
        p_bias = p_bias_values[attempt % len(p_bias_values)]
        steps = min(base_steps + attempt * 2000, 100000)
        
        success, result, min_unsat = walksat(best_assign, steps, p_bias, seed)
        
        if success:
            return result
        
        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]
        
        attempt += 1
        
        # If we've tried enough with diminishing returns, try a fresh random start
        if attempt % 5 == 0:
            fresh = [random.Random(attempt).random() < 0.5 for _ in range(n)]
            fu = count_unsat(fresh)
            if fu < best_unsat:
                best_assign = fresh
                best_unsat = fu
    
    # Final check
    if is_satisfied(best_assign):
        return best_assign
    
    # Last resort: try a few more random flips
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
