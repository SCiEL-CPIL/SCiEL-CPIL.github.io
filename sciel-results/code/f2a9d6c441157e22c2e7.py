"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import heapq


def solve(instance):
    # Model-and-optimize style SAT solver:
    # 1) Deterministic preprocessing (unit propagation, pure literal elimination)
    # 2) WalkSAT-style local search with incremental clause tracking and smooth scoring
    # 3) Time-bounded multi-restart with adaptive parameters
    n, clauses = instance["n"], instance["clauses"]
    
    deadline = time.time() + 290  # safety margin before 330s kill
    
    # Build clause list as list of tuples of literals (signed 1-based)
    cnf = [tuple(c) for c in clauses]
    m = len(cnf)
    
    # Build variable-to-clause index for faster updates
    var_to_clauses = [[] for _ in range(n)]
    clause_lit_pos = []  # for each clause, list of (var_idx, sign)
    for ci, c in enumerate(cnf):
        positions = []
        for lit in c:
            vi = abs(lit) - 1
            var_to_clauses[vi].append(ci)
            positions.append((vi, 1 if lit > 0 else -1))
        clause_lit_pos.append(positions)
    
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
    
    def walksat(assign, max_steps, p_bias, seed):
        """Run WalkSAT with incremental clause tracking and smooth scoring."""
        if time.time() > deadline:
            return False, assign, count_unsat(assign)
        
        rng = random.Random(seed)
        cur = assign[:]
        
        # Incremental clause satisfaction tracking
        # clause_sat_count[ci] = number of literals in clause ci that are satisfied
        clause_sat_count = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_lit_pos[ci]:
                if cur[vi] == (sign > 0):
                    cnt += 1
            clause_sat_count[ci] = cnt
            if cnt > 0:
                clause_sat_count[ci] = 1  # cap at 1 for boolean satisfaction
        
        # More precise: track actual count for smooth scoring
        clause_true_count = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_lit_pos[ci]:
                if cur[vi] == (sign > 0):
                    cnt += 1
            clause_true_count[ci] = cnt
        
        cur_unsat = sum(1 for c in clause_true_count if c == 0)
        min_unsat = cur_unsat
        min_assign = cur[:]
        
        # Smooth clause score: use soft-min via log-sum-exp-like approach
        # Score = exp(-alpha * true_count) for unsatisfied clauses, 0 for satisfied
        # This gives a smooth surrogate for the hard unsat count
        alpha = 2.0
        import math
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0
            
            # Find unsatisfied clauses and compute smooth scores
            unsat_clauses = []
            smooth_scores = []
            for ci in range(m):
                if clause_true_count[ci] == 0:
                    unsat_clauses.append(ci)
                    # Smooth score: higher means more "urgent"
                    # Use a soft penalty that increases as we get closer to satisfaction
                    smooth_scores.append(math.exp(alpha))  # exp(2.0) for fully unsat
                elif clause_true_count[ci] == 1:
                    # Partially satisfied - still track for future
                    pass
            
            if not unsat_clauses:
                return True, cur, 0
            
            # Select clause: weighted by smooth score (all unsat clauses get same base weight)
            # Use recency bias: prefer clauses that have been unsatisfied longer
            # Approximate with random selection among unsat clauses
            ci = rng.choice(unsat_clauses)
            
            # Select literal within clause using smooth variable scoring
            clause = cnf[ci]
            weights = []
            for lit in clause:
                vi = abs(lit) - 1
                # Variable importance: fewer clauses = less impact
                var_degree = len(var_to_clauses[vi])
                w = 1.0 / (1 + var_degree)
                
                # Smooth clause impact: for each clause this var is in,
                # estimate how much flipping helps/hurts using smooth counts
                # If flipping makes this literal true, check impact on other clauses
                if cur[vi] == (lit > 0):
                    # Already satisfies this literal; flipping would break it
                    w *= 0.2
                else:
                    # Flipping would satisfy this literal
                    # Check how many other clauses would be affected
                    help_count = 0
                    hurt_count = 0
                    for aci in var_to_clauses[vi]:
                        if aci == ci:
                            continue
                        # Check if flipping vi helps or hurts clause aci
                        for vi2, sign2 in clause_lit_pos[aci]:
                            if vi2 == vi:
                                if sign2 > 0:
                                    # Positive literal: flipping to True helps if currently False
                                    if not cur[vi]:
                                        help_count += 1
                                    else:
                                        hurt_count += 1
                                else:
                                    # Negative literal: flipping to False helps if currently True
                                    if cur[vi]:
                                        help_count += 1
                                    else:
                                        hurt_count += 1
                                break
                    
                    # Smooth adjustment: favor flips that help more than they hurt
                    net_help = help_count - hurt_count
                    w *= (1.0 + 0.3 * net_help)
                    if w < 0.1:
                        w = 0.1
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
            
            # With probability p_bias, flip to satisfy the clause
            # Otherwise, random flip
            if rng.random() < p_bias:
                cur[vi] = (literal > 0)
            else:
                cur[vi] = not cur[vi]
            
            # Incremental update of clause satisfaction counts
            old_val = not cur[vi]  # value before flip (we just flipped)
            # Actually, we need the old value. Let's track it properly.
            # Re-derive: we just set cur[vi], so old is the opposite
            # But we need to know what the old value was to update counts correctly
            
            # Simpler: just recompute affected clauses
            for aci in var_to_clauses[vi]:
                old_count = clause_true_count[aci]
                new_count = 0
                for vi2, sign2 in clause_lit_pos[aci]:
                    if cur[vi2] == (sign2 > 0):
                        new_count += 1
                clause_true_count[aci] = new_count
                # Update unsat count
                if old_count == 0 and new_count > 0:
                    cur_unsat -= 1
                elif old_count > 0 and new_count == 0:
                    cur_unsat += 1
            
            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]
        
        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat
    
    # Multi-restart strategy with increasing effort
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765]
    base_steps = 5000
    p_bias_values = [0.8, 0.75, 0.7, 0.85, 0.6, 0.9, 0.65, 0.78, 0.55, 0.82]
    
    while time.time() < deadline - 5:
        seed = seeds[attempt % len(seeds)]
        p_bias = p_bias_values[attempt % len(p_bias_values)]
        steps = min(base_steps + attempt * 1000, 50000)
        
        success, result, min_unsat = walksat(best_assign, steps, p_bias, seed)
        
        if success:
            return result
        
        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]
        
        attempt += 1
        
        # Every 5 attempts, try a fresh random start
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
    for _ in range(1000):
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
