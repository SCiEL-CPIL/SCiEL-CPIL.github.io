import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Precompute clause structure for fast access
    # clause_data[i] = list of (var_idx, is_positive)
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            var = abs(l) - 1
            pos = l > 0
            cd.append((var, pos))
        clause_data.append(cd)
    
    # Build variable-to-clause index for fast scoring
    # var_clauses[var] = list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci, cd in enumerate(clause_data):
        for var, pos in cd:
            var_clauses[var].append((ci, pos))
    
    deadline = time.time() + 285  # Safety margin from 330s limit
    rng = random.Random(42)
    
    def is_clause_sat(clause_idx, assignment):
        for var, pos in clause_data[clause_idx]:
            if assignment[var] == pos:
                return True
        return False
    
    def count_satisfied(assignment):
        count = 0
        for i in range(num_clauses):
            if is_clause_sat(i, assignment):
                count += 1
        return count
    
    def score_literal(var, pos, assignment):
        """Score: net gain in satisfied clauses if we set variable to pos."""
        gain = 0
        lose = 0
        for ci, polarity in var_clauses[var]:
            # Check if clause is currently satisfied
            cur_sat = False
            for v, p in clause_data[ci]:
                if assignment[v] == p:
                    cur_sat = True
                    break
            
            # After setting var=pos, is this clause satisfied?
            if polarity == pos:
                new_sat = True
            else:
                new_sat = False
                for v, p in clause_data[ci]:
                    if v == var:
                        continue
                    if assignment[v] == p:
                        new_sat = True
                        break
            
            if new_sat and not cur_sat:
                gain += 1
            elif not new_sat and cur_sat:
                lose += 1
        return gain - lose
    
    def walk_sat_trial(assignment, max_iters, deadline):
        """Run WalkSAT with score-based literal selection and adaptive randomization."""
        cur_count = count_satisfied(assignment)
        
        if cur_count == num_clauses:
            return assignment, cur_count
        
        no_improve = 0
        restart_threshold = 1500
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_indices = []
            for i in range(num_clauses):
                if not is_clause_sat(i, assignment):
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return assignment, num_clauses
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_indices)
            clause = clause_data[clause_idx]
            
            # Score each literal in the clause
            scores = []
            for var, pos in clause:
                s = score_literal(var, pos, assignment)
                scores.append(s)
            
            # Adaptive selection: prefer higher scores but allow randomness
            max_score = max(scores)
            min_score = min(scores)
            spread = max_score - min_score
            
            if spread == 0:
                # All scores equal, pick randomly
                chosen_idx = rng.randrange(len(clause))
            else:
                # Soft selection with temperature that decreases with no_improve
                temperature = max(0.1, 1.0 - no_improve / 2000.0)
                # Weight = exp(score / temperature)
                weights = []
                max_w = max(s / temperature for s in scores)
                for s in scores:
                    w = 1.0 if s == max_score * temperature else (s / temperature - max_w + 1.0)
                    weights.append(max(0.01, w))
                
                total_weight = sum(weights)
                r = rng.random() * total_weight
                cumulative = 0
                chosen_idx = 0
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_idx = i
                        break
            
            var, pos = clause[chosen_idx]
            assignment[var] = pos
            
            new_count = count_satisfied(assignment)
            if new_count > cur_count:
                cur_count = new_count
                no_improve = 0
                if cur_count == num_clauses:
                    return assignment, cur_count
            else:
                no_improve += 1
            
            if no_improve >= restart_threshold:
                break
        
        return assignment, cur_count
    
    best_assignment = [False] * n
    best_count = -1
    
    max_restarts = 80
    iters_per_restart = 3000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        assignment, cur_count = walk_sat_trial(assignment, iters_per_restart, deadline)
        
        if cur_count > best_count:
            best_count = cur_count
            best_assignment = assignment[:]
        
        if best_count == num_clauses:
            return best_assignment
    
    return best_assignment
