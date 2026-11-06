"""Synthetic planted 3-SAT solver with improved WalkSAT."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted satisfiable 3-CNF instance using an improved WalkSAT
    with:
      - Precomputed clause structure for speed
      - Noise parameter tuning
      - Restart strategy with increasing flip budgets
      - Best-assignment tracking
      - Deadline guard (safety margin before 330s)
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Precompute: for each clause, store (literal, variable_index, polarity)
    # Also build a reverse index: variable -> list of (clause_idx, literal_idx_in_clause)
    clause_data = []
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        cdata = []
        for lit in clause:
            v = abs(lit) - 1
            pol = 1 if lit > 0 else 0
            cdata.append((lit, v, pol))
            var_to_clauses[v].append((ci, len(cdata) - 1))
        clause_data.append(cdata)
    
    m = len(clauses)
    rng = random.Random(0)
    
    def random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]
    
    def is_satisfied(assignment):
        for cdata in clause_data:
            sat = False
            for lit, v, pol in cdata:
                if assignment[v] == (pol == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_violated(assignment):
        count = 0
        for cdata in clause_data:
            sat = False
            for lit, v, pol in cdata:
                if assignment[v] == (pol == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def flip_cost(assignment, var_idx):
        """Compute the change in number of violated clauses if we flip var_idx."""
        # For each clause containing var_idx, check if flipping changes satisfaction
        delta = 0
        for ci, li in var_to_clauses[var_idx]:
            cdata = clause_data[ci]
            lit, v, pol = cdata[li]
            current_val = assignment[v]
            new_val = not current_val
            
            # Check if clause is currently satisfied by another literal
            # and whether it will still be satisfied after flip
            currently_sat_by_other = False
            after_sat_by_other = False
            for j, (l2, v2, p2) in enumerate(cdata):
                if j == li:
                    continue
                if assignment[v2] == (p2 == 1):
                    currently_sat_by_other = True
                # After flip, v2 unchanged
                if assignment[v2] == (p2 == 1):
                    after_sat_by_other = True
            
            # Currently satisfied?
            currently_sat = currently_sat_by_other or (current_val == (pol == 1))
            # After flip satisfied?
            after_sat = after_sat_by_other or (new_val == (pol == 1))
            
            if not currently_sat and after_sat:
                delta -= 1  # good: unsat becomes sat
            elif currently_sat and not after_sat:
                delta += 1  # bad: sat becomes unsat
        
        return delta
    
    deadline = time.time() + 320  # safety margin
    
    best_assignment = random_assignment()
    best_count = count_violated(best_assignment)
    
    if best_count == 0:
        return best_assignment
    
    # Restart strategy: increasing noise and flip budgets
    restarts = 0
    max_restarts = 200
    base_budget = 5000
    noise = 0.2
    
    while restarts < max_restarts and time.time() < deadline:
        assignment = random_assignment()
        current_count = count_violated(assignment)
        
        if current_count < best_count:
            best_count = current_count
            best_assignment = assignment[:]
        
        if current_count == 0:
            return assignment
        
        # Adaptive noise: decrease slightly over restarts
        eff_noise = max(0.1, noise - restarts * 0.002)
        
        budget = base_budget + restarts * 1000
        
        for step in range(budget):
            if time.time() > deadline:
                break
            
            if current_count == 0:
                return assignment
            
            # Find violated clauses
            violated = []
            for ci, cdata in enumerate(clause_data):
                sat = False
                for lit, v, pol in cdata:
                    if assignment[v] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    violated.append(ci)
            
            if not violated:
                return assignment
            
            # Pick a random violated clause
            ci = rng.choice(violated)
            cdata = clause_data[ci]
            
            # With probability (1 - eff_noise), pick the literal that minimizes flip cost
            # With probability eff_noise, pick randomly
            if rng.random() < eff_noise:
                _, v, _ = rng.choice(cdata)
            else:
                best_v = None
                best_cost = float('inf')
                for lit, v, pol in cdata:
                    cost = flip_cost(assignment, v)
                    if cost < best_cost:
                        best_cost = cost
                        best_v = v
                v = best_v
            
            assignment[v] = not assignment[v]
            
            # Update current_count using flip_cost (we already computed it for the chosen v in the non-noise case,
            # but for simplicity just recount; for 50 vars this is fast)
            current_count = count_violated(assignment)
            
            if current_count < best_count:
                best_count = current_count
                best_assignment = assignment[:]
        
        restarts += 1
        # Slightly increase noise every few restarts to escape local optima
        if restarts % 5 == 0:
            noise = min(0.5, noise + 0.02)
    
    return best_assignment
# EVOLVE-BLOCK-END
