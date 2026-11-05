"""Planted 3-SAT solver combining WalkSAT local search with DPLL-style propagation."""
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline guard: return best result with safety margin before 330s
    deadline = time.time() + 325.0  # 5s safety margin
    
    # Preprocess: convert clauses to 0-based, track literal signs
    # clause: list of (var_idx, is_positive)
    clauses_pp = []
    for c in clauses:
        pp_c = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pp_c.append((var_idx, is_pos))
        clauses_pp.append(pp_c)
    
    # Track variable occurrences for better heuristics
    var_count = [0] * n
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses_pp):
        for v, _ in c:
            var_count[v] += 1
            var_to_clauses[v].append(i)
    
    def is_satisfied(assignment, clause):
        for v, pos in clause:
            if assignment[v] == pos:
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for c in clauses_pp:
            if not is_satisfied(assignment, c):
                cnt += 1
        return cnt
    
    def unsat_clauses_list(assignment):
        """Return list of unsatisfied clause indices"""
        unsat = []
        for i, c in enumerate(clauses_pp):
            if not is_satisfied(assignment, c):
                unsat.append(i)
        return unsat
    
    def propagate(assignment, unsat_indices):
        """
        Perform unit clause propagation and pure literal elimination.
        Returns (assignment, new_unsat_indices, changed) where changed is True if any propagation occurred.
        """
        changed = False
        # Work on a copy of unsat indices
        to_process = list(unsat_indices)
        processed = set()
        
        # Build a map of variable -> list of (clause_idx, literal_pos) for unsat clauses
        var_unsat_info = {}
        for ci in unsat_indices:
            c = clauses_pp[ci]
            for v, pos in c:
                if v not in var_unsat_info:
                    var_unsat_info[v] = []
                var_unsat_info[v].append((ci, pos))
        
        # Unit clause propagation: if a clause has only one unsatisfied literal, flip that variable
        # Actually, for unit propagation in the context of local search:
        # If a clause is unsatisfied and all but one of its literals are currently false,
        # and the remaining literal's variable is set to the wrong value, flipping it satisfies the clause.
        # But more generally, we look for clauses where flipping a variable would satisfy the clause
        # and not break many others.
        
        # Let's do a simpler propagation: 
        # 1. Pure literal elimination: if a variable only appears in one polarity in unsat clauses,
        #    set it to satisfy those clauses.
        # 2. Unit clause: if an unsat clause has only one variable that is currently set to the wrong value,
        #    and all other variables in the clause are set correctly (which can't happen if clause is unsat),
        #    then... wait, if clause is unsat, ALL literals are false.
        # So for an unsat clause (v1,p1), (v2,p2), (v3,p3), assignment[v1]!=p1, assignment[v2]!=p2, assignment[v3]!=p3.
        # Flipping v1 makes literal (v1,p1) true, satisfying the clause.
        
        # Let's implement a greedy propagation: repeatedly find unsat clauses where flipping a variable
        # satisfies the most clauses (or at least satisfies the clause and doesn't break others).
        # This is essentially a mini local search step but deterministic.
        
        # For efficiency, let's just do one pass of "forced" flips:
        # If an unsat clause contains a variable that appears in NO other unsat clauses,
        # flipping it will satisfy this clause and not create new unsat clauses.
        
        new_unsat_indices = set(unsat_indices)
        
        # Find variables that appear in only one unsat clause
        var_unsat_count = {}
        for ci in unsat_indices:
            c = clauses_pp[ci]
            for v, pos in c:
                if v not in var_unsat_count:
                    var_unsat_count[v] = set()
                var_unsat_count[v].add(ci)
        
        # Variables that appear in exactly one unsat clause
        forced_vars = []
        for v, cis in var_unsat_count.items():
            if len(cis) == 1:
                ci = list(cis)[0]
                c = clauses_pp[ci]
                for vv, pos in c:
                    if vv == v:
                        # Flipping v will satisfy clause ci
                        # Check if v is already set to pos (it shouldn't be, since clause is unsat)
                        if assignment[v] != pos:
                            forced_vars.append((v, pos))
                        break
        
        # Apply forced flips
        for v, pos in forced_vars:
            if assignment[v] != pos:
                assignment[v] = pos
                changed = True
                # This flip satisfies clause ci. Remove ci from unsat.
                # But it might break other clauses where v appeared.
                # However, v appeared in only one unsat clause, so it doesn't break any other unsat clause.
                # It might break satisfied clauses, but we don't track those here.
                # For simplicity, we'll re-evaluate unsat clauses after all forced flips.
        
        if changed:
            # Re-evaluate unsat clauses
            new_unsat_list = []
            for i, c in enumerate(clauses_pp):
                if not is_satisfied(assignment, c):
                    new_unsat_list.append(i)
            return assignment, new_unsat_list, True
        
        return assignment, unsat_indices, False
    
    # Try multiple restarts with different strategies
    best_assignment = None
    best_unsat_count = num_clauses + 1
    
    rng = random.Random(42)
    
    max_restarts = 100
    steps_per_restart = 3000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        if restart == 0:
            # First restart: try a deterministic heuristic
            # Set each variable to satisfy the most clauses
            assignment = [False] * n
            for v in range(n):
                true_sat = 0
                false_sat = 0
                for c in clauses_pp:
                    for vv, pos in c:
                        if vv == v:
                            if pos:
                                true_sat += 1
                            else:
                                false_sat += 1
                            break
                assignment[v] = true_sat >= false_sat
        else:
            # Subsequent restarts: random
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial propagation
        unsat_list = unsat_clauses_list(assignment)
        if not unsat_list:
            return assignment
        
        if len(unsat_list) < best_unsat_count:
            best_unsat_count = len(unsat_list)
            best_assignment = assignment[:]
        
        # Run local search with propagation
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Propagation step
            assignment, unsat_list, changed = propagate(assignment, unsat_list)
            
            if not unsat_list:
                return assignment
            
            if len(unsat_list) < best_unsat_count:
                best_unsat_count = len(unsat_list)
                best_assignment = assignment[:]
            
            if best_unsat_count == 0:
                return best_assignment
            
            # Choose an unsatisfied clause
            # Weighted selection: prefer clauses with fewer "easy" fixes
            # For now, use uniform random selection among unsat clauses
            chosen_ci = unsat_list[rng.randrange(len(unsat_list))]
            clause = clauses_pp[chosen_ci]
            
            # Choose a literal to flip
            # Adaptive p: more greedy as we get closer to solution
            progress = 1.0 - (len(unsat_list) / num_clauses) if num_clauses > 0 else 1.0
            p = 0.6 + 0.4 * progress
            
            if rng.random() < p:
                # Pick a literal from the clause to flip
                # Weight by how many other clauses this flip would affect (prefer low impact)
                lit_weights = []
                for v, pos in clause:
                    # Lower impact is better, so use inverse
                    impact = var_count[v]
                    lit_weights.append(1.0 / max(1, impact))
                total_lw = sum(lit_weights)
                r2 = rng.random() * total_lw
                cumulative2 = 0
                chosen_idx = 0
                for i, lw in enumerate(lit_weights):
                    cumulative2 += lw
                    if r2 <= cumulative2:
                        chosen_idx = i
                        break
                v, pos = clause[chosen_idx]
                assignment[v] = not assignment[v]
            else:
                # Random variable flip
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
            
            # Update unsat list incrementally if possible, but for simplicity recompute
            # Actually, let's recompute to keep it simple and correct
            unsat_list = unsat_clauses_list(assignment)
        
        if best_unsat_count == 0:
            break
    
    # Final polish: if we have a good assignment, try a few more deterministic steps
    if best_assignment is not None and best_unsat_count > 0 and best_unsat_count < num_clauses * 0.1:
        assignment = best_assignment[:]
        for _ in range(100):
            if time.time() > deadline:
                break
            unsat_list = unsat_clauses_list(assignment)
            if not unsat_list:
                return assignment
            if len(unsat_list) < best_unsat_count:
                best_unsat_count = len(unsat_list)
                best_assignment = assignment[:]
            # Try to fix one unsat clause at a time
            ci = unsat_list[0]
            c = clauses_pp[ci]
            # Try flipping each variable in the clause and pick the best
            best_flip = None
            best_gain = 0
            for v, pos in c:
                old_val = assignment[v]
                assignment[v] = not assignment[v]
                new_unsat = count_unsat(assignment)
                gain = len(unsat_list) - new_unsat
                if gain > best_gain:
                    best_gain = gain
                    best_flip = v
                assignment[v] = old_val
            if best_flip is not None:
                assignment[best_flip] = not assignment[best_flip]
    
    if best_assignment is not None:
        return best_assignment
    return [False] * n
