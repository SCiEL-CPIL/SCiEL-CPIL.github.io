"""Optimized 3-SAT solver with sat_count tracking and efficient min-confl WalkSAT."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 285.0  # safety margin below 330s kill
    num_clauses = len(clauses)

    # Build variable-to-clauses index
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)

    # Precompute clause info: (var_idx, target_val)
    clause_info = []
    for clause in clauses:
        info = []
        for lit in clause:
            var_idx = abs(lit) - 1
            target_val = (lit > 0)
            info.append((var_idx, target_val))
        clause_info.append(info)

    def initialize_sat_counts(assignment):
        """Compute initial sat_counts for all clauses."""
        sat_counts = [0] * num_clauses
        for ci in range(num_clauses):
            count = 0
            for var_idx, target_val in clause_info[ci]:
                if assignment[var_idx] == target_val:
                    count += 1
            sat_counts[ci] = count
        return sat_counts

    def compute_unsat_list(sat_counts):
        return [ci for ci in range(num_clauses) if sat_counts[ci] == 0]

    def update_sat_counts_and_unsat(var_idx, old_val, new_val, sat_counts, unsat_set):
        """
        Updates sat_counts and unsat_set after flipping var_idx.
        Returns new unsat_list.
        """
        for ci in var_to_clauses[var_idx]:
            # Find the literal for var_idx in clause ci
            # We need to know if old_val satisfied it and if new_val satisfies it
            # clause_info[ci] is list of (v, t)
            lit_satisfied_old = False
            lit_satisfied_new = False
            for v_idx, t_val in clause_info[ci]:
                if v_idx == var_idx:
                    if old_val == t_val:
                        lit_satisfied_old = True
                    if new_val == t_val:
                        lit_satisfied_new = True
                    break
            
            old_sat_count = sat_counts[ci]
            if lit_satisfied_old:
                old_sat_count -= 1
            if lit_satisfied_new:
                old_sat_count += 1
            
            sat_counts[ci] = old_sat_count
            
            was_unsat = ci in unsat_set
            is_unsat = old_sat_count == 0
            
            if was_unsat and not is_unsat:
                unsat_set.discard(ci)
            elif not was_unsat and is_unsat:
                unsat_set.add(ci)
        
        return list(unsat_set)

    def make_random_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def walksat_step(assignment, sat_counts, unsat_list, rng, p_walk):
        """
        Perform one WalkSAT step.
        Returns (new_unsat_list, flipped_var, old_val).
        """
        if not unsat_list:
            return unsat_list, -1, None

        ci = rng.choice(unsat_list)
        clause = clause_info[ci]

        best_var = None
        best_score = float('inf')
        best_lit_val = None

        # Min-conflicts: count satisfied clauses that would become unsat
        for var_idx, target_val in clause:
            conflict = 0
            for other_ci in var_to_clauses[var_idx]:
                if other_ci == ci:
                    continue
                
                # Check if other_ci is currently satisfied
                if sat_counts[other_ci] == 0:
                    continue
                
                # Check if flipping var_idx would break other_ci
                # This happens if:
                # 1. The literal for var_idx in other_ci is currently satisfied
                # 2. No other literal in other_ci is satisfied (sat_count == 1)
                
                if sat_counts[other_ci] == 1:
                    # Check if var_idx's literal is the one satisfying it
                    for v_idx, t_val in clause_info[other_ci]:
                        if v_idx == var_idx:
                            if assignment[var_idx] == t_val:
                                conflict += 1
                            break

            score = conflict
            if score < best_score:
                best_score = score
                best_var = var_idx
                best_lit_val = target_val

        if best_var is None:
            return unsat_list, -1, None

        # WalkSAT randomization
        if rng.random() < p_walk:
            var_idx, target_val = rng.choice(clause)
            best_var = var_idx
            best_lit_val = target_val

        old_val = assignment[best_var]
        new_val = not old_val
        assignment[best_var] = new_val
        
        unsat_set = set(unsat_list)
        new_unsat_list = update_sat_counts_and_unsat(best_var, old_val, new_val, sat_counts, unsat_set)

        return new_unsat_list, best_var, old_val

    best_assignment = None
    best_unsat = num_clauses + 1

    # Try multiple random restarts
    # Increased restart count potential due to speedup
    for attempt in range(20000): 
        if time.time() > deadline:
            break

        assignment = make_random_assignment(attempt * 7919 + 13)
        sat_counts = initialize_sat_counts(assignment)
        unsat_list = compute_unsat_list(sat_counts)
        cur_unsat = len(unsat_list)

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        rng = random.Random(attempt * 31337 + 42)
        steps = 0
        max_steps = 20000  # Increased max steps

        p_walk = 0.05
        stuck_counter = 0
        last_improvement_step = 0

        while cur_unsat > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1

            new_unsat_list, flipped_var, old_val = walksat_step(assignment, sat_counts, unsat_list, rng, p_walk)
            cur_unsat = len(new_unsat_list)
            unsat_list = new_unsat_list

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
                last_improvement_step = steps
            if cur_unsat == 0:
                return assignment

            # Adaptive parameters
            if steps - last_improvement_step > 200:
                stuck_counter += 1
                if stuck_counter > 3:
                    p_walk = min(0.5, p_walk * 1.5)
                    stuck_counter = 0
            else:
                stuck_counter = 0
                if cur_unsat < best_unsat:
                    p_walk = max(0.02, p_walk * 0.85)

            # Polish if close
            if cur_unsat <= 3 and steps > 5000:
                # Simple polish: just continue with lower p_walk
                p_walk = 0.01

    if best_assignment is not None:
        return best_assignment

    return make_random_assignment(0)
# EVOLVE-BLOCK-END
