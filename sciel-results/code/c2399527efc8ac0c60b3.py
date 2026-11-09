import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin
    
    # Pre-process: variable occurrences
    # var_occurrences[v] = list of (clause_idx, literal)
    var_occurrences = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_occurrences[v].append((i, l))
    
    # Helper: compute initial clause_sat_count and unsat_clauses set
    def init_state(assignment):
        counts = [0] * len(clauses)
        unsat_set = set()
        for i, c in enumerate(clauses):
            count = 0
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    count += 1
            counts[i] = count
            if count == 0:
                unsat_set.add(i)
        return counts, unsat_set

    # Helper: check if satisfied
    def is_satisfied(unsat_set):
        return len(unsat_set) == 0

    # Helper: apply flip and update state
    # Returns nothing, modifies assignment, counts, unsat_set
    def apply_flip(v, assignment, counts, unsat_set):
        assignment[v] = not assignment[v]
        for ci, lit in var_occurrences[v]:
            # Determine if the literal is satisfied AFTER the flip
            # Since we just flipped v, we can check the new state
            v_val = assignment[v]
            if (lit > 0 and v_val) or (lit < 0 and not v_val):
                # Literal is now satisfied
                old_count = counts[ci]
                new_count = old_count + 1
                counts[ci] = new_count
                if old_count == 0:
                    unsat_set.discard(ci)
            else:
                # Literal is now unsatisfied
                old_count = counts[ci]
                new_count = old_count - 1
                counts[ci] = new_count
                if old_count == 1 and new_count == 0:
                    unsat_set.add(ci)

    # Max-SAT formulation: smooth penalty for unsatisfied clauses
    # penalty_weight increases over time to guide search toward SAT
    def compute_smooth_penalty(counts, penalty_weight):
        """
        Compute a smooth surrogate for the number of unsatisfied clauses.
        Uses log-sum-exp for soft-min of clause satisfaction.
        For each clause, the 'cost' is -log(satisfaction_count + 1) if unsat, 0 if sat.
        This creates a smooth landscape where partially satisfied clauses have lower cost
        than completely unsatisfied ones.
        """
        total_penalty = 0.0
        for ci in range(len(clauses)):
            count = counts[ci]
            if count == 0:
                # Fully unsatisfied: high penalty
                total_penalty += penalty_weight
            elif count == 1:
                # Barely satisfied: moderate penalty (encourages finding more robust solutions)
                total_penalty += penalty_weight * 0.3
            # count >= 2: no penalty
        return total_penalty

    # Helper: evaluate flip impact using Max-SAT smooth objective
    def evaluate_flip(v, assignment, counts, penalty_weight):
        """
        Evaluate the change in smooth penalty if we flip variable v.
        Returns the new penalty value (lower is better).
        """
        v_val = assignment[v]
        delta = 0.0
        
        for ci, lit in var_occurrences[v]:
            old_count = counts[ci]
            is_sat = (lit > 0 and v_val) or (lit < 0 and not v_val)
            
            if is_sat:
                # Literal will become unsatisfied
                new_count = old_count - 1
                # Change in penalty for this clause
                old_pen = penalty_weight if old_count == 0 else (penalty_weight * 0.3 if old_count == 1 else 0.0)
                new_pen = penalty_weight if new_count == 0 else (penalty_weight * 0.3 if new_count == 1 else 0.0)
                delta += new_pen - old_pen
            else:
                # Literal will become satisfied
                new_count = old_count + 1
                old_pen = penalty_weight if old_count == 0 else (penalty_weight * 0.3 if old_count == 1 else 0.0)
                new_pen = penalty_weight if new_count == 0 else (penalty_weight * 0.3 if new_count == 1 else 0.0)
                delta += new_pen - old_pen
        
        return delta

    rng = random.Random(42)
    
    # Global best tracking
    best_assignment = None
    best_unsat_count = float('inf')
    best_smooth_score = float('inf')
    
    # ILS (Iterated Local Search) parameters
    max_steps_per_run = 3000
    perturb_size = 8  # Number of random flips for perturbation
    
    # Main loop
    while time.time() < deadline:
        # 1. Initialize or Perturb
        if best_assignment is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # With high probability, perturb the best solution (ILS)
            # With low probability, start fresh
            if rng.random() < 0.75:
                assignment = best_assignment[:]
                # Perturb using efficient re-initialization
                for _ in range(perturb_size):
                    v = rng.randint(0, n - 1)
                    assignment[v] = not assignment[v]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]
                
        counts, unsat_set = init_state(assignment)
        
        # Check if satisfied
        if is_satisfied(unsat_set):
            return assignment
            
        current_unsat = len(unsat_set)
        
        # Compute smooth score for tracking
        # Use a dynamic penalty that increases with search progress
        search_phase = min(1.0, (time.time() - start_time) / 200.0)  # Ramp up over 200s
        base_penalty = 1.0 + 2.0 * search_phase  # Penalty weight increases from 1.0 to 3.0
        
        current_smooth = compute_smooth_penalty(counts, base_penalty)
        
        if current_unsat < best_unsat_count or (current_unsat == best_unsat_count and current_smooth < best_smooth_score):
            best_unsat_count = current_unsat
            best_smooth_score = current_smooth
            best_assignment = assignment[:]
            
        # Local Search Phase
        steps = 0
        noise = 0.25  # Initial noise
        steps_since_improvement = 0
        run_best_unsat = current_unsat
        run_best_smooth = current_smooth
        run_best_assignment = assignment[:]
        
        while steps < max_steps_per_run and time.time() < deadline:
            if is_satisfied(unsat_set):
                return assignment
            
            # Pick a random unsatisfied clause
            if not unsat_set:
                return assignment
                
            clause_idx = rng.choice(list(unsat_set))
            clause = clauses[clause_idx]
            
            # Adaptive Noise
            # If we haven't improved in a while, increase noise to escape local minima
            if steps_since_improvement > 40:
                noise = min(0.85, noise + 0.06)
            elif steps_since_improvement > 10:
                noise = max(0.15, noise - 0.01)
                
            # Heuristic: Min-Conflicts with Max-SAT smooth scoring
            # Calculate score for each variable in the clause using smooth penalty
            best_v = None
            best_delta = float('inf')
            
            for l in clause:
                v = abs(l) - 1
                # Evaluate flip using smooth Max-SAT objective
                delta = evaluate_flip(v, assignment, counts, base_penalty)
                
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
                elif delta == best_delta:
                    # Tie breaker: random
                    if rng.random() < 0.5:
                        best_v = v
            
            # Apply Noise: With probability 'noise', pick a random variable from the clause
            if rng.random() < noise:
                l = rng.choice(clause)
                v = abs(l) - 1
            else:
                v = best_v
                
            # Apply flip
            apply_flip(v, assignment, counts, unsat_set)
            
            steps += 1
            
            # Update improvement tracking
            current_unsat = len(unsat_set)
            current_smooth = compute_smooth_penalty(counts, base_penalty)
            
            if current_unsat < run_best_unsat or (current_unsat == run_best_unsat and current_smooth < run_best_smooth):
                run_best_unsat = current_unsat
                run_best_smooth = current_smooth
                run_best_assignment = assignment[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            # If we found a better global best, update it
            if current_unsat < best_unsat_count or (current_unsat == best_unsat_count and current_smooth < best_smooth_score):
                best_unsat_count = current_unsat
                best_smooth_score = current_smooth
                best_assignment = assignment[:]
                
            # Early exit if we get stuck for a very long time and noise is maxed
            if steps_since_improvement > 800 and noise >= 0.85:
                break

        # End of local search run
        # If we improved the global best, we keep it.
        # The loop will restart, potentially perturbing the new best.
        
    return best_assignment if best_assignment is not None else assignment
