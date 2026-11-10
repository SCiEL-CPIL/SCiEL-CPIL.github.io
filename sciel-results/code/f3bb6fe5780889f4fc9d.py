import random
import math
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # Precompute variable to clause indices
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_to_clauses[var].append(i)
    
    # Helper to check if literal is satisfied
    # assignment is list of bools, index 0..n-1
    # lit > 0 means var must be True, lit < 0 means var must be False
    def is_lit_satisfied(lit, assignment):
        var = abs(lit) - 1
        if lit > 0:
            return assignment[var]
        else:
            return not assignment[var]

    # Helper to compute initial clause counts
    def init_counts(assignment):
        counts = [0] * m
        for i, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                if is_lit_satisfied(lit, assignment):
                    count += 1
            counts[i] = count
        return counts

    # SA Parameters
    T0 = 1.0
    alpha = 0.995  # Cooling rate
    T_min = 0.001
    max_iter_per_restart = 50000
    
    best_energy = float('inf')
    best_assignment = None
    
    # Try multiple restarts
    while time.time() < deadline:
        # Initialize random assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        clause_counts = init_counts(assignment)
        
        # Calculate initial energy
        energy = sum(1 for c in clause_counts if c == 0)
        
        if energy == 0:
            return assignment
        
        if energy < best_energy:
            best_energy = energy
            best_assignment = assignment[:]
        
        T = T0
        iters = 0
        
        while T > T_min and iters < max_iter_per_restart:
            if time.time() > deadline:
                break
            
            # Pick a random unsatisfied clause
            # To avoid O(m) scan every time, we can keep a list of unsat clause indices
            # But for simplicity and speed, if m is large, scanning is slow.
            # Let's maintain a list of unsatisfied clause indices.
            # Actually, let's just pick a random variable from the whole set?
            # No, focused is better.
            # Let's find an unsatisfied clause.
            # If we maintain a set/list of unsat clauses, we can pick from there.
            
            # Optimization: Maintain a list of unsatisfied clause indices.
            # This requires updating it on every flip.
            # Let's try a simpler approach first: Random variable selection.
            # If random variable selection is too slow to converge, we switch to focused.
            
            # Let's stick to Random Variable Selection for now, as it's robust and simple.
            # But wait, the knowledge says "O(1) amortized delta... enabling 50x more search steps".
            # This implies the bottleneck was delta calculation. My delta calc is O(degree).
            # With random variable selection, we might pick variables that don't affect many unsat clauses.
            
            # Let's use Focused Selection: Pick random unsat clause, then random var in it.
            # To pick random unsat clause efficiently:
            # We can keep a list `unsat_clauses`.
            # Initially, build it.
            unsat_clauses = [i for i, c in enumerate(clause_counts) if c == 0]
            
            if not unsat_clauses:
                break # Solved
            
            # Pick random unsat clause
            c_idx = random.choice(unsat_clauses)
            clause = clauses[c_idx]
            
            # Pick random literal/variable from this clause
            lit = random.choice(clause)
            var = abs(lit) - 1
            
            # Calculate delta E for flipping 'var'
            delta_e = 0
            for c2_idx in var_to_clauses[var]:
                c2 = clauses[c2_idx]
                # Check if 'var' is in c2 (it is, by construction)
                # Find the literal of 'var' in c2
                # Since clauses are small (size 3), we can just loop
                lit_in_c2 = 0
                for l in c2:
                    if abs(l) - 1 == var:
                        lit_in_c2 = l
                        break
                
                current_count = clause_counts[c2_idx]
                is_sat = is_lit_satisfied(lit_in_c2, assignment)
                
                if is_sat:
                    # Flipping makes it unsat. Count decreases by 1.
                    if current_count == 1:
                        # Clause was satisfied (only this lit), becomes unsat
                        delta_e += 1
                else:
                    # Flipping makes it sat. Count increases by 1.
                    if current_count == 0:
                        # Clause was unsat, becomes sat
                        delta_e -= 1
            
            # Accept?
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                # Flip
                assignment[var] = not assignment[var]
                
                # Update clause_counts and unsat_clauses
                # This is the expensive part if not done carefully.
                # We need to update counts for all clauses containing 'var'.
                # And update unsat_clauses list.
                
                # To keep unsat_clauses consistent, we might need to rebuild it or update it.
                # Rebuilding is O(m). Updating is tricky with list removals.
                # Given m is likely not huge (planted 3-SAT usually has m ~ 4-5n), O(m) rebuild might be acceptable if it allows O(1) access to unsat clauses?
                # No, O(m) rebuild every step kills the speedup.
                
                # Alternative: Don't maintain unsat_clauses list.
                # Instead, pick a random clause from ALL clauses?
                # If we pick a random clause, and it's satisfied, we still pick a var and flip.
                # This is "Random Clause Selection".
                # Probability of picking an unsat clause is energy/m.
                # This is simpler.
                
                # Let's switch strategy:
                # 1. Pick random clause index c_idx from [0, m)
                # 2. Pick random var from that clause
                # 3. Calculate delta
                # 4. Accept/Reject
                # 5. Update counts
                # This avoids maintaining unsat_clauses list.
                # The "focus" is lost, but SA with cooling should still work.
                
                # Wait, if I don't focus on unsat clauses, I might waste many moves on already satisfied clauses.
                # However, the delta calculation is still O(degree).
                # And the cost of picking a clause is O(1).
                # The cost of updating counts is O(degree).
                # So each step is O(degree).
                # This is the "O(1) amortized" (well, O(degree) but degree is small and fixed) approach.
                
                # Let's restart the loop logic with Random Clause Selection.
                
                # Actually, I already calculated delta_e above based on a specific var.
                # If I accept, I must update counts.
                
                # Let's update counts for all clauses containing 'var'
                for c2_idx in var_to_clauses[var]:
                    c2 = clauses[c2_idx]
                    lit_in_c2 = 0
                    for l in c2:
                        if abs(l) - 1 == var:
                            lit_in_c2 = l
                            break
                    # The literal 'lit_in_c2' has just been flipped.
                    # If it was satisfied, it's now unsat -> count - 1
                    # If it was unsat, it's now sat -> count + 1
                    # We can determine this from the old assignment state (before flip)
                    # But we already flipped assignment[var].
                    # So we can check the NEW state.
                    # If NEW state is satisfied, it was unsat before -> count + 1
                    # If NEW state is unsatisfied, it was sat before -> count - 1
                    if is_lit_satisfied(lit_in_c2, assignment):
                        clause_counts[c2_idx] += 1
                    else:
                        clause_counts[c2_idx] -= 1
                
                energy += delta_e
                
                if energy == 0:
                    return assignment
                
                if energy < best_energy:
                    best_energy = energy
                    best_assignment = assignment[:]
            
            # Cool down
            T *= alpha
            iters += 1
            
            # Optional: If stuck for too long, restart
            # We handle restarts in the outer loop.
        
        # End of inner loop (cooled down or max iter)
        # If not solved, outer loop will restart with new random assignment

    # If deadline reached, return best found
    if best_assignment:
        return best_assignment
    else:
        # Fallback: return last assignment or random
        return [False] * n
