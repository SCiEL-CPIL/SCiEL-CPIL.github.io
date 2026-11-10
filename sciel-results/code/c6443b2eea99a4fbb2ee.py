"""Planted 3-SAT solver: spectral init + incremental WalkSAT + BP-guided restarts."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    
    Strategy: 
    1. Spectral initialization from literal co-occurrence matrix
    2. Fast incremental WalkSAT with multiple heuristics
    3. BP-guided perturbations for diverse restarts
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 310.0
    
    # Pre-process clauses for faster access
    # Each clause is a list of signed literals (1-based).
    # Convert to 0-based indices and signs.
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Build variable-to-clauses index for efficient updates
    var_to_clauses = [[] for _ in range(n)]
    for i, fc in enumerate(fast_clauses):
        for var_idx, _ in fc:
            var_to_clauses[var_idx].append(i)
    
    # Pre-compute for each clause and each position in the clause, the position of that variable
    # to speed up flip_variable
    clause_var_position = []
    for fc in fast_clauses:
        pos = {}
        for j, (var_idx, req) in enumerate(fc):
            pos[var_idx] = j
        clause_var_position.append(pos)
    
    # Incremental satisfaction tracking
    # clause_satisfied[i] = True if clause i is satisfied
    # clause_sat_count[i] = number of satisfied literals in clause i
    
    def init_satisfaction(assignment):
        """Initialize clause satisfaction state."""
        clause_satisfied = [False] * num_clauses
        clause_sat_count = [0] * num_clauses
        for i, fc in enumerate(fast_clauses):
            count = 0
            for var_idx, required in fc:
                if assignment[var_idx] == required:
                    count += 1
            clause_sat_count[i] = count
            clause_satisfied[i] = count > 0
        return clause_satisfied, clause_sat_count
    
    def flip_variable(var_idx, assignment, clause_satisfied, clause_sat_count):
        """Flip variable and update clause satisfaction incrementally."""
        old_val = assignment[var_idx]
        assignment[var_idx] = not old_val
        unsat_delta = 0
        for c_idx in var_to_clauses[var_idx]:
            fc = fast_clauses[c_idx]
            was_sat = clause_satisfied[c_idx]
            old_count = clause_sat_count[c_idx]
            # Find the literal for this variable in the clause
            for var, required in fc:
                if var == var_idx:
                    if old_val == required:
                        # Was satisfied, now not
                        clause_sat_count[c_idx] = old_count - 1
                    else:
                        # Was not satisfied, now is
                        clause_sat_count[c_idx] = old_count + 1
                    break
            new_count = clause_sat_count[c_idx]
            new_sat = new_count > 0
            clause_satisfied[c_idx] = new_sat
            if was_sat and not new_sat:
                unsat_delta += 1
            elif not was_sat and new_sat:
                unsat_delta -= 1
        return unsat_delta
    
    def count_unsat_fast(clause_satisfied):
        """Count unsatisfied clauses."""
        return sum(1 for s in clause_satisfied if not s)
    
    # Spectral initialization
    def spectral_init():
        """
        Compute principal eigenvector of literal co-occurrence matrix
        to initialize the assignment.
        """
        num_literals = 2 * n
        
        # Count co-occurrences
        co_occurrence = defaultdict(float)
        
        for fc in fast_clauses:
            for i in range(len(fc)):
                for j in range(i + 1, len(fc)):
                    var_i, req_i = fc[i]
                    var_j, req_j = fc[j]
                    lit_i = var_i * 2 + (0 if req_i else 1)
                    lit_j = var_j * 2 + (0 if req_j else 1)
                    if req_i == req_j:
                        co_occurrence[(lit_i, lit_j)] += 1.0
                        co_occurrence[(lit_j, lit_i)] += 1.0
                    else:
                        co_occurrence[(lit_i, lit_j)] -= 1.0
                        co_occurrence[(lit_j, lit_i)] -= 1.0
        
        # Power iteration for principal eigenvector
        rng = random.Random(42)
        vec = [rng.gauss(0, 1) for _ in range(num_literals)]
        
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        
        for _ in range(50):
            if time.time() > deadline:
                break
            new_vec = [0.0] * num_literals
            for lit_i in range(num_literals):
                for lit_j in range(num_literals):
                    if (lit_i, lit_j) in co_occurrence:
                        new_vec[lit_i] += co_occurrence[(lit_i, lit_j)] * vec[lit_j]
            
            norm = math.sqrt(sum(x * x for x in new_vec))
            if norm < 1e-10:
                break
            vec = [x / norm for x in new_vec]
        
        assignment = []
        for var in range(n):
            pos_val = vec[2 * var]
            neg_val = vec[2 * var + 1]
            assignment.append(pos_val > neg_val)
        
        return assignment
    
    # Fast WalkSAT with incremental updates
    def fast_walksat(start_assignment, max_steps=20000, heuristic="min_conflicts"):
        """
        Fast WalkSAT with incremental clause tracking.
        Returns (assignment, solved, unsat_count).
        """
        assignment = start_assignment[:]
        clause_satisfied, clause_sat_count = init_satisfaction(assignment)
        unsat_count = count_unsat_fast(clause_satisfied)
        
        if unsat_count == 0:
            return assignment, True, 0
        
        best_unsat = unsat_count
        best_assignment = assignment[:]
        
        unsat_clauses = set(i for i in range(num_clauses) if not clause_satisfied[i])
        
        # Precompute unsat clause counts per variable for faster heuristics
        var_unsat_count = [0] * n
        for c_idx in unsat_clauses:
            for var_idx, _ in fast_clauses[c_idx]:
                var_unsat_count[var_idx] += 1
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment, True, 0
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # Choose a clause from unsat_clauses
            clause_idx = random.choice(list(unsat_clauses))
            clause = fast_clauses[clause_idx]
            
            # Heuristic for variable selection
            if heuristic == "min_conflicts":
                # Prefer literals that appear in fewer unsat clauses
                scores = [var_unsat_count[var_idx] for var_idx, _ in clause]
                min_score = min(scores)
                candidates = [i for i, s in enumerate(scores) if s == min_score]
                chosen_idx = random.choice(candidates)
                
            elif heuristic == "weighted_random":
                # Weighted random choice
                scores = []
                for var_idx, required in clause:
                    score = 1 + var_unsat_count[var_idx]
                    scores.append(score)
                
                total_score = sum(scores)
                r = random.random() * total_score
                cumulative = 0
                chosen_idx = 0
                for i, score in enumerate(scores):
                    cumulative += score
                    if r <= cumulative:
                        chosen_idx = i
                        break
                        
            else:  # random
                chosen_idx = random.randint(0, len(clause) - 1)
            
            var_idx, required = clause[chosen_idx]
            
            # Update var_unsat_count before flipping
            old_unsat = list(var_unsat_count)
            for c_idx in var_to_clauses[var_idx]:
                if not clause_satisfied[c_idx]:
                    var_unsat_count[var_idx] -= 1
            
            # Flip the variable and update
            delta = flip_variable(var_idx, assignment, clause_satisfied, clause_sat_count)
            unsat_count += delta
            
            # Update unsat_clauses set and var_unsat_count
            for c_idx in var_to_clauses[var_idx]:
                if clause_satisfied[c_idx]:
                    if c_idx in unsat_clauses:
                        unsat_clauses.discard(c_idx)
                        for v, _ in fast_clauses[c_idx]:
                            var_unsat_count[v] -= 1
                else:
                    if c_idx not in unsat_clauses:
                        unsat_clauses.add(c_idx)
                        for v, _ in fast_clauses[c_idx]:
                            var_unsat_count[v] += 1
        
        return best_assignment, False, best_unsat
    
    # Belief Propagation (simplified, for diversity in restarts)
    def run_bp_simple(max_iterations=20):
        """
        Simplified BP for generating diverse starting points.
        """
        beliefs = [0.0] * n
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            new_beliefs = [0.0] * n
            for var_idx in range(n):
                sum_msg = 0.0
                for c_idx in var_to_clauses[var_idx]:
                    fc = fast_clauses[c_idx]
                    for var, required in fc:
                        if var != var_idx:
                            sum_msg += beliefs[var] * (1 if required else -1)
                new_beliefs[var_idx] = sum_msg
            
            for i in range(n):
                beliefs[i] = 0.5 * beliefs[i] + 0.5 * new_beliefs[i]
        
        assignment = [b > 0 for b in beliefs]
        confidences = [abs(b) for b in beliefs]
        return assignment, confidences
    
    # Main solver
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Phase 1: Spectral initialization
    spectral_assignment = spectral_init()
    spec_unsat = count_unsat_fast(init_satisfaction(spectral_assignment)[0])
    
    if spec_unsat == 0:
        return spectral_assignment
    
    # Fast WalkSAT from spectral init
    ls_assignment, solved, ls_unsat = fast_walksat(spectral_assignment, max_steps=30000, heuristic="min_conflicts")
    if solved:
        return ls_assignment
    
    if ls_unsat < best_unsat_count:
        best_unsat_count = ls_unsat
        best_assignment = ls_assignment[:]
    
    # Phase 2: BP-guided perturbations
    bp_assignment, bp_confidences = run_bp_simple(max_iterations=15)
    bp_unsat = count_unsat_fast(init_satisfaction(bp_assignment)[0])
    
    if bp_unsat == 0:
        return bp_assignment
    
    sorted_vars = sorted(range(n), key=lambda i: bp_confidences[i])
    
    num_perturbed_restarts = 80
    for restart in range(num_perturbed_restarts):
        if time.time() > deadline:
            break
        
        assignment = bp_assignment[:]
        num_flips = random.randint(1, min(3, n))
        for _ in range(num_flips):
            if random.random() < 0.7 and sorted_vars:
                var_idx = sorted_vars[random.randint(0, min(8, len(sorted_vars) - 1))]
            else:
                var_idx = random.randint(0, n - 1)
            assignment[var_idx] = not assignment[var_idx]
        
        heuristic = "min_conflicts" if restart % 2 == 0 else "weighted_random"
        ls_assignment, solved, ls_unsat = fast_walksat(assignment, max_steps=15000, heuristic=heuristic)
        if solved:
            return ls_assignment
        
        if ls_unsat < best_unsat_count:
            best_unsat_count = ls_unsat
            best_assignment = ls_assignment[:]
    
    # Phase 3: Pure random restarts
    num_random_restarts = 30
    for restart in range(num_random_restarts):
        if time.time() > deadline:
            break
        
        assignment = [random.random() < 0.5 for _ in range(n)]
        heuristic = "min_conflicts" if restart % 2 == 0 else "random"
        ls_assignment, solved, ls_unsat = fast_walksat(assignment, max_steps=10000, heuristic=heuristic)
        if solved:
            return ls_assignment
        
        if ls_unsat < best_unsat_count:
            best_unsat_count = ls_unsat
            best_assignment = ls_assignment[:]
    
    if best_assignment is not None:
        return best_assignment
    else:
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
