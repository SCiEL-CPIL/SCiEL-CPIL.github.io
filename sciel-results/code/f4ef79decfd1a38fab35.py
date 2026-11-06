# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.optimize import linprog

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n
    
    rng = random.Random(42)
    
    # Precompute clause structure for LP and Search
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_ = [abs(l) - 1 for l in clause]
        signs = [l > 0 for l in clause]
        clause_vars.append(vars_)
        clause_signs.append(signs)
    
    # Build variable-to-clause index for fast lookup in local search
    var_to_clauses = [[] for _ in range(n)]
    for i, vars_ in enumerate(clause_vars):
        for v in vars_:
            var_to_clauses[v].append(i)

    # --- Step 1: LP Relaxation (Gaussian Elimination on Linear Relaxation) ---
    # Minimize sum(z_c)
    # Subject to:
    # For each clause c: sum_{l in c} u_l >= 1 - z_c
    # u_l = x_v if l>0, 1-x_v if l<0
    # sum_{l in c} u_l = sum_{l in c, l>0} x_l + sum_{l in c, l<0} (1 - x_{|l|})
    #                = sum_{l in c, l>0} x_l - sum_{l in c, l<0} x_{|l|} + count_neg(c)
    # Constraint: sum_{l in c, l>0} x_l - sum_{l in c, l<0} x_{|l|} + z_c >= 1 - count_neg(c)
    #             -sum_{l in c, l>0} x_l + sum_{l in c, l<0} x_{|l|} - z_c <= count_neg(c) - 1
    
    # Variables: x_0...x_{n-1}, z_0...z_{M-1}
    # c vector (objective): minimize sum(z) -> [0...0, 1...1]
    c_obj = np.zeros(n + num_clauses)
    c_obj[n:] = 1.0
    
    # Inequality constraints: A_ub x <= b_ub
    # We have M constraints.
    # Each row corresponds to a clause.
    A_ub = np.zeros((num_clauses, n + num_clauses))
    b_ub = np.zeros(num_clauses)
    
    for i, (vars_, signs) in enumerate(zip(clause_vars, clause_signs)):
        count_neg = sum(1 for s in signs if not s)
        b_ub[i] = count_neg - 1.0
        
        # Coefficients for x
        for j, (v, s) in enumerate(zip(vars_, signs)):
            if s: # Positive literal x_v
                # Term is +x_v in sum_u, so -x_v in constraint (-sum_pos + sum_neg - z <= ...)
                A_ub[i, v] -= 1.0
            else: # Negative literal (1-x_v)
                # Term is -x_v in sum_u, so +x_v in constraint
                A_ub[i, v] += 1.0
        
        # Coefficient for z_i is -1
        A_ub[i, n + i] = -1.0
    
    # Bounds for x: [0, 1], for z: [0, inf]
    bounds = [(0, 1)] * n + [(0, None)] * num_clauses
    
    # Solve LP
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x_lp = res.x[:n]
        else:
            # Fallback to random if LP fails
            x_lp = np.random.rand(n)
    except Exception:
        x_lp = np.random.rand(n)
    
    # Round LP solution to get initial boolean assignment
    # Use 0.5 threshold, maybe with slight noise to break ties
    assignment = np.zeros(n, dtype=bool)
    for i in range(n):
        # If LP value is close to 0.5, randomize
        if 0.4 < x_lp[i] < 0.6:
            assignment[i] = rng.random() < 0.5
        else:
            assignment[i] = x_lp[i] > 0.5

    # --- Step 2: Targeted Repair Phase (Optimized Local Search) ---
    
    # Vectorized satisfaction checking setup
    max_clause_len = max(len(c) for c in clauses)
    clause_matrix = np.full((num_clauses, max_clause_len), -1, dtype=int)
    clause_sign_matrix = np.zeros((num_clauses, max_clause_len), dtype=bool)
    clause_lengths = np.zeros(num_clauses, dtype=int)
    
    for i, (vars_, signs) in enumerate(zip(clause_vars, clause_signs)):
        length = len(vars_)
        clause_lengths[i] = length
        clause_matrix[i, :length] = vars_
        clause_sign_matrix[i, :length] = signs
    
    valid_mask = np.arange(max_clause_len)[None, :] < clause_lengths[:, None]
    
    def get_unsatisfied_indices():
        var_indices = clause_matrix.copy()
        var_indices[~valid_mask] = 0
        var_values = assignment[var_indices]
        lit_satisfied = var_values == clause_sign_matrix
        lit_satisfied[~valid_mask] = True
        clause_satisfied = lit_satisfied.any(axis=1)
        return np.where(~clause_satisfied)[0]
    
    max_iterations = 100000
    best_unsat_count = float('inf')
    best_assignment = assignment.copy()
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        unsat_indices = get_unsatisfied_indices()
        unsat_count = len(unsat_indices)
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, reset to best
        if no_improve_count > 5000:
            assignment = best_assignment.copy()
            no_improve_count = 0
        
        # Pick a random unsatisfied clause
        clause_idx = rng.choice(unsat_indices.tolist())
        clause = clause_vars[clause_idx]
        
        # Find variables in this clause that appear in other unsatisfied clauses
        # Prioritize flipping variables that affect many unsatisfied clauses
        # Optimization: Precompute unsat set for faster lookup
        unsat_set = set(unsat_indices.tolist())
        
        var_scores = {}
        for v in clause:
            # Count how many unsatisfied clauses contain this variable
            score = 0
            for ci in var_to_clauses[v]:
                if ci in unsat_set:
                    score += 1
            var_scores[v] = score
        
        # With some probability, pick the highest-scoring variable
        # With some probability, pick randomly (to escape local minima)
        if rng.random() < 0.3:
            var_idx = rng.choice(clause)
        else:
            # Pick variable with highest score, break ties randomly
            max_score = max(var_scores.values())
            candidates = [v for v in clause if var_scores[v] == max_score]
            var_idx = rng.choice(candidates)
        
        assignment[var_idx] = not assignment[var_idx]
    
    # Return best assignment found
    return best_assignment.tolist()
# EVOLVE-BLOCK-END
