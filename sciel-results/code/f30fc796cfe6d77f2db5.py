"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit
    
    # Convert to 0-based internal representation
    clauses_internal = []
    for c in clauses:
        clause_int = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_int.append((var, sign))
        clauses_internal.append(clause_int)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Precompute clause data as numpy arrays for efficient WalkSAT
    clause_vars = np.array([[v for v, s in c] for c in clauses_internal], dtype=np.int32)
    clause_signs = np.array([[s for v, s in c] for c in clauses_internal], dtype=np.int8)
    
    # === SPECTRAL CLUSTERING: Partition into connected components ===
    # Build variable co-occurrence graph (variables appear together in clauses)
    row_idx = []
    col_idx = []
    data = []
    for ci, clause in enumerate(clauses_internal):
        vars_in_clause = [v for v, s in clause]
        for i in range(len(vars_in_clause)):
            for j in range(len(vars_in_clause)):
                if i != j:
                    row_idx.append(vars_in_clause[i])
                    col_idx.append(vars_in_clause[j])
                    data.append(1)
    
    if len(row_idx) > 0:
        cooccurrence = csr_matrix(
            (data, (row_idx, col_idx)), shape=(n, n)
        )
    else:
        cooccurrence = csr_matrix((n, n))
    
    # Find connected components
    n_components, component_labels = connected_components(cooccurrence, directed=False)
    
    # Group clauses by their primary component (based on first variable)
    component_clauses = {}
    component_vars = {}
    for comp_id in range(n_components):
        vars_in_comp = [v for v in range(n) if component_labels[v] == comp_id]
        component_vars[comp_id] = vars_in_comp
    
    for ci, clause in enumerate(clauses_internal):
        # Assign clause to the component of its first variable
        primary_comp = component_labels[clause[0][0]]
        if primary_comp not in component_clauses:
            component_clauses[primary_comp] = []
        component_clauses[primary_comp].append(ci)
    
    # Solve each component independently
    # Sort components by size (largest first for time budget allocation)
    comp_sizes = [(comp_id, len(component_vars[comp_id])) for comp_id in range(n_components)]
    comp_sizes.sort(key=lambda x: -x[1])
    
    # Allocate time budget: give more time to larger components
    total_time_budget = deadline - start_time - 5.0  # 5s safety margin
    # Use sqrt scaling: larger components get proportionally more time
    size_weights = [np.sqrt(size) for _, size in comp_sizes]
    total_weight = sum(size_weights)
    comp_time_budgets = {}
    for (comp_id, size), weight in zip(comp_sizes, size_weights):
        comp_time_budgets[comp_id] = (weight / total_weight) * total_time_budget
    
    # Result storage
    final_assignment = [None] * n
    solved_components = 0
    
    # Helper functions for component-level solving
    def count_unsatisfied_comp(assign_dict, clause_indices):
        """Count unsatisfied clauses for a subset using a dict assignment."""
        unsat_count = 0
        for ci in clause_indices:
            clause = clauses_internal[ci]
            sat = False
            for var, sign in clause:
                val = assign_dict.get(var)
                if val is not None and val == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        return unsat_count
    
    def dpll_comp(assignment, clause_indices, time_limit):
        """
        DPLL for a subset of clauses.
        assignment: dict var -> bool/None
        Returns: dict if SAT, None if UNSAT or timeout
        """
        if time.time() > time_limit:
            return None
        
        # Unit propagation
        while True:
            if time.time() > time_limit:
                return None
            
            unit_vars = []
            conflict = False
            
            for ci in clause_indices:
                clause = clauses_internal[ci]
                unassigned = []
                for var, sign in clause:
                    val = assignment.get(var)
                    if val is None:
                        unassigned.append((var, sign))
                    elif val == (sign == 1):
                        break  # clause satisfied
                else:
                    if len(unassigned) == 0:
                        return None  # conflict
                    elif len(unassigned) == 1:
                        unit_vars.append(unassigned[0])
            
            if not unit_vars:
                break
            
            for var, sign in unit_vars:
                val = (sign == 1)
                if var in assignment:
                    if assignment[var] != val:
                        return None  # conflict
                    continue
                assignment[var] = val
        
        # Check if all variables in clauses are assigned
        all_vars_in_clauses = set()
        for ci in clause_indices:
            for var, sign in clauses_internal[ci]:
                all_vars_in_clauses.add(var)
        
        all_assigned = all(assignment.get(v) is not None for v in all_vars_in_clauses)
        if all_assigned:
            return assignment
        
        # VSIDS variable selection
        var_scores = {}
        for ci in clause_indices:
            clause = clauses_internal[ci]
            clause_unsat = True
            for var, sign in clause:
                val = assignment.get(var)
                if val is not None and val == (sign == 1):
                    clause_unsat = False
                    break
            if clause_unsat:
                for var, sign in clause:
                    if assignment.get(var) is None:
                        var_scores[var] = var_scores.get(var, 0) + 1
        
        best_var = -1
        best_score = -1
        for var in all_vars_in_clauses:
            if assignment.get(var) is None and var_scores.get(var, 0) > best_score:
                best_score = var_scores.get(var, 0)
                best_var = var
        
        if best_var == -1:
            return None
        
        # Decide: try True first
        assignment[best_var] = True
        saved = assignment.copy()
        
        result = dpll_comp(assignment, clause_indices, time_limit)
        if result is not None:
            return result
        
        # Backtrack
        assignment.clear()
        assignment.update(saved)
        
        # Try False
        assignment[best_var] = False
        saved = assignment.copy()
        
        result = dpll_comp(assignment, clause_indices, time_limit)
        if result is not None:
            return result
        
        # Backtrack again
        assignment.clear()
        assignment.update(saved)
        
        return None
    
    def walksat_comp(assign_list, var_map, clause_indices, time_limit, rng):
        """
        WalkSAT for a subset of clauses with incremental delta computation.
        assign_list: list of bools for the component variables
        var_map: dict local_idx -> global_var
        clause_indices: list of global clause indices
        Returns: list of bools or None
        """
        local_n = len(var_map)
        
        # Precompute local clause structure with efficient lookup
        var_to_local = {v: i for i, v in var_map.items()}
        
        local_clauses = []
        for ci in clause_indices:
            local_clause = []
            for var, sign in clauses_internal[ci]:
                if var in var_to_local:
                    local_clause.append((var_to_local[var], sign))
            local_clauses.append(local_clause)
        
        # Build local variable-to-clause index for delta computation
        local_var_to_clauses = [[] for _ in range(local_n)]
        for ci, lc in enumerate(local_clauses):
            for li, sign in lc:
                local_var_to_clauses[li].append((ci, sign))
        
        def local_unsat_count(assign):
            count = 0
            for lc in local_clauses:
                sat = False
                for li, sign in lc:
                    if assign[li] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    count += 1
            return count
        
        def compute_delta(li, new_val, assign):
            """Compute delta (change in unsatisfied count) when flipping var li to new_val."""
            delta = 0
            for ci, sign in local_var_to_clauses[li]:
                lc = local_clauses[ci]
                # Check if clause is currently satisfied
                was_sat = False
                for lli, s in lc:
                    if assign[lli] == (s == 1):
                        was_sat = True
                        break
                
                # Check if clause would be satisfied after flip
                would_be_sat = False
                for lli, s in lc:
                    if lli == li:
                        if new_val == (s == 1):
                            would_be_sat = True
                            break
                    else:
                        if assign[lli] == (s == 1):
                            would_be_sat = True
                            break
                
                if was_sat and not would_be_sat:
                    delta += 1
                elif not was_sat and would_be_sat:
                    delta -= 1
            return delta
        
        best_assign = None
        best_unsat = float('inf')
        noise_start = 0.45
        patience = 5000
        
        while time.time() < time_limit:
            # Initialize
            current = [rng.random() < 0.5 for _ in range(local_n)]
            current_unsat = local_unsat_count(current)
            run_best = current_unsat
            no_improve = 0
            
            for step in range(100000):
                if time.time() > time_limit:
                    break
                
                if current_unsat == 0:
                    return current
                
                if current_unsat < run_best:
                    run_best = current_unsat
                    no_improve = 0
                    if current_unsat < best_unsat:
                        best_unsat = current_unsat
                        best_assign = list(current)
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                
                # Find unsatisfied clauses
                unsat_clauses = []
                for ci, lc in enumerate(local_clauses):
                    sat = False
                    for li, sign in lc:
                        if current[li] == (sign == 1):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(ci)
                
                if not unsat_clauses:
                    return current
                
                chosen = rng.choice(unsat_clauses)
                lc = local_clauses[chosen]
                
                # Pick variable to flip
                if rng.random() < noise_start:
                    candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                    if not candidates:
                        candidates = [li for li, sign in lc]
                    best_var = rng.choice(candidates)
                else:
                    # Min-delta heuristic with incremental computation
                    best_var = None
                    best_delta = float('inf')
                    for li, sign in lc:
                        if current[li] == (sign == 1):
                            continue
                        new_val = not current[li]
                        delta = compute_delta(li, new_val, current)
                        if delta < best_delta:
                            best_delta = delta
                            best_var = li
                    if best_var is None:
                        candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                        if not candidates:
                            candidates = [li for li, sign in lc]
                        best_var = rng.choice(candidates)
                
                current[best_var] = not current[best_var]
                current_unsat = local_unsat_count(current)
            
            noise_start = max(0.15, noise_start * 0.95)
        
        return best_assign
    
    # Solve each component
    for comp_id, size in comp_sizes:
        if time.time() > deadline - 5:
            break
        
        vars_in_comp = component_vars[comp_id]
        clauses_in_comp = component_clauses.get(comp_id, [])
        
        if not clauses_in_comp:
            # No clauses, assign arbitrarily
            for v in vars_in_comp:
                final_assignment[v] = False
            solved_components += 1
            continue
        
        comp_budget = comp_time_budgets[comp_id]
        comp_deadline = start_time + (deadline - start_time) * 0.95  # Use 95% of total time for all comps
        
        # Try DPLL first for small components
        if size <= 25:
            dpll_limit = min(start_time + 3.0, comp_deadline - 2)
            assignment_dict = {}
            result = dpll_comp(assignment_dict, clauses_in_comp, dpll_limit)
            if result is not None:
                for v in vars_in_comp:
                    final_assignment[v] = result.get(v, False)
                solved_components += 1
                continue
        
        # WalkSAT for larger components
        rng = random.Random(hash(comp_id) % (2**31))
        var_map = {i: v for i, v in enumerate(vars_in_comp)}
        
        ws_limit = min(start_time + comp_budget, comp_deadline - 2)
        result = walksat_comp(
            [False] * size,
            var_map,
            clauses_in_comp,
            ws_limit,
            rng
        )
        
        if result is not None:
            for i, v in enumerate(vars_in_comp):
                final_assignment[v] = result[i]
            solved_components += 1
        else:
            # Fallback: random assignment
            for v in vars_in_comp:
                final_assignment[v] = random.Random(v).random() < 0.5
    
    # Fill any unassigned variables
    for v in range(n):
        if final_assignment[v] is None:
            final_assignment[v] = False
    
    return [bool(x) for x in final_assignment]


# EVOLVE-BLOCK-END
