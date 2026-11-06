import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Helper: Fast 2-colorability check using SAT-style DPLL with watched literals
    # We model: For each edge e, clause C1_e = OR(x_v for v in e) [not all 0]
    #           clause C2_e = OR(not x_v for v in e) [not all 1]
    # Actually, simpler: 
    # A coloring is invalid if there exists an edge e such that all v in e are 0 OR all v in e are 1.
    # We want to know if there is a valid coloring.
    # This is equivalent to: Is the hypergraph 2-colorable?
    
    def is_2_colorable_sat(n, edges, deadline=None):
        if not edges:
            return True
            
        # Convert edges to lists of vertex indices
        # We use a simple backtracking with heuristic variable ordering (MRV)
        # and unit propagation.
        
        # To be "structurally different", we use a Clause-based representation
        # Clause: (l1, l2, l3, l4, l5) where li is (var, sign).
        # Sign 1 means var=1, Sign 0 means var=0.
        # Clause is satisfied if at least one literal is True.
        # Constraint: Not (all 0) -> Clause: (v1, 1), (v2, 1), (v3, 1), (v4, 1), (v5, 1) is NOT right.
        # "Not all 0" means at least one is 1. Clause: (v1, 1) OR (v2, 1) ...
        # "Not all 1" means at least one is 0. Clause: (v1, 0) OR (v2, 0) ...
        
        clauses = []
        for e in edges:
            # Clause 1: At least one vertex is 1
            clauses.append([(v, 1) for v in e])
            # Clause 2: At least one vertex is 0
            clauses.append([(v, 0) for v in e])
            
        # Watched Literals data structure
        # For each literal (v, s), store list of clauses watching it.
        # Literal representation: v * 2 + s (s=1 for True, s=0 for False? No, standard SAT: var>0 is True, var<0 is False)
        # Let's use standard SAT: x is True if assigned 1. 
        # Literal (v, 1) is satisfied if x_v == 1.
        # Literal (v, 0) is satisfied if x_v == 0.
        
        # We will implement a simplified DPLL.
        # Given the small size (n<=26, m<=200), a well-optimized backtracker is fine.
        # The "Structural" difference is the use of Clause Lists and Unit Propagation on Clauses, 
        # rather than counting zeros/ones in edges.
        
        num_vars = n
        assignment = [None] * num_vars # None, 0, 1
        
        # Pre-compute: for each variable, which clauses contain it?
        var_clauses = [[] for _ in range(num_vars)]
        for i, clause in enumerate(clauses):
            for lit in clause:
                v, s = lit
                var_clauses[v].append(i)
                
        # State: 
        # clause_status: 0=unassigned, 1=unit, 2=satisfied, 3=conflict
        # Actually, we just check conflicts and propagate units.
        
        def propagate():
            # Simple queue-based unit propagation
            queue = []
            # Initialize queue with unit clauses? 
            # For efficiency, we only propagate when we assign a variable.
            # This function is called after an assignment.
            pass # We'll integrate propagation into the assignment step for speed in this small context.
            
        def backtrack(level, trail_len):
            if time.time() > deadline:
                return None # Timeout, treat as unknown
            
            # Find unassigned variable with fewest unassigned values in its clauses (MRV)
            # Or just pick the first unassigned variable for simplicity in this "different" implementation
            # To be faster, let's pick a variable involved in the most unassigned clauses.
            
            min_unassigned = num_vars + 1
            best_v = -1
            for v in range(num_vars):
                if assignment[v] is not None:
                    continue
                # Count unassigned literals in clauses containing v
                count = 0
                for c_idx in var_clauses[v]:
                    clause = clauses[c_idx]
                    # Check if clause is already satisfied
                    sat = False
                    for lit in clause:
                        vv, ss = lit
                        if assignment[vv] is not None:
                            if (ss == 1 and assignment[vv] == 1) or (ss == 0 and assignment[vv] == 0):
                                sat = True
                                break
                    if not sat:
                        # Count unassigned literals in this clause
                        for lit in clause:
                            vv, ss = lit
                            if assignment[vv] is None:
                                count += 1
                if count < min_unassigned:
                    min_unassigned = count
                    best_v = v
                    if count == 0:
                        break # Can't get better than 0? Actually 0 means all clauses are satisfied or unit? 
                        # If count is 0, it means all clauses containing v are satisfied by other vars.
                        # We should avoid such variables if possible? No, if they are satisfied, they don't constrain v.
                        # We want variables that are in many UNSATISFIED clauses.
            
            if best_v == -1:
                # All assigned
                # Check for conflicts
                for clause in clauses:
                    sat = False
                    for lit in clause:
                        v, s = lit
                        if (s == 1 and assignment[v] == 1) or (s == 0 and assignment[v] == 0):
                            sat = True
                            break
                    if not sat:
                        return False # Conflict
                return True # Satisfied
                
            v = best_v
            
            # Try assignment 1
            assignment[v] = 1
            # Check for immediate conflicts
            conflict = False
            for c_idx in var_clauses[v]:
                clause = clauses[c_idx]
                # If clause becomes false (all other literals are false), conflict
                # If clause becomes unit, propagate? 
                # For simplicity, we just check if it's false. 
                # Propagation is handled by the next MRV step picking the forced variable.
                # However, without explicit propagation, we might miss units and branch unnecessarily.
                # Let's add simple propagation:
                # If a clause has only 1 unassigned literal, that literal is forced.
                
                # We'll do a quick check for conflict
                unassigned_lits = []
                for lit in clause:
                    vv, ss = lit
                    if assignment[vv] is None:
                        unassigned_lits.append(lit)
                    else:
                        if (ss == 1 and assignment[vv] == 1) or (ss == 0 and assignment[vv] == 0):
                            break # Satisfied
                else:
                    # Not satisfied by assigned vars
                    if len(unassigned_lits) == 0:
                        conflict = True
                        break
                    # If len == 1, it's a unit clause. We don't propagate here, 
                    # but the next MRV will pick it with count 1.
                
            if not conflict:
                res = backtrack(level + 1, trail_len)
                if res is not None:
                    return res
                    
            assignment[v] = 0
            # Try assignment 0
            conflict = False
            for c_idx in var_clauses[v]:
                clause = clauses[c_idx]
                unassigned_lits = []
                for lit in clause:
                    vv, ss = lit
                    if assignment[vv] is None:
                        unassigned_lits.append(lit)
                    else:
                        if (ss == 1 and assignment[vv] == 1) or (ss == 0 and assignment[vv] == 0):
                            break
                else:
                    if len(unassigned_lits) == 0:
                        conflict = True
                        break
            
            if not conflict:
                res = backtrack(level + 1, trail_len)
                if res is not None:
                    return res
                    
            assignment[v] = None
            return False
            
        return backtrack(0, 0)

    # Search Strategy: "Add then Remove"
    # 1. Try to find a small non-colorable hypergraph by starting small and adding edges.
    # 2. Once found, greedily remove edges.
    
    # First, try a known small n
    for n in range(9, 14):
        if time.time() > deadline - 5:
            break
            
        all_possible = list(combinations(range(n), 5))
        
        # Randomize the order of edges
        random.shuffle(all_possible)
        
        # Start with a small number of edges, e.g., 30
        # If n=9, total 126.
        # If n=10, total 252.
        
        # We want to find a non-colorable core.
        # Strategy: Random subset of size M. Check. If colorable, add more.
        
        current_edges = []
        
        # Phase 1: Build a non-colorable hypergraph
        # Start with 40 edges
        initial_size = 40
        if len(all_possible) < initial_size:
            initial_size = len(all_possible)
            
        current_edges = [list(e) for e in all_possible[:initial_size]]
        
        # Check if non-colorable
        if not is_2_colorable_sat(n, current_edges, deadline):
            # We have a candidate!
            # Phase 2: Greedy removal
            removed_any = True
            while removed_any and time.time() < deadline - 10:
                removed_any = False
                # Shuffle current edges to try different removals
                indices = list(range(len(current_edges)))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline - 10:
                        break
                    if i >= len(current_edges):
                        continue
                    # Try removing edge i
                    new_edges = current_edges[:i] + current_edges[i+1:]
                    # Check if still non-colorable
                    if not is_2_colorable_sat(n, new_edges, deadline):
                        current_edges = new_edges
                        removed_any = True
                        break # Restart shuffle after removal
                
                if len(current_edges) < best_m:
                    best_m = len(current_edges)
                    best_edges = current_edges[:]
                    best_n = n
                    if best_m <= 51: # Target
                        # Try to go lower?
                        pass
                    else:
                        break # Found something good, try next n or stop?
                        # Actually, keep searching for better
            continue # Try next n? Or break if we found 51?
        else:
            # Colorable, add more edges
            added = initial_size
            while added < len(all_possible) and time.time() < deadline - 10:
                # Add 5 edges at a time to speed up
                batch = 5
                new_edges_batch = all_possible[added:added+batch]
                if not new_edges_batch:
                    break
                current_edges.extend([list(e) for e in new_edges_batch])
                added += batch
                
                if not is_2_colorable_sat(n, current_edges, deadline):
                    # Found non-colorable, now reduce
                    removed_any = True
                    while removed_any and time.time() < deadline - 10:
                        removed_any = False
                        indices = list(range(len(current_edges)))
                        random.shuffle(indices)
                        for i in indices:
                            if time.time() > deadline - 10:
                                break
                            if i >= len(current_edges):
                                continue
                            new_edges = current_edges[:i] + current_edges[i+1:]
                            if not is_2_colorable_sat(n, new_edges, deadline):
                                current_edges = new_edges
                                removed_any = True
                                break
                    if len(current_edges) < best_m:
                        best_m = len(current_edges)
                        best_edges = current_edges[:]
                        best_n = n
                    break # Move to next n or restart?
                    
    if best_edges is None:
        # Fallback: Complete graph on 9 vertices
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
