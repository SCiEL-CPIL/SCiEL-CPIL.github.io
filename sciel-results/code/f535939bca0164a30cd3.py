import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Encode clauses
    # Literal: var+1 for positive, -(var+1) for negative
    clauses = []
    for e in edges:
        pos_clause = [v + 1 for v in e]
        neg_clause = [-(v + 1) for v in e]
        clauses.append(pos_clause)
        clauses.append(neg_clause)
        
    # Initial assignment
    assign = [-1] * n
    
    def propagate(clauses, assign):
        """
        Applies unit propagation.
        Returns False if conflict, True if no conflict.
        Modifies clauses and assign in place.
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                sat = False
                new_clause = []
                for lit in clause:
                    var = abs(lit) - 1
                    sign = 1 if lit > 0 else -1
                    if assign[var] == sign:
                        sat = True
                        break
                    elif assign[var] == -sign:
                        continue
                    else:
                        new_clause.append(lit)
                        
                if sat:
                    continue
                if not new_clause:
                    return False
                if len(new_clause) == 1:
                    lit = new_clause[0]
                    var = abs(lit) - 1
                    sign = 1 if lit > 0 else -1
                    if assign[var] == -1:
                        assign[var] = sign
                        changed = True
                        # Clause is satisfied by this assignment, so skip adding to new_clauses
                        continue
                    elif assign[var] == -sign:
                        return False
                new_clauses.append(new_clause)
            clauses[:] = new_clauses
        return True

    def solve(clauses, assign):
        if not propagate(clauses, assign):
            return False
            
        # Find unassigned variable
        # Heuristic: Pick variable with highest frequency in remaining clauses
        var_counts = [0] * n
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                var_counts[var] += 1
                
        # Find unassigned var with max count
        best_var = -1
        max_count = -1
        for v in range(n):
            if assign[v] == -1 and var_counts[v] > max_count:
                max_count = var_counts[v]
                best_var = v
                
        if best_var == -1:
            return True # All assigned, no conflict found
        
        # Try 0
        assign[best_var] = 0
        if solve(clauses[:], assign[:]):
            return True
        assign[best_var] = -1
        
        # Try 1
        assign[best_var] = 1
        if solve(clauses[:], assign[:]):
            return True
        assign[best_var] = -1
        
        return False
        
    return solve(clauses, assign)

def construct():
    import random
    from itertools import combinations
    
    best_edges = None
    best_n = 0
    best_edge_count = float('inf')
    
    # Try different n values
    # Optimal n is likely around 15-20
    for n in range(15, 21):
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        found = False
        # Try to find a non-2-colorable instance
        # Start with a reasonably dense graph
        for attempt in range(10):
            # Try different sizes
            size = min(len(all_edges), 80 + random.randint(0, 40))
            edges = random.sample(all_edges, size)
            edges = [list(e) for e in edges]
            
            if not is_2_colorable(n, edges):
                found = True
                break
        
        if not found:
            continue
            
        current_edges = edges[:]
        
        # Greedy deletion
        # We want to minimize edges.
        # We iterate through edges and try to remove them.
        # To avoid getting stuck in a local minimum, we can do multiple passes with random shuffles.
        
        for _ in range(5): # 5 passes
            random.shuffle(current_edges)
            changed = True
            while changed:
                changed = False
                for i in range(len(current_edges) - 1, -1, -1):
                    e_to_remove = current_edges[i]
                    temp_edges = current_edges[:i] + current_edges[i+1:]
                    
                    # Check if still non-2-colorable
                    if not is_2_colorable(n, temp_edges):
                        current_edges.pop(i)
                        changed = True
                        break # Restart pass after removal
        
        if len(current_edges) < best_edge_count:
            best_edge_count = len(current_edges)
            best_edges = current_edges
            best_n = n
            
    if best_edges is None:
        # Fallback
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
