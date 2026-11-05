"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations, product

def _check_unsat(n, edges):
    """
    Check if the hypergraph has no proper 2-coloring using DPLL with unit propagation.
    For each edge, we add two clauses: not(all 0) and not(all 1).
    We check if the resulting CNF is UNSAT.
    """
    # Each edge e = [v0,v1,v2,v3,v4] gives clauses:
    # (v0 or v1 or v2 or v3 or v4)  -- not all zero
    # (not v0 or not v1 or not v2 or not v3 or not v4)  -- not all one
    
    # We'll do DPLL with unit propagation
    # Variables: x[0..n-1], True = color 1, False = color 0
    
    # Build clauses as lists of (var, sign) where sign=1 means var, sign=-1 means not var
    clauses = []
    for edge in edges:
        # Clause: at least one vertex is color 1 -> (v0 or v1 or v2 or v3 or v4)
        clauses.append([(v, 1) for v in edge])
        # Clause: at least one vertex is color 0 -> (not v0 or not v1 or not v2 or not v3 or not v4)
        clauses.append([(v, -1) for v in edge])
    
    def dpll(clauses, assignment):
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Simplify clause
                satisfied = False
                remaining = []
                for (var, sign) in clause:
                    if var in assignment:
                        val = assignment[var]
                        if (val and sign == 1) or (not val and sign == -1):
                            satisfied = True
                            break
                    else:
                        remaining.append((var, sign))
                if satisfied:
                    continue
                if not remaining:
                    return None  # UNSAT
                if len(remaining) == 1:
                    var, sign = remaining[0]
                    if var in assignment:
                        # Already assigned, check
                        val = assignment[var]
                        if (val and sign == 1) or (not val and sign == -1):
                            continue
                        else:
                            return None  # UNSAT
                    assignment[var] = (sign == 1)
                    changed = True
                else:
                    new_clauses.append(remaining)
            clauses = new_clauses
        
        if not clauses:
            return assignment  # SAT
        
        # Choose a variable to branch on
        # Pick variable appearing in most clauses
        var_count = {}
        for clause in clauses:
            for (var, sign) in clause:
                var_count[var] = var_count.get(var, 0) + 1
        if not var_count:
            return assignment
        
        var = max(var_count, key=lambda v: var_count[v])
        
        # Try True
        assignment[var] = True
        result = dpll([c[:] for c in clauses], dict(assignment))
        if result is not None:
            return result
        del assignment[var]
        
        # Try False
        assignment[var] = False
        result = dpll([c[:] for c in clauses], dict(assignment))
        if result is not None:
            return result
        del assignment[var]
        
        return None  # UNSAT
    
    result = dpll(clauses, {})
    return result is None  # True if UNSAT (no 2-coloring)


def _generate_random_hypergraph(n, num_edges):
    """Generate a random 5-uniform hypergraph with given number of edges."""
    all_edges = list(combinations(range(n), 5))
    random.shuffle(all_edges)
    edges = [list(e) for e in all_edges[:num_edges]]
    return {'n': n, 'edges': edges}


def construct():
    # Start with known good construction and try to improve
    # The best known is 51 edges. Let's try to find something smaller.
    
    # First, let's try a systematic search
    # Start with a base that we know works, then try to remove edges
    
    # Known: The complete 5-graph on 9 vertices works (126 edges)
    # Let's try to find smaller ones
    
    # Strategy: Try different n and edge counts, verify with DPLL
    # Focus on ranges that are likely to work
    
    best_result = None
    best_edges = float('inf')
    
    # Try a set of configurations
    configs = [
        # (n, num_edges) - try around known bounds
        (10, 35), (10, 36), (10, 37), (10, 38), (10, 39), (10, 40),
        (11, 35), (11, 36), (11, 37), (11, 38), (11, 39), (11, 40),
        (12, 35), (12, 36), (12, 37), (12, 38), (12, 39), (12, 40),
        (9, 50), (9, 51), (9, 52),
    ]
    
    random.seed(42)
    
    for n, num_edges in configs:
        # Try multiple random instances
        for trial in range(20):
            hg = _generate_random_hypergraph(n, num_edges)
            if _check_unsat(hg['n'], hg['edges']):
                if num_edges < best_edges:
                    best_edges = num_edges
                    best_result = hg
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the complete 5-graph on 9 vertices
    from itertools import combinations
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# EVOLVE-BLOCK-END
