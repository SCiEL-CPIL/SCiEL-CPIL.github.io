import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 10.0

def _verify_not_2colorable(n, edges):
    """Verify hypergraph has no proper 2-coloring using backtracking with unit propagation."""
    # For each edge, a proper coloring must have at least one 0 and at least one 1.
    # So for each edge, NOT(all 0) AND NOT(all 1), i.e., at least one 0 AND at least one 1.
    # This is a CNF: for each edge e, (OR of x_v for v in e) AND (OR of (1-x_v) for v in e)
    # We use backtracking with unit propagation.
    
    num_vars = n
    # clauses: each clause is a list of (var, sign) where sign=1 means x, sign=-1 means not x
    clauses = []
    for e in edges:
        # Clause: at least one vertex in e is colored 1
        clause1 = [(v, 1) for v in e]
        # Clause: at least one vertex in e is colored 0
        clause2 = [(v, -1) for v in e]
        clauses.append(clause1)
        clauses.append(clause2)
    
    # Backtracking with unit propagation
    assignment = [None] * num_vars  # None, 0, or 1
    
    def propagate(clauses, assignment):
        """Unit propagation. Returns (success, new_assignment, remaining_clauses)"""
        changed = True
        remaining = []
        while changed:
            changed = False
            new_remaining = []
            for clause in remaining:
                # Check if clause is satisfied
                sat = False
                unassigned = []
                for var, sign in clause:
                    if assignment[var] is not None:
                        if sign == 1 and assignment[var] == 1:
                            sat = True
                            break
                        if sign == -1 and assignment[var] == 0:
                            sat = True
                            break
                    else:
                        unassigned.append((var, sign))
                if sat:
                    continue
                if len(unassigned) == 0:
                    return (False, None, None)  # Conflict
                if len(unassigned) == 1:
                    var, sign = unassigned[0]
                    if assignment[var] is not None:
                        return (False, None, None)
                    val = 1 if sign == 1 else 0
                    assignment[var] = val
                    changed = True
                new_remaining.append(unassigned)
            remaining = new_remaining
        return (True, assignment, remaining)
    
    def backtrack(remaining_clauses, assignment):
        if not remaining_clauses:
            return True
        # Pick a clause with fewest unassigned
        min_idx = 0
        min_len = len(remaining_clauses[0])
        for i in range(1, len(remaining_clauses)):
            if len(remaining_clauses[i]) < min_len:
                min_len = len(remaining_clauses[i])
                min_idx = i
        clause = remaining_clauses[min_idx]
        # Try first unassigned variable
        var, sign = clause[0]
        # Try value that satisfies this literal first
        val1 = 1 if sign == 1 else 0
        val2 = 1 - val1
        
        # Try val1
        old = assignment[var]
        assignment[var] = val1
        ok, new_assign, new_remaining = propagate(remaining_clauses, assignment[:])
        if ok and backtrack(new_remaining, new_assign):
            return True
        assignment[var] = val2
        ok, new_assign, new_remaining = propagate(remaining_clauses, assignment[:])
        if ok and backtrack(new_remaining, new_assign):
            return True
        assignment[var] = old
        return False
    
    # Initial propagation
    ok, new_assign, new_remaining = propagate(clauses, assignment[:])
    if not ok:
        return False  # Already unsatisfiable, so not 2-colorable
    if new_remaining is None:
        return False
    return not backtrack(new_remaining, new_assign)

def _try_greedy_set_cover(n, deadline):
    """Find a small set of 5-uniform edges on n vertices that is not 2-colorable."""
    all_edges = list(combinations(range(n), 5))
    num_colorings = 1 << n
    full_mask = (1 << num_colorings) - 1
    
    # Precompute masks: for each edge, which colorings make it monochromatic
    edge_masks = []
    for e in all_edges:
        v1, v2, v3, v4, v5 = e
        mask = 0
        for c in range(num_colorings):
            if ((c >> v1) & 1) == ((c >> v2) & 1) == ((c >> v3) & 1) == ((c >> v4) & 1) == ((c >> v5) & 1):
                mask |= (1 << c)
        edge_masks.append(mask)
    
    # Greedy set cover
    best_indices = None
    best_len = float('inf')
    
    for trial in range(30):
        if time.time() > deadline:
            break
        
        uncovered = full_mask
        selected = []
        
        for _ in range(200):  # Max edges to consider
            if time.time() > deadline:
                break
            if not uncovered:
                break
            
            best_gain = -1
            best_idx = -1
            for idx in range(len(all_edges)):
                if idx in selected:
                    continue
                gain = (edge_masks[idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                elif gain == best_gain and gain > 0 and trial > 0 and random.random() < 0.3:
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            
            if len(selected) >= best_len:
                break
        
        if not uncovered and len(selected) < best_len:
            best_len = len(selected)
            best_indices = selected[:]
            if best_len <= 51:
                break
    
    if best_indices is None:
        return None
    
    return [list(all_edges[idx]) for idx in best_indices]

def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    
    # Strategy 1: Try to find a small construction via greedy set cover on n=11
    best_edges = None
    best_n = None
    
    for n in (11, 12):
        if time.time() > deadline - 5:
            break
        try:
            result = _try_greedy_set_cover(n, deadline - 5)
            if result is not None:
                # Verify it's actually not 2-colorable
                if _verify_not_2colorable(n, result):
                    if best_edges is None or len(result) < len(best_edges):
                        best_edges = result
                        best_n = n
        except Exception:
            pass
    
    # Strategy 2: Fallback - use the complete 5-graph on a small number of vertices
    # C(9,5) = 126 edges, definitely not 2-colorable
    if best_edges is None or len(best_edges) > 100:
        # Try a denser but smaller construction
        for n in (9, 10):
            if time.time() > deadline:
                break
            edges = [list(e) for e in combinations(range(n), 5)]
            if _verify_not_2colorable(n, edges):
                if best_edges is None or len(edges) < len(best_edges):
                    best_edges = edges
                    best_n = n
    
    # Final fallback: complete 5-graph on 9 vertices
    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == '__main__':
    construct()
