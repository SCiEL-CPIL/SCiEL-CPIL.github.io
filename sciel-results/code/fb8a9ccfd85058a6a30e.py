"""Find a small 5-uniform hypergraph with no proper 2-colouring."""
import time
import random
from itertools import combinations
from collections import defaultdict

def _check_2colorable(n, edges, deadline):
    """Check if hypergraph has a proper 2-colouring using backtracking with unit propagation.
    Returns True if 2-colourable (SAT), False if not (UNSAT)."""
    # Build adjacency: for each edge, which vertices
    # Unit propagation: if 4 of 5 in an edge have same colour, 5th must be opposite
    # Contradiction: all 5 same colour
    
    # Precompute: for each vertex, list of edges containing it
    vert_edges = defaultdict(list)
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    m = len(edges)
    # assignment: -1 = unassigned, 0 or 1 = colour
    assignment = [-1] * n
    
    # For unit propagation, we track for each edge how many of each colour
    # edge_colour_count[edge_idx] = [count0, count1]
    # But we need to be careful with backtracking, so let's use a different approach
    
    # Actually, let's implement a simpler but correct backtracking with unit propagation
    # State: assignment array, and we propagate
    
    def propagate(assignment, edges, vert_edges, n):
        """Apply unit propagation. Returns (new_assignment, contradiction_flag, forced_changes)
        forced_changes is list of (vertex, colour) that were forced."""
        changed = True
        forced = []
        while changed:
            changed = False
            for e_idx, e in enumerate(edges):
                # Count colours in this edge
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    a = assignment[v]
                    if a == 0:
                        c0 += 1
                    elif a == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return assignment, True, forced  # contradiction
                
                # If 4 are same colour, the 5th must be opposite
                if len(unassigned) == 1:
                    v = unassigned[0]
                    if c0 == 4:
                        assignment[v] = 1
                        forced.append((v, 1))
                        changed = True
                    elif c1 == 4:
                        assignment[v] = 0
                        forced.append((v, 0))
                        changed = True
        return assignment, False, forced
    
    def backtrack(assignment, edges, vert_edges, n, deadline):
        """Returns True if a valid 2-colouring exists (SAT)."""
        if time.time() > deadline:
            return True  # assume colourable if we run out of time (conservative)
        
        # Check for contradiction
        for e_idx, e in enumerate(edges):
            c0 = sum(1 for v in e if assignment[v] == 0)
            c1 = sum(1 for v in e if assignment[v] == 1)
            if c0 == 5 or c1 == 5:
                return False  # contradiction, this branch fails
        
        # Find first unassigned vertex
        unassigned_verts = [v for v in range(n) if assignment[v] == -1]
        if not unassigned_verts:
            # All assigned, check all edges
            for e in edges:
                c0 = sum(1 for v in e if assignment[v] == 0)
                c1 = sum(1 for v in e if assignment[v] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True
        
        # Pick vertex with most constraints (heuristic)
        v = min(unassigned_verts, key=lambda x: -len(vert_edges[x]))
        
        # Try colour 0
        assignment[v] = 0
        # Propagate
        new_assign, contra, _ = propagate(assignment[:], edges, vert_edges, n)
        if not contra:
            if backtrack(new_assign, edges, vert_edges, n, deadline):
                return True
        assignment[v] = -1
        
        # Try colour 1
        assignment[v] = 1
        new_assign, contra, _ = propagate(assignment[:], edges, vert_edges, n)
        if not contra:
            if backtrack(new_assign, edges, vert_edges, n, deadline):
                return True
        assignment[v] = -1
        
        return False
    
    return backtrack(assignment, edges, vert_edges, n, deadline)


def _find_minimal_non2colorable(n, all_edges, deadline, seed=42):
    """Try to find a small non-2-colourable subhypergraph of all_edges on n vertices."""
    rng = random.Random(seed)
    m = len(all_edges)
    
    # Start with all edges, try to remove while maintaining non-2-colourability
    # But first, check if the full hypergraph is non-2-colourable
    if time.time() > deadline:
        return None
    
    current_edges = list(range(m))  # indices into all_edges
    
    # Check if full is non-2-colourable
    if _check_2colorable(n, [all_edges[i] for i in current_edges], deadline):
        return None  # Full is 2-colourable, can't get non-2-colourable subset
    
    # Try to remove edges greedily
    order = list(range(m))
    rng.shuffle(order)
    
    for e_idx in order:
        if time.time() > deadline:
            break
        if e_idx not in current_edges:
            continue
        # Try removing this edge
        new_edges = [i for i in current_edges if i != e_idx]
        if not new_edges:
            break
        # Check if still non-2-colourable
        if not _check_2colorable(n, [all_edges[i] for i in new_edges], deadline):
            current_edges = new_edges
    
    return [all_edges[i] for i in current_edges]


def construct():
    deadline = time.time() + 110  # safety margin
    
    # Known: complete 5-uniform on 9 vertices has 126 edges and is non-2-colourable
    # Let's try to find something smaller
    
    best_result = None
    best_edge_count = float('inf')
    
    # Try different vertex counts
    for n in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        if time.time() > deadline:
            break
        
        all_edges = [list(e) for e in combinations(range(n), 5)]
        m = len(all_edges)
        
        if m > 200:
            # Too many edges to consider all, but we can try subsets
            # For now, skip if too many
            if m > 500:
                continue
        
        # First, verify the complete hypergraph is non-2-colourable
        if not _check_2colorable(n, all_edges, deadline):
            # It's non-2-colourable, try to minimize
            result = _find_minimal_non2colorable(n, all_edges, deadline)
            if result is not None:
                edge_count = len(result)
                if edge_count < best_edge_count:
                    best_edge_count = edge_count
                    best_result = {'n': n, 'edges': result}
        
        # Also try random subsets to find non-2-colourable ones
        # Try with random edge subsets of various sizes
        for trial in range(5):
            if time.time() > deadline:
                break
            # Try different target sizes
            for target_size in [30, 40, 50, 60, 80, 100, 120]:
                if target_size > m:
                    continue
                if time.time() > deadline:
                    break
                
                rng = random.Random(n * 1000 + trial * 100 + target_size)
                subset_idx = rng.sample(range(m), target_size)
                subset_edges = [all_edges[i] for i in subset_idx]
                
                if not _check_2colorable(n, subset_edges, deadline):
                    # Found a non-2-colourable subset!
                    # Try to minimize it
                    current = list(subset_idx)
                    # Greedy removal
                    removal_order = list(range(target_size))
                    rng.shuffle(removal_order)
                    for pos in removal_order:
                        if time.time() > deadline:
                            break
                        e_idx = current[pos]
                        if e_idx is None:
                            continue
                        new_current = [x for x in current if x != e_idx]
                        if not new_current:
                            break
                        if not _check_2colorable(n, [all_edges[i] for i in new_current], deadline):
                            current = new_current
                    
                    if current:
                        final_edges = [all_edges[i] for i in current]
                        edge_count = len(final_edges)
                        if edge_count < best_edge_count:
                            best_edge_count = edge_count
                            best_result = {'n': n, 'edges': final_edges}
    
    if best_result is None:
        # Fallback: complete 5-uniform on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return best_result
