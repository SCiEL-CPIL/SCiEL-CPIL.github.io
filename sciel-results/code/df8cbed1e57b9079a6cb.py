"""Construct a dense edge-coloured graph on 20 vertices with rainbow 4-cycles."""
import time
import itertools
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 20
    num_colors = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges_total = len(all_edges)
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d.
    # The edges are (a,b), (b,c), (c,d), (d,a).
    # To avoid duplicates, we can iterate over combinations of 4 vertices.
    # For each set {a,b,c,d}, there are 3 distinct 4-cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d) ? 
    # Actually, for 4 vertices, the number of Hamiltonian cycles in K4 is (4-1)!/2 = 3.
    # Let's just generate all 4-cycles explicitly to be safe.
    
    cycles = []
    # To avoid duplicates, we can enforce a canonical ordering.
    # A cycle is a set of 4 edges.
    # Let's generate all 4-tuples (a,b,c,d) with a<b<c<d? No, the cycle structure matters.
    # Let's iterate a < b < c < d.
    # The 3 cycles on {a,b,c,d} are:
    # 1. a-b-c-d-a (edges: ab, bc, cd, da)
    # 2. a-b-d-c-a (edges: ab, bd, dc, ca)
    # 3. a-c-b-d-a (edges: ac, cb, bd, da)
    
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Cycle 1: a-b-c-d-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,c), max(b,c))
        e3 = (min(c,d), max(c,d))
        e4 = (min(d,a), max(d,a))
        cycles.append([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]])
        
        # Cycle 2: a-b-d-c-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,d), max(b,d))
        e3 = (min(d,c), max(d,c))
        e4 = (min(c,a), max(c,a))
        cycles.append([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]])
        
        # Cycle 3: a-c-b-d-a
        e1 = (min(a,c), max(a,c))
        e2 = (min(c,b), max(c,b))
        e3 = (min(b,d), max(b,d))
        e4 = (min(d,a), max(d,a))
        cycles.append([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]])
    
    # Map edge to cycles it belongs to
    edge_cycles = [[] for _ in range(num_edges_total)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_cycles[e].append(i)
            
    # CSP Solver
    # State:
    # colors: array of size num_edges_total, -1 for unassigned/ABSTAIN, 0..19 for color
    # domains: list of sets for each edge. Initially 0..19 U {ABSTAIN}
    
    # We will use a recursive backtracking solver with forward checking.
    # To maximize edges, we prefer assigning a color over ABSTAIN.
    
    best_solution = None
    best_edge_count = 0
    
    def solve_csp(seed):
        nonlocal best_solution, best_edge_count
        
        rng = random.Random(seed)
        
        # Shuffle edge order for variable ordering
        order = list(range(num_edges_total))
        rng.shuffle(order)
        
        # Reverse order so we process in the shuffled order
        # We'll use a stack-based recursion or just recursion with a depth counter.
        # Recursion depth 190 is safe.
        
        colors = [-1] * num_edges_total
        # Domains: 0..19 and -1 (ABSTAIN)
        # Using lists for domains is faster than sets for small sizes if we check membership manually?
        # Or just sets.
        domains = [set(range(num_colors + 1)) for _ in range(num_edges_total)] # -1 to 19
        
        # We need to track the number of edges currently assigned (non -1)
        # and update it during backtracking.
        
        def backtrack(depth):
            nonlocal best_solution, best_edge_count
            
            if time.time() > deadline:
                return False
            
            if depth == num_edges_total:
                # Check if better
                count = sum(1 for c in colors if c != -1)
                if count > best_edge_count:
                    best_edge_count = count
                    best_solution = colors[:]
                return True # Found a solution, but we want to keep searching for better?
                # Actually, we want to find the *best*. 
                # If we return True, we stop this branch. 
                # We should continue searching if we have time.
                # But backtracking naturally explores all solutions.
                # To maximize, we should prune branches that cannot beat best_edge_count.
                # Max possible edges in remaining = num_edges_total - depth
                # Current edges = count_so_far
                # If count_so_far + remaining <= best_edge_count: prune.
                # We need to track count_so_far.
                
            # Heuristic: Select next variable. 
            # For now, use static order (depth). 
            # Dynamic MRV would be better but expensive.
            e_idx = order[depth]
            
            # Try colors first (to maximize edges), then ABSTAIN
            # We can shuffle the colors to add randomness
            colors_to_try = list(range(num_colors))
            rng.shuffle(colors_to_try)
            
            # Add ABSTAIN at the end
            # Actually, if we want to maximize, we should try colors first.
            # If all colors fail, try ABSTAIN.
            
            for c in colors_to_try:
                if c not in domains[e_idx]:
                    continue
                
                # Assign
                colors[e_idx] = c
                domains[e_idx].clear()
                domains[e_idx].add(c)
                
                # Forward Check
                valid = True
                # Store domains to restore
                # We need to restore domains of neighbors
                affected_edges = []
                
                for cyc_idx in edge_cycles[e_idx]:
                    cyc = cycles[cyc_idx]
                    # Find other edges in this cycle
                    others = [x for x in cyc if x != e_idx]
                    
                    # Check for conflicts among assigned edges in this cycle
                    assigned_colors = []
                    unassigned = []
                    for o in others:
                        if colors[o] != -1:
                            assigned_colors.append(colors[o])
                        else:
                            unassigned.append(o)
                    
                    # If we have 3 assigned colors (including current e_idx), check distinctness
                    if len(assigned_colors) == 3:
                        # e_idx is c, others are 3 assigned
                        if len(set(assigned_colors + [c])) != 4:
                            valid = False
                            break
                    elif len(assigned_colors) == 2:
                        # e_idx is c, 2 others assigned, 1 unassigned
                        # The unassigned one cannot be c, col1, col2
                        u = unassigned[0]
                        forbidden = {c} | set(assigned_colors)
                        # Remove forbidden from domains[u]
                        original_domain = domains[u].copy()
                        new_domain = domains[u] - forbidden
                        if not new_domain:
                            valid = False
                            break
                        domains[u] = new_domain
                        affected_edges.append((u, original_domain))
                    elif len(assigned_colors) == 1:
                        # e_idx is c, 1 other assigned, 2 unassigned
                        # The 2 unassigned cannot be c or the assigned color
                        assigned_c = assigned_colors[0]
                        forbidden = {c, assigned_c}
                        for u in unassigned:
                            original_domain = domains[u].copy()
                            new_domain = domains[u] - forbidden
                            if not new_domain:
                                valid = False
                                break
                            domains[u] = new_domain
                            affected_edges.append((u, original_domain))
                        if not valid:
                            break
                    elif len(assigned_colors) == 0:
                        # e_idx is c, 3 others unassigned
                        # The 3 others cannot be c
                        forbidden = {c}
                        for u in unassigned:
                            original_domain = domains[u].copy()
                            new_domain = domains[u] - forbidden
                            if not new_domain:
                                valid = False
                                break
                            domains[u] = new_domain
                            affected_edges.append((u, original_domain))
                        if not valid:
                            break
                    
                    if not valid:
                        break

                if valid:
                    # Recurse
                    if backtrack(depth + 1):
                        # We found a solution. 
                        # Do we want to keep searching?
                        # If we return True, we stop this branch.
                        # To find the BEST, we should continue searching other branches.
                        # However, backtracking is exponential.
                        # Let's return True to signal "solution found in this branch".
                        # The caller will continue trying other values for its variable.
                        pass 
                
                # Undo
                colors[e_idx] = -1
                domains[e_idx] = set(range(num_colors + 1)) # Restore initial domain?
                # Wait, the domain of e_idx might have been reduced by previous assignments.
                # We should save the domain before assignment.
                # Let's refactor to save/restore properly.
                
                # Restore affected domains
                for u, orig_dom in affected_edges:
                    domains[u] = orig_dom
            
            # Try ABSTAIN
            if -1 in domains[e_idx]:
                colors[e_idx] = -1
                domains[e_idx].clear()
                domains[e_idx].add(-1)
                
                # No forward check needed for ABSTAIN?
                # If an edge is ABSTAIN, it doesn't participate in 4-cycle constraints 
                # unless the cycle is "completed" by other edges?
                # No, a 4-cycle constraint only applies if ALL 4 edges are present.
                # If one is ABSTAIN, the constraint is inactive.
                # So assigning ABSTAIN never creates a conflict.
                # It only removes constraints.
                
                if backtrack(depth + 1):
                    pass
                    
                colors[e_idx] = -1
                # Restore domain
                # We need to know the domain before we cleared it.
                # This simple save/restore is tricky with the current structure.
                # Let's simplify: Don't use domains? 
                # Without domains, forward checking is weaker (only checks completed cycles).
                # Given time constraints, maybe simple backtracking with random restarts is safer.
                # Or, implement domains properly with a stack of changes.
                
            return False
        
        # Re-implement with proper domain management
        # To keep it simple and robust within the time limit, I will use a simpler backtracking
        # that only checks for immediate violations (completed cycles) and relies on restarts.
        # This is less efficient but much easier to implement correctly.
        
        def backtrack_simple(depth):
            nonlocal best_solution, best_edge_count
            
            if time.time() > deadline:
                return False
            
            if depth == num_edges_total:
                count = sum(1 for c in colors if c != -1)
                if count > best_edge_count:
                    best_edge_count = count
                    best_solution = colors[:]
                return False # Continue searching
                
            e_idx = order[depth]
            
            # Try colors
            colors_to_try = list(range(num_colors))
            rng.shuffle(colors_to_try)
            
            for c in colors_to_try:
                colors[e_idx] = c
                
                # Check conflicts
                valid = True
                for cyc_idx in edge_cycles[e_idx]:
                    cyc = cycles[cyc_idx]
                    # Check if all 4 edges in this cycle are assigned
                    if all(colors[x] != -1 for x in cyc):
                        cols = [colors[x] for x in cyc]
                        if len(set(cols)) != 4:
                            valid = False
                            break
                
                if valid:
                    backtrack_simple(depth + 1)
                
                colors[e_idx] = -1
            
            # Try ABSTAIN
            colors[e_idx] = -1
            backtrack_simple(depth + 1)
            colors[e_idx] = -1
            
            return False
        
        backtrack_simple(0)
        return best_edge_count

    # Run multiple seeds
    for seed in range(100):
        if time.time() > deadline - 2:
            break
        solve_csp(seed)
        
        # Early stop if we found a very good solution
        if best_edge_count > 170:
            break

    # Convert to list format
    result = []
    if best_solution is not None:
        for i, e in enumerate(all_edges):
            c = best_solution[i]
            if c != -1:
                result.append([e[0], e[1], int(c)])
    
    return result
# EVOLVE-BLOCK-END
