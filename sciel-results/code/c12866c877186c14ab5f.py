"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # 1. Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                
    m = len(all_edges)  # 448
    
    # 2. Precompute all C4s (faces)
    # A C4 is defined by a vertex x and two distinct directions i, j.
    # Vertices: x, x^i, x^j, x^i^j
    # Edges: (x, x^i), (x^j, x^i^j), (x, x^j), (x^i, x^i^j)
    c4s = []
    seen_c4s = set()
    
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v3, v4), max(v3, v4))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v4), max(v2, v4))
                
                idx1 = edge_to_idx.get(e1)
                idx2 = edge_to_idx.get(e2)
                idx3 = edge_to_idx.get(e3)
                idx4 = edge_to_idx.get(e4)
                
                if None not in (idx1, idx2, idx3, idx4):
                    c4 = tuple(sorted([idx1, idx2, idx3, idx4]))
                    if c4 not in seen_c4s:
                        seen_c4s.add(c4)
                        c4s.append(c4)
                        
    num_c4s = len(c4s)
    
    # 3. Build incidence structures
    # edge_to_c4s[edge_idx] = list of c4 indices
    edge_to_c4s = [[] for _ in range(m)]
    # c4_to_edges[c4_idx] = list of 4 edge indices
    c4_to_edges = [list(c4) for c4 in c4s]
    
    for c4_idx, c4 in enumerate(c4s):
        for e in c4:
            edge_to_c4s[e].append(c4_idx)
            
    # 4. Fast Heuristic Initialization
    # Start with a random subset or greedy construction.
    # Greedy: Iterate edges, add if it doesn't complete a C4.
    # This is O(m * deg) which is fast.
    selected = np.zeros(m, dtype=bool)
    c4_count = np.zeros(num_c4s, dtype=int)
    
    # Randomize edge order for diversity
    edge_order = list(range(m))
    random.shuffle(edge_order)
    
    for e in edge_order:
        # Check if adding e completes any C4
        can_add = True
        for c4_idx in edge_to_c4s[e]:
            if c4_count[c4_idx] == 3:
                can_add = False
                break
        if can_add:
            selected[e] = True
            for c4_idx in edge_to_c4s[e]:
                c4_count[c4_idx] += 1
                
    # Calculate blocked counts
    blocked = np.zeros(m, dtype=int)
    for e in range(m):
        if not selected[e]:
            for c4_idx in edge_to_c4s[e]:
                if c4_count[c4_idx] == 3:
                    blocked[e] += 1
                    
    current_size = np.sum(selected)
    best_size = current_size
    best_selected = selected.copy()
    
    # 5. Optimized Local Search
    # We use a simple hill-climbing with occasional swaps to escape local optima.
    # To speed up, we inline the state update logic.
    
    steps = 0
    max_steps = 500000  # Hard cap on iterations to prevent infinite loops if time check fails
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        e = random.randint(0, m - 1)
        
        if selected[e]:
            # Try removing edge e.
            # We accept removal if it allows adding more edges later, or just random walk.
            # For simplicity, let's do a 2-opt style swap if possible, or just remove with low prob.
            # Actually, a better strategy: 
            # 1. If we can add an unselected edge, do it.
            # 2. Else, try a swap: remove one selected, add one unselected.
            
            # This branch is rarely used for improvement directly, 
            # but we need to move around the space.
            # Let's stick to: 
            # - If e is selected, do nothing (or maybe remove with very low prob to perturb).
            # - If e is unselected, try to add. If blocked, try swap.
            
            # To allow exploration, we occasionally remove a random selected edge.
            if random.random() < 0.05:
                # Remove e
                selected[e] = False
                current_size -= 1
                # Update state for removal
                for c4_idx in edge_to_c4s[e]:
                    old_count = c4_count[c4_idx]
                    c4_count[c4_idx] = old_count - 1
                    new_count = c4_count[c4_idx]
                    if new_count == 2 and old_count == 3:
                        # Edges in this C4 that are unselected are no longer blocked by this C4
                        for other_e in c4_to_edges[c4_idx]:
                            if other_e != e and not selected[other_e]:
                                blocked[other_e] -= 1
        else:
            # e is unselected
            if blocked[e] == 0:
                # Add e
                selected[e] = True
                current_size += 1
                # Update state for addition
                for c4_idx in edge_to_c4s[e]:
                    old_count = c4_count[c4_idx]
                    c4_count[c4_idx] = old_count + 1
                    new_count = c4_count[c4_idx]
                    if old_count == 2 and new_count == 3:
                        # Edges in this C4 that are unselected become blocked by this C4
                        for other_e in c4_to_edges[c4_idx]:
                            if other_e != e and not selected[other_e]:
                                blocked[other_e] += 1
                                
                if current_size > best_size:
                    best_size = current_size
                    best_selected = selected.copy()
            else:
                # Try swap
                # Find a blocking C4
                # We pick a random C4 from edge_to_c4s[e] that has count 3
                # Note: blocked[e] > 0 implies at least one such C4 exists.
                blocking_c4_found = False
                # To avoid list creation, just pick one at random from the list and check
                # Since degree is small, iterating is fine.
                c4_indices = edge_to_c4s[e]
                # Shuffle indices to pick random blocking one
                random.shuffle(c4_indices)
                
                for c4_idx in c4_indices:
                    if c4_count[c4_idx] == 3:
                        blocking_c4_found = True
                        break
                        
                if blocking_c4_found:
                    # Find a selected edge v in this C4 to remove
                    # C4 has 3 selected, 1 unselected (e)
                    c4_edges = c4_to_edges[c4_idx]
                    # Pick a random selected edge from these 4
                    # We know e is unselected, so pick from the other 3
                    selected_in_c4 = [oe for oe in c4_edges if selected[oe]]
                    if len(selected_in_c4) == 3:
                        v = random.choice(selected_in_c4)
                        
                        # Perform Swap: Remove v, Add e
                        # 1. Remove v
                        selected[v] = False
                        # current_size stays same (remove 1, add 1)
                        
                        # Update state for removal of v
                        for c4_idx_v in edge_to_c4s[v]:
                            old_count = c4_count[c4_idx_v]
                            c4_count[c4_idx_v] = old_count - 1
                            new_count = c4_count[c4_idx_v]
                            if new_count == 2 and old_count == 3:
                                for other_e in c4_to_edges[c4_idx_v]:
                                    if other_e != v and not selected[other_e]:
                                        blocked[other_e] -= 1
                                        
                        # 2. Check if e can be added now
                        if blocked[e] == 0:
                            # Add e
                            selected[e] = True
                            
                            # Update state for addition of e
                            for c4_idx_e in edge_to_c4s[e]:
                                old_count = c4_count[c4_idx_e]
                                c4_count[c4_idx_e] = old_count + 1
                                new_count = c4_count[c4_idx_e]
                                if old_count == 2 and new_count == 3:
                                    for other_e in c4_to_edges[c4_idx_e]:
                                        if other_e != e and not selected[other_e]:
                                            blocked[other_e] += 1
                                            # Note: e itself is selected, so it's not updated here.
                                            # But wait, if e was blocking something? 
                                            # e is now selected, so it doesn't block itself.
                                            # It might block other edges in C4s where count becomes 3.
                                            # This is handled.
                            
                            # Swap successful. Size unchanged.
                            # We might have created new opportunities for other edges.
                        else:
                            # Revert: Add v back
                            selected[v] = True
                            
                            # Update state for addition of v (revert)
                            for c4_idx_v in edge_to_c4s[v]:
                                old_count = c4_count[c4_idx_v]
                                c4_count[c4_idx_v] = old_count + 1
                                new_count = c4_count[c4_idx_v]
                                if old_count == 2 and new_count == 3:
                                    for other_e in c4_to_edges[c4_idx_v]:
                                        if other_e != v and not selected[other_e]:
                                            blocked[other_e] += 1

    # Final greedy polish
    changed = True
    while changed:
        changed = False
        for i in range(m):
            if not selected[i] and blocked[i] == 0:
                selected[i] = True
                current_size += 1
                changed = True
                for c4_idx in edge_to_c4s[i]:
                    old_count = c4_count[c4_idx]
                    c4_count[c4_idx] = old_count + 1
                    if old_count == 2:
                        for other_e in c4_to_edges[c4_idx]:
                            if other_e != i and not selected[other_e]:
                                blocked[other_e] += 1

    if current_size > best_size:
        best_size = current_size
        best_selected = selected.copy()
        
    # Convert to output
    out = []
    for idx in range(m):
        if best_selected[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
            
    return out

# EVOLVE-BLOCK-END
