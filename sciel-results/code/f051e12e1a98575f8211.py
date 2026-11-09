import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for the 90s kill
    
    N = 256
    
    # Precompute squares up to 2*N
    max_sum = 2 * N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter nodes that violate the diagonal condition (a=a => 2a is square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    
    # 2. Build conflict graph using adjacency lists
    # neighbors[i] is a set of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    neighbors = [set() for _ in range(M)]
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Calculate degrees
    degrees = [len(neighbors[i]) for i in range(M)]
    
    # Strategy 1: Initial greedy MIS with degree-based ordering
    def greedy_mis(order, neighbors):
        selected = []
        blocked = set()
        
        for idx in order:
            if idx in blocked:
                continue
            # Select idx
            selected.append(idx)
            # Block idx and all its neighbors
            blocked.add(idx)
            blocked.update(neighbors[idx])
            
        return selected
    
    # Order by degree descending (common heuristic for MIS)
    order_desc = sorted(range(M), key=lambda x: degrees[x], reverse=True)
    set_indices = greedy_mis(order_desc, neighbors)
    best_set = [valid_nodes[i] for i in set_indices]
    best_size = len(best_set)
    
    # Order by degree ascending
    order_asc = sorted(range(M), key=lambda x: degrees[x])
    set_indices = greedy_mis(order_asc, neighbors)
    if len(set_indices) > best_size:
        best_set = [valid_nodes[i] for i in set_indices]
        best_size = len(set_indices)
        
    # Strategy 2: Local Search (Add + Swap) to improve the greedy solution
    # We represent the set as a boolean array for fast lookup
    def local_search(indices, neighbors, valid_nodes, max_time):
        selected = set(indices)
        size = len(selected)
        
        # Precompute conflict sets for faster checking? 
        # neighbors[i] is already a set.
        
        # Iterative improvement
        improved = True
        while improved:
            if time.time() > max_time:
                break
            improved = False
            
            # 1. Try to add a node
            # Sort candidates by degree (ascending) to prioritize "safe" nodes first?
            # Or just iterate.
            candidates = [i for i in range(M) if i not in selected]
            # Heuristic: try nodes with fewer conflicts with current set first?
            # Cost of adding i is 0 if no conflicts, but it blocks neighbors.
            # Actually, we can only add i if no neighbor of i is in selected.
            
            add_candidates = []
            for i in candidates:
                # Check if any neighbor is in selected
                has_conflict = False
                for nb in neighbors[i]:
                    if nb in selected:
                        has_conflict = True
                        break
                if not has_conflict:
                    # Potential add. Score by how many neighbors it blocks?
                    # Blocking more is worse. So prefer nodes with fewer neighbors in candidates?
                    # Or just add the one that blocks the fewest *potential* future adds?
                    # Simple heuristic: add node with minimum degree.
                    add_candidates.append((degrees[i], i))
            
            if add_candidates:
                add_candidates.sort()
                _, best_add = add_candidates[0]
                selected.add(best_add)
                size += 1
                improved = True
                continue # Re-evaluate immediately as adding one node might enable/disable others
            
            # 2. Try to swap a node in with a node out
            # For each node in selected, try removing it and adding a neighbor it was blocking?
            # Or try swapping i in selected with j not in selected if it increases size?
            # Size stays same in a swap. So swap is only useful if it enables an ADD later.
            # This is complex. A simpler local search is 1-opt (add) and maybe 2-opt (swap+add).
            
            # Let's try a simple swap: 
            # Remove i, add j. 
            # Benefit: j might have fewer conflicts than i, allowing more adds later.
            # This is hard to evaluate greedily.
            
            # Alternative: Random restarts with local search.
            # Since we are stuck in "no add possible", we can try to randomly remove a node 
            # and then try to add nodes again.
            
            # Let's try removing a random node and see if we can recover size.
            if selected:
                # Remove a random node
                remove_idx = random.choice(list(selected))
                selected.remove(remove_idx)
                size -= 1
                improved = True # Mark as changed to continue loop
                # We don't break, we let the next iteration try to add.
                # If we can't add back, we've lost size. 
                # To prevent getting worse, we should only do this if we have a backup or 
                # if we can guarantee recovery? 
                # Actually, standard local search for MIS often uses "kick" moves.
                # Let's be careful. If we remove and can't add, we have a smaller set.
                # We should keep track of best set found so far in this local search run.
                
                # Better approach for swap:
                # Iterate over all pairs (i in selected, j not in selected) where i and j are NOT neighbors?
                # No, if they are not neighbors, swapping doesn't change validity directly, 
                # but changes the "blocking" pattern.
                
                # Given time constraints, let's stick to "Add" and "Random Kick" (remove 1, try add).
                # We accept the kick only if we can restore size within a few steps?
                # Or just accept it and let the "Add" phase try to fix it.
                # If we end up with smaller size, we return the best found.
                
                # Let's modify the loop to track best_size_local.
                # But the current structure returns immediately.
                # Let's refine:
                pass
        
        return list(selected)

    # The above local search is a bit loose. Let's implement a more robust one.
    
    def robust_local_search(initial_indices, neighbors, valid_nodes, max_time):
        selected = set(initial_indices)
        best_selected = set(selected)
        best_size = len(selected)
        
        # 1. Add phase
        while True:
            if time.time() > max_time:
                break
            added = False
            # Find any node that can be added
            # To speed up, we can maintain a list of "free" nodes, but M is small (~200).
            # Check all non-selected nodes.
            for i in range(M):
                if i in selected:
                    continue
                # Check conflicts
                can_add = True
                for nb in neighbors[i]:
                    if nb in selected:
                        can_add = False
                        break
                if can_add:
                    selected.add(i)
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    added = True
                    break # Restart scan after add
            if not added:
                break
        
        # 2. Swap/Kick phase
        # Try to swap a node in selected with a node not in selected to potentially enable more adds.
        # We do this by removing a random node, then running the Add phase, and keeping the result
        # only if it's larger or equal? No, if it's smaller, we revert.
        
        # Let's try a limited number of kicks.
        num_kicks = 100
        for _ in range(num_kicks):
            if time.time() > max_time:
                break
            if not selected:
                break
            
            # Save current state
            current_selected = set(selected)
            current_size = len(selected)
            
            # Remove a random node
            remove_idx = random.choice(list(current_selected))
            selected.remove(remove_idx)
            
            # Run Add phase
            while True:
                if time.time() > max_time:
                    break
                added = False
                for i in range(M):
                    if i in selected:
                        continue
                    can_add = True
                    for nb in neighbors[i]:
                        if nb in selected:
                            can_add = False
                            break
                    if can_add:
                        selected.add(i)
                        added = True
                        break
                if not added:
                    break
            
            # Check if we improved
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)
            elif len(selected) < current_size:
                # Revert
                selected = current_selected
            # If equal, keep the new configuration as it might be different and allow future kicks
            # But to be safe, we can also just keep it.
            
        return list(best_selected)

    # Run local search on the best greedy result so far
    if time.time() < deadline - 5:
        improved_indices = robust_local_search(best_set_indices if 'best_set_indices' in dir() else [i for i in range(M) if valid_nodes[i] in best_set], neighbors, valid_nodes, deadline - 5)
        if len(improved_indices) > best_size:
            best_set = [valid_nodes[i] for i in improved_indices]
            best_size = len(improved_indices)

    # Strategy 3: Randomized Greedy Restarts
    # We try many random orderings.
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random order
        random_order = list(range(M))
        random.shuffle(random_order)
        
        set_indices = greedy_mis(random_order, neighbors)
        
        # Apply local search to this result
        if time.time() < deadline - 1:
            improved_indices = robust_local_search(set_indices, neighbors, valid_nodes, deadline - 1)
            if len(improved_indices) > best_size:
                best_set = [valid_nodes[i] for i in improved_indices]
                best_size = len(improved_indices)
        else:
            if len(set_indices) > best_size:
                best_set = [valid_nodes[i] for i in set_indices]
                best_size = len(set_indices)

    # Ensure non-empty
    if not best_set:
        if valid_nodes:
            best_set = [valid_nodes[0]]
        else:
            best_set = [1]
            
    return best_set

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
