"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    N = 512
    nodes = list(range(1, N))  # 1..511
    n = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}
    
    # Weight: 1/(512-a), higher is better
    weights = [1.0 / (512.0 - v) for v in nodes]
    
    # Conflict graph: edge between i,j if gcd(nodes[i], nodes[j]) > 1
    # We want independent set in this graph = pairwise coprime set
    conflict = [0] * n  # bitmasks for conflicts
    
    for i in range(n):
        vi = nodes[i]
        for j in range(i + 1, n):
            vj = nodes[j]
            if gcd(vi, vj) > 1:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Sort nodes by weight descending for better pruning
    order = sorted(range(n), key=lambda i: -weights[i])
    rank = [0] * n
    for r, idx in enumerate(order):
        rank[idx] = r
    
    # Reorder nodes so that higher-weight nodes come first
    new_nodes = [nodes[i] for i in order]
    new_weights = [weights[i] for i in order]
    new_rank = list(range(n))  # new_rank[i] = i since we sorted by weight
    
    # Rebuild conflict in new order
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(order)}
    new_conflict = [0] * n
    for i in range(n):
        old_i = order[i]
        mask = conflict[old_i]
        new_mask = 0
        for j in range(n):
            if mask & (1 << j):
                new_j = old_to_new[j]
                new_mask |= (1 << new_j)
        new_conflict[i] = new_mask
    
    # Now we have the problem in weight-sorted order
    # Independent set: no two selected nodes have an edge between them
    # Use branch and bound with coloring for upper bound
    
    best_score = 0.0
    best_set = []
    
    # Greedy initial solution
    greedy_set = []
    used = [False] * n
    for i in range(n):
        can_add = True
        for j in range(i):
            if used[j] and (new_conflict[i] >> j) & 1:
                can_add = False
                break
        if can_add:
            greedy_set.append(i)
            used[i] = True
    
    greedy_score = sum(new_weights[i] for i in greedy_set)
    best_score = greedy_score
    best_set = list(greedy_set)
    
    # Coloring for upper bound: greedy coloring of the subgraph
    # Color i means we can pick at most one node from each color class
    # Upper bound = sum of max weight in each color class
    
    def greedy_color(candidates):
        """Color the subgraph induced by candidates. Returns list of color classes (each a bitmask)."""
        if not candidates:
            return []
        
        colors = []  # list of (color_mask, max_weight)
        uncolored = candidates
        
        while uncolored:
            # Pick the node with highest weight
            best_i = -1
            best_w = -1
            for i in range(n):
                if (uncolored >> i) & 1:
                    if new_weights[i] > best_w:
                        best_w = new_weights[i]
                        best_i = i
            
            if best_i < 0:
                break
            
            # Find all nodes in uncolored that are NOT adjacent to best_i
            # These can share a color with best_i
            color_mask = 0
            for i in range(n):
                if (uncolored >> i) & 1:
                    # Check if i is non-adjacent to best_i (or i == best_i)
                    if i == best_i or not ((new_conflict[best_i] >> i) & 1):
                        color_mask |= (1 << i)
            
            colors.append((color_mask, best_w))
            uncolored &= ~color_mask
        
        return colors
    
    # Upper bound from coloring
    def upper_bound(candidates):
        colors = greedy_color(candidates)
        return sum(max_w for _, max_w in colors)
    
    # Branch and bound
    # State: current selected set (as bitmask), remaining candidates (as bitmask)
    # We process in order of weight (already sorted)
    
    def branch_and_bound(selected_mask, remaining_mask, current_score):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        if not remaining_mask:
            if current_score > best_score:
                best_score = current_score
                best_set = [i for i in range(n) if (selected_mask >> i) & 1]
            return
        
        # Upper bound
        ub = current_score + upper_bound(remaining_mask)
        if ub <= best_score + 1e-12:
            return
        
        # Pick the next node to branch on
        # Choose the node with highest weight from remaining
        best_i = -1
        best_w = -1
        for i in range(n):
            if (remaining_mask >> i) & 1:
                if new_weights[i] > best_w:
                    best_w = new_weights[i]
                    best_i = i
        
        if best_i < 0:
            return
        
        # Branch 1: Include best_i
        # Check if best_i is compatible with all selected
        compatible = True
        for i in range(n):
            if (selected_mask >> i) & 1:
                if (new_conflict[best_i] >> i) & 1:
                    compatible = False
                    break
        
        if compatible:
            new_selected = selected_mask | (1 << best_i)
            # Remove best_i and its neighbors from remaining
            new_remaining = remaining_mask & ~(1 << best_i)
            # Remove neighbors of best_i
            for j in range(n):
                if (new_conflict[best_i] >> j) & 1:
                    new_remaining &= ~(1 << j)
            
            branch_and_bound(new_selected, new_remaining, current_score + new_weights[best_i])
        
        # Branch 2: Exclude best_i
        new_remaining = remaining_mask & ~(1 << best_i)
        branch_and_bound(selected_mask, new_remaining, current_score)
    
    # Run B&B with a time limit
    # First, try to improve with a more targeted approach
    # Use the greedy solution as starting point
    
    # Convert greedy set to bitmask
    greedy_mask = 0
    for i in greedy_set:
        greedy_mask |= (1 << i)
    
    # Remaining candidates: those not in greedy set and not conflicting with greedy set
    remaining = 0
    for i in range(n):
        if not (greedy_mask >> i) & 1:
            # Check if i conflicts with any in greedy set
            ok = True
            for j in range(n):
                if (greedy_mask >> j) & 1 and (new_conflict[i] >> j) & 1:
                    ok = False
                    break
            if ok:
                remaining |= (1 << i)
    
    # Now do local search: try swapping elements
    # For each element in greedy_set, try removing it and adding a better compatible set
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_mask = greedy_mask
        current_score_val = best_score
        
        for i in range(n):
            if not (current_mask >> i) & 1:
                continue
            
            # Try removing i
            new_mask = current_mask & ~(1 << i)
            new_score = current_score_val - new_weights[i]
            
            # Find all nodes compatible with new_mask
            candidates = 0
            for j in range(n):
                if not (current_mask >> j) & 1 and not (new_mask >> j) & 1:
                    ok = True
                    for k in range(n):
                        if (new_mask >> k) & 1 and (new_conflict[j] >> k) & 1:
                            ok = False
                            break
                    if ok:
                        candidates |= (1 << j)
            
            # Greedy fill from candidates
            fill = []
            for j in range(n):
                if (candidates >> j) & 1:
                    ok = True
                    for k in fill:
                        if (new_conflict[j] >> k) & 1:
                            ok = False
                            break
                    if ok:
                        fill.append(j)
            
            fill_score = sum(new_weights[j] for j in fill)
            if new_score + fill_score > current_score_val + 1e-9:
                current_mask = new_mask
                for j in fill:
                    current_mask |= (1 << j)
                current_score_val = new_score + fill_score
                greedy_mask = current_mask
                best_score = current_score_val
                best_set = [i for i in range(n) if (current_mask >> i) & 1]
                improved = True
                break
    
    # Now try B&B on the remaining structure for further improvement
    # Only if we have time
    if time.time() < deadline - 10:
        # Reset for B&B from current best
        best_score = current_score_val
        best_set = list(best_set)
        
        # B&B from empty selection, using current best as incumbent
        # But this might be too slow. Instead, do targeted B&B on swaps
        
        # Try adding one node at a time
        for i in range(n):
            if (current_mask >> i) & 1:
                continue
            
            # Check if i is compatible with current_mask
            ok = True
            for j in range(n):
                if (current_mask >> j) & 1 and (new_conflict[i] >> j) & 1:
                    ok = False
                    break
            
            if ok:
                # Try adding i and see if we can remove some lower-value nodes
                # This is complex; skip for now
                pass
        
        # Final polish: try all pairs of swaps
        for i in range(n):
            if not (current_mask >> i) & 1:
                continue
            for j in range(n):
                if (current_mask >> j) & 1:
                    continue
                
                # Check if j is compatible with current_mask minus i
                new_mask = current_mask & ~(1 << i)
                ok = True
                for k in range(n):
                    if (new_mask >> k) & 1 and (new_conflict[j] >> k) & 1:
                        ok = False
                        break
                
                if ok:
                    new_score = current_score_val - new_weights[i] + new_weights[j]
                    if new_score > current_score_val + 1e-9:
                        current_mask = new_mask | (1 << j)
                        current_score_val = new_score
                        best_score = current_score_val
                        best_set = [i for i in range(n) if (current_mask >> i) & 1]
    
    # Convert back to original node values
    result = [new_nodes[i] for i in best_set]
    return result
# EVOLVE-BLOCK-END
