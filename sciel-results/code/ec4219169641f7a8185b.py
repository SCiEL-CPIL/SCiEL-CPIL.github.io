import random
import time
import numpy as np
from collections import defaultdict

def construct():
    d = 7
    n = 2 ** d
    num_edges = n * d // 2  # 448
    
    # 1. Precompute edges and their indices
    # Edge index mapping: (u, v) -> idx
    # We'll just generate edges and assign indices 0..447
    edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    idx = 0
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                idx_to_edge[idx] = (x, y)
                idx += 1
                
    # 2. Precompute C4s
    # A C4 is defined by two distinct dimensions i, j and a base vertex x
    # such that bits i and j are 0 in x (or we just iterate all x and mask out duplicates?)
    # Standard way: Iterate all x in 0..127. For each pair of dims (i, j) with i < j:
    # The square is x, x^i, x^j, x^i^j.
    # To avoid duplicates, we can require that x has 0 in bit i and 0 in bit j.
    # Then the square is unique.
    c4s = []
    edge_to_c4s = [[] for _ in range(num_edges)]
    c4_edge_sets = []  # Store as frozenset for faster intersection
    
    for x in range(n):
        for i in range(d):
            if x & (1 << i):
                continue
            for j in range(i + 1, d):
                if x & (1 << j):
                    continue
                
                # Vertices of the square
                v0 = x
                v1 = x ^ (1 << i)
                v2 = x ^ (1 << j)
                v3 = x ^ (1 << i) ^ (1 << j)
                
                # Edges of the square
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, v0), max(v2, v0))
                
                c4_edges = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                c4s.append(c4_edges)
                c4_edge_sets.append(frozenset(c4_edges))
                
                for e_idx in c4_edges:
                    edge_to_c4s[e_idx].append(len(c4s) - 1)
    
    num_c4s = len(c4s)
    
    # Precompute for each edge, the list of other edges that share a C4 with it
    # This helps with swap operations
    edge_conflicts = [set() for _ in range(num_edges)]
    for e1_idx in range(num_edges):
        for c4_idx in edge_to_c4s[e1_idx]:
            for e2_idx in c4s[c4_idx]:
                if e2_idx != e1_idx:
                    edge_conflicts[e1_idx].add(e2_idx)
    
    # Data structures for fast updates
    selected = np.zeros(num_edges, dtype=np.bool_)
    c4_counts = np.zeros(num_c4s, dtype=np.int8)
    
    def add_edge(e_idx):
        selected[e_idx] = True
        for c4_idx in edge_to_c4s[e_idx]:
            c4_counts[c4_idx] += 1
            
    def remove_edge(e_idx):
        selected[e_idx] = False
        for c4_idx in edge_to_c4s[e_idx]:
            c4_counts[c4_idx] -= 1
            
    def can_add(e_idx):
        # Check if adding e_idx would create a C4 with 4 edges
        # i.e., if any c4 containing e_idx already has 3 edges selected
        for c4_idx in edge_to_c4s[e_idx]:
            if c4_counts[c4_idx] >= 3:
                return False
        return True
    
    def get_addable_edges():
        """Return list of all edges that can currently be added"""
        addable = []
        for e in range(num_edges):
            if not selected[e] and can_add(e):
                addable.append(e)
        return addable
    
    def count_addable():
        """Count how many edges can be added"""
        count = 0
        for e in range(num_edges):
            if not selected[e] and can_add(e):
                count += 1
        return count

    best_count = 0
    best_selected = selected.copy()
    
    deadline = time.time() + 110
    
    # Greedy initialization with multiple strategies
    def greedy_init(strategy=0):
        nonlocal selected, c4_counts, best_count, best_selected
        selected[:] = False
        c4_counts[:] = 0
        count = 0
        
        if strategy == 0:
            # Random order
            order = list(range(num_edges))
            random.shuffle(order)
        elif strategy == 1:
            # Dimension-grouped order: process all edges of dim 0, then dim 1, etc.
            order = []
            for dim in range(d):
                dim_edges = []
                for e_idx in range(num_edges):
                    u, v = idx_to_edge[e_idx]
                    if (u ^ v) == (1 << dim):
                        dim_edges.append(e_idx)
                random.shuffle(dim_edges)
                order.extend(dim_edges)
        elif strategy == 2:
            # Vertex-grouped order: process all edges incident to vertex 0, then vertex 1, etc.
            order = []
            for v in range(n):
                v_edges = []
                for e_idx in range(num_edges):
                    u, w = idx_to_edge[e_idx]
                    if u == v or w == v:
                        v_edges.append(e_idx)
                random.shuffle(v_edges)
                order.extend(v_edges)
        else:
            order = list(range(num_edges))
            random.shuffle(order)
        
        for e in order:
            if can_add(e):
                add_edge(e)
                count += 1
        
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
        return count
    
    def greedy_fill():
        """Try to add as many edges as possible in random order"""
        nonlocal selected, c4_counts, best_count, best_selected
        count = np.sum(selected)
        order = list(range(num_edges))
        random.shuffle(order)
        for e in order:
            if not selected[e] and can_add(e):
                add_edge(e)
                count += 1
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
        return count

    # Local Search with improved strategies
    def local_search(current_count):
        nonlocal selected, c4_counts, best_count, best_selected
        count = current_count
        no_improve_steps = 0
        max_no_improve = 3000
        last_best_update = time.time()
        
        while time.time() < deadline - 5: # Leave 5s for final cleanup
            action = random.random()
            
            if action < 0.3:
                # Try Add: pick random non-selected edge
                e = random.randint(0, num_edges - 1)
                if not selected[e]:
                    if can_add(e):
                        add_edge(e)
                        count += 1
                        no_improve_steps = 0
                        if count > best_count:
                            best_count = count
                            best_selected = selected.copy()
                            last_best_update = time.time()
                    else:
                        no_improve_steps += 1
            elif action < 0.6:
                # Try targeted swap: remove an edge that blocks many additions, add a better one
                # Find edges whose removal would allow the most new additions
                best_swap = None
                best_gain = 0
                # Sample some selected edges to find good swap candidates
                for _ in range(10):
                    e1 = random.randint(0, num_edges - 1)
                    if selected[e1]:
                        # Try removing e1
                        remove_edge(e1)
                        # Count how many edges can now be added
                        gain = 0
                        for e2 in edge_conflicts[e1]:
                            if not selected[e2] and can_add(e2):
                                gain += 1
                        # Also check other edges
                        for _ in range(20):
                            e2 = random.randint(0, num_edges - 1)
                            if not selected[e2] and e2 != e1 and can_add(e2):
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (e1, gain)
                        add_edge(e1)  # Restore
                        if best_gain >= 3:
                            break
                
                if best_swap and best_gain > 0:
                    e1 = best_swap[0]
                    remove_edge(e1)
                    # Try to add multiple edges
                    added_any = False
                    order = list(range(num_edges))
                    random.shuffle(order)
                    for e2 in order:
                        if not selected[e2] and e2 != e1 and can_add(e2):
                            add_edge(e2)
                            count += 1
                            added_any = True
                            break
                    if not added_any:
                        add_edge(e1)
                    no_improve_steps += 1
            elif action < 0.8:
                # Try 2-swap: remove two edges, add one or two
                e1 = random.randint(0, num_edges - 1)
                if selected[e1]:
                    e2 = random.randint(0, num_edges - 1)
                    if selected[e2] and e2 != e1:
                        remove_edge(e1)
                        remove_edge(e2)
                        count -= 2
                        # Try to add up to 2 edges
                        added = 0
                        order = list(range(num_edges))
                        random.shuffle(order)
                        for e3 in order:
                            if not selected[e3] and e3 != e1 and e3 != e2 and can_add(e3):
                                add_edge(e3)
                                count += 1
                                added += 1
                                if added == 2:
                                    break
                        no_improve_steps += 1
            else:
                # Simple swap
                e1 = random.randint(0, num_edges - 1)
                if selected[e1]:
                    e2 = random.randint(0, num_edges - 1)
                    if not selected[e2]:
                        remove_edge(e1)
                        if can_add(e2):
                            add_edge(e2)
                            no_improve_steps += 1
                        else:
                            add_edge(e1)
                            no_improve_steps += 1
                            
            # Occasional Kick: Remove some random edges and greedily re-add
            if no_improve_steps > max_no_improve or (time.time() - last_best_update > 4 and random.random() < 0.15):
                # Smarter kick: remove edges from a random dimension or region
                kick_strategy = random.randint(0, 2)
                to_remove = []
                
                if kick_strategy == 0:
                    # Remove all edges of a random dimension
                    dim = random.randint(0, d - 1)
                    for e_idx in range(num_edges):
                        u, v = idx_to_edge[e_idx]
                        if (u ^ v) == (1 << dim) and selected[e_idx]:
                            to_remove.append(e_idx)
                elif kick_strategy == 1:
                    # Remove edges incident to a random set of vertices
                    for _ in range(random.randint(3, 8)):
                        v = random.randint(0, n - 1)
                        for e_idx in range(num_edges):
                            u, w = idx_to_edge[e_idx]
                            if (u == v or w == v) and selected[e_idx] and e_idx not in to_remove:
                                to_remove.append(e_idx)
                else:
                    # Random removal
                    kick_size = random.randint(5, 15)
                    attempts = 0
                    while len(to_remove) < kick_size and attempts < 100:
                        e = random.randint(0, num_edges - 1)
                        if selected[e] and e not in to_remove:
                            to_remove.append(e)
                        attempts += 1
                
                for e in to_remove:
                    remove_edge(e)
                    count -= 1
                
                # Greedy re-add with multiple passes
                for _ in range(2):
                    order = list(range(num_edges))
                    random.shuffle(order)
                    for e in order:
                        if not selected[e] and can_add(e):
                            add_edge(e)
                            count += 1
                        
                no_improve_steps = 0
                if count > best_count:
                    best_count = count
                    best_selected = selected.copy()
                    last_best_update = time.time()

    # Main Loop
    # Run multiple restarts with different strategies
    strategy = 0
    while time.time() < deadline - 10:
        # Greedy init with rotating strategies
        curr_count = greedy_init(strategy % 3)
        strategy += 1
        
        # Local search
        local_search(curr_count)
        
        # If we are close to the end, just keep polishing the best found so far?
        # Or just let the loop continue. The local_search will exit when deadline - 5.
        # We need to ensure we don't start a new greedy init if time is low.
        if time.time() > deadline - 15:
            break

    # Final polish: Try to add any remaining edges to best_selected?
    # The local search already tries to add. But let's do one final greedy pass on the best solution.
    # Actually, best_selected is a snapshot. We should restore it and try to improve it one last time.
    selected[:] = best_selected
    c4_counts[:] = 0
    for i in range(num_edges):
        if selected[i]:
            add_edge(i) # This is O(N * 6) which is fine.
    
    # Final intensive polish: multiple passes with different orders
    for pass_num in range(5):
        if time.time() > deadline - 2:
            break
        order = list(range(num_edges))
        random.shuffle(order)
        improved = False
        for e in order:
            if not selected[e]:
                if can_add(e):
                    add_edge(e)
                    best_count += 1
                    improved = True
        if not improved:
            break
    
    # Try targeted swaps in final polish
    if time.time() < deadline - 1:
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            # Try to find a swap that improves
            e1 = random.randint(0, num_edges - 1)
            if selected[e1]:
                # Try removing e1 and see if we can add more
                remove_edge(e1)
                added = 0
                for e2 in range(num_edges):
                    if not selected[e2] and e2 != e1 and can_add(e2):
                        add_edge(e2)
                        best_count += 1
                        added += 1
                        break
                if added == 0:
                    add_edge(e1)

    # Convert back to list of [u, v]
    result = []
    for i in range(num_edges):
        if selected[i]:
            result.append(list(idx_to_edge[i]))
            
    return result
