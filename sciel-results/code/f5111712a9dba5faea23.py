import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s kill
    
    n = 20
    # Precompute all edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(all_edges)
    
    # Edge index for fast lookup
    edge_index = {}
    for idx, (i, j) in enumerate(all_edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
    
    # Precompute for each edge, the list of triples (i,j,k) that would be violated
    # if this edge is present. A triple (i,j,k) is violated if there exist 3 common
    # neighbors. Adding edge (u,v) can only affect triples containing u or v.
    
    # For efficiency, we'll use bitmasks
    # mask[i] is an integer where bit j is 1 if (i,j) is an edge
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_valid(masks):
        # Check all triples
        for i in range(n):
            for j in range(i+1, n):
                common_ij = masks[i] & masks[j]
                if popcount(common_ij) < 3:
                    continue
                for k in range(j+1, n):
                    if popcount(common_ij & masks[k]) >= 3:
                        return False
        return True
    
    def find_violation(masks):
        for i in range(n):
            for j in range(i+1, n):
                common_ij = masks[i] & masks[j]
                if popcount(common_ij) < 3:
                    continue
                for k in range(j+1, n):
                    common_ijk = common_ij & masks[k]
                    if popcount(common_ijk) >= 3:
                        return (i, j, k, common_ijk)
        return None
    
    def count_edges(masks):
        return sum(popcount(m) for m in masks) // 2
    
    def add_edge(masks, u, v):
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    
    def remove_edge(masks, u, v):
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
    
    best_edges = 0
    best_masks = [0] * n
    
    # Strategy: Start from various initializations and use local search with swaps
    # Key insight: We can try to add edges, and if that creates a violation,
    # we try to remove an edge from the violating structure instead of just reverting.
    
    def local_search(masks, edge_count):
        """Local search: try to add edges, use swaps to escape local optima."""
        nonlocal best_edges, best_masks
        nonlocal start_time, deadline
        
        # Phase 1: Try to add all possible edges
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (masks[i] & (1 << j)):
                    missing.append((i, j))
        
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            # Try adding (u,v)
            add_edge(masks, u, v)
            if is_valid(masks):
                edge_count += 1
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_masks = masks[:]
            else:
                # Revert
                remove_edge(masks, u, v)
        
        # Phase 2: Swap-based local search
        # Try to swap: remove edge (a,b) and add edge (c,d)
        # This can help escape local optima
        num_swaps = 50
        for _ in range(num_swaps):
            if time.time() > deadline - 3:
                break
            
            # Pick a random edge to remove
            present = []
            for i in range(n):
                for j in range(i+1, n):
                    if masks[i] & (1 << j):
                        present.append((i, j))
            
            if not present:
                break
            
            a, b = random.choice(present)
            
            # Pick a random missing edge to add
            missing2 = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (masks[i] & (1 << j)):
                        missing2.append((i, j))
            
            if not missing2:
                break
            
            c, d = random.choice(missing2)
            
            # Try the swap
            remove_edge(masks, a, b)
            add_edge(masks, c, d)
            
            if is_valid(masks):
                edge_count += 1  # -1 + 1 = net 0, but we're swapping
                # Actually edge count stays same: remove 1, add 1
                # So edge_count is unchanged
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_masks = masks[:]
            else:
                # Revert
                remove_edge(masks, c, d)
                add_edge(masks, a, b)
        
        return edge_count
    
    # Multiple restarts with different initializations
    while time.time() < deadline - 5:
        # Choose initialization strategy
        r = random.random()
        
        if r < 0.3:
            # Complete graph, then repair
            masks = [0] * n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        masks[i] |= (1 << j)
        elif r < 0.5:
            # Random dense graph
            masks = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.7:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        elif r < 0.7:
            # Complement of a random sparse graph
            masks = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.9:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        elif r < 0.85:
            # Circulant graph: connect i to i+1, i+2, i+3, i+4 mod 20
            masks = [0] * n
            for i in range(n):
                for d in range(1, 5):
                    j = (i + d) % n
                    if j > i:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        else:
            # Start from best found so far with perturbation
            masks = best_masks[:]
            # Remove a few random edges
            for _ in range(random.randint(3, 8)):
                present = []
                for i in range(n):
                    for j in range(i+1, n):
                        if masks[i] & (1 << j):
                            present.append((i, j))
                if present:
                    u, v = random.choice(present)
                    remove_edge(masks, u, v)
        
        # Repair phase: remove edges until valid
        max_repair_steps = 1000
        steps = 0
        while not is_valid(masks) and steps < max_repair_steps:
            if time.time() > deadline - 5:
                break
            v = find_violation(masks)
            if v is None:
                break
            i, j, k, common_mask = v
            # Get 3 vertices from common_mask
            neighbors = []
            m = common_mask
            while len(neighbors) < 3 and m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                neighbors.append(idx)
                m ^= lsb
            
            # Remove a random edge from the K3,3 structure
            u = random.choice([i, j, k])
            v_neighbor = random.choice(neighbors)
            remove_edge(masks, u, v_neighbor)
            steps += 1
        
        if is_valid(masks):
            edge_count = count_edges(masks)
            if edge_count > best_edges:
                best_edges = edge_count
                best_masks = masks[:]
            
            # Local search
            edge_count = local_search(masks, edge_count)
    
    # Final polish: try to add any remaining edges
    if time.time() < deadline - 1:
        masks = best_masks[:]
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (masks[i] & (1 << j)):
                    missing.append((i, j))
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 0.5:
                break
            add_edge(masks, u, v)
            if is_valid(masks):
                ec = count_edges(masks)
                if ec > best_edges:
                    best_edges = ec
                    best_masks = masks[:]
            else:
                remove_edge(masks, u, v)
    
    # Convert best_masks to edge list
    edges = []
    for i in range(n):
        m = best_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
    
    return edges
