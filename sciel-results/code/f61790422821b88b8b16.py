"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable_fast(n, edges):
    """Fast 2-colorability check using bitmask enumeration for n <= 16."""
    if not edges:
        return True
    
    if n <= 16:
        # Precompute for each edge the bitmask of vertices
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # For each coloring (bitmask), check if all edges are non-monochromatic
        # An edge is monochromatic if all 5 bits are 0 or all 5 bits are 1
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                # Count set bits in em & mask and em & ~mask
                # If all 5 vertices same color, edge is monochromatic
                # em & mask gives vertices colored 1
                # em & (~mask) & ((1<<n)-1) gives vertices colored 0
                ones = bin(em & mask).count('1')
                zeros = bin(em & (~mask) & ((1 << n) - 1)).count('1')
                if ones == 5 or zeros == 5:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        # Backtracking with unit propagation for larger n
        color = [-1] * n
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for edge in edges:
                    c0 = sum(1 for v in edge if color[v] == 0)
                    c1 = sum(1 for v in edge if color[v] == 1)
                    un = sum(1 for v in edge if color[v] == -1)
                    
                    if c0 == 5 or c1 == 5:
                        return False
                    if c0 == 4 and un == 1:
                        for v in edge:
                            if color[v] == -1:
                                color[v] = 1
                                changed = True
                    elif c1 == 4 and un == 1:
                        for v in edge:
                            if color[v] == -1:
                                color[v] = 0
                                changed = True
            return True
        
        def backtrack(idx):
            if idx >= n:
                return True
            for c in [0, 1]:
                color[idx] = c
                if propagate():
                    if backtrack(idx + 1):
                        return True
                color[idx] = -1
            return False
        
        return backtrack(0)

def _try_remove_edges(n, edges, deadline, max_iterations=100000):
    """Greedily remove edges while maintaining non-2-colorability."""
    edges = list(edges)
    i = 0
    iterations = 0
    while i < len(edges) and time.time() < deadline and iterations < max_iterations:
        iterations += 1
        e = edges[i]
        test_edges = edges[:i] + edges[i+1:]
        if not _is_2_colorable_fast(n, test_edges):
            edges = test_edges
            # Don't increment i, try next edge
        else:
            i += 1
    return edges

def _local_search(n, edges, deadline, max_steps=5000):
    """Local search: try swapping edges to find smaller non-2-colorable subgraphs."""
    edges = list(edges)
    all_possible = list(combinations(range(n), 5))
    edge_set = set(edges)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Try removing a random edge
        if edges:
            idx = random.randrange(len(edges))
            e = edges[idx]
            test_edges = edges[:idx] + edges[idx+1:]
            if not _is_2_colorable_fast(n, test_edges):
                edges = test_edges
                edge_set.discard(e)
                continue
        
        # Try swapping: remove one edge, add another
        if len(edges) > 0 and len(edge_set) < len(all_possible):
            remove_idx = random.randrange(len(edges))
            e_remove = edges[remove_idx]
            
            # Find candidate edges to add
            candidates = [e for e in all_possible if e not in edge_set]
            if candidates:
                e_add = random.choice(candidates)
                test_edges = edges[:remove_idx] + edges[remove_idx+1:] + [list(e_add)]
                if not _is_2_colorable_fast(n, test_edges):
                    edges = test_edges
                    edge_set.discard(e_remove)
                    edge_set.add(e_add)
    
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_n = 8
    best_edges = None
    best_len = float('inf')
    
    # Start with K_8^5 (56 edges) - known non-2-colorable
    n = 8
    all_edges_8 = [list(e) for e in combinations(range(n), 5)]
    
    # First pass: greedy removal from K_8^5
    edges = _try_remove_edges(n, all_edges_8, deadline)
    if not _is_2_colorable_fast(n, edges):
        best_edges = edges[:]
        best_len = len(edges)
        best_n = n
    
    if time.time() > deadline - 20:
        return {'n': best_n, 'edges': best_edges}
    
    # Second pass: local search to find better solutions
    edges = _local_search(n, best_edges, deadline, max_steps=3000)
    if not _is_2_colorable_fast(n, edges) and len(edges) < best_len:
        best_edges = edges[:]
        best_len = len(edges)
        best_n = n
    
    if time.time() > deadline - 10:
        return {'n': best_n, 'edges': best_edges}
    
    # Try with n=9: start from a random subset and minimize
    n9 = 9
    all_edges_9 = list(combinations(range(n9), 5))
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        # Start with a random subset of ~60 edges
        initial_size = min(60, len(all_edges_9))
        random_edges = random.sample(all_edges_9, initial_size)
        edges9 = [list(e) for e in random_edges]
        
        # Ensure non-2-colorable by adding edges if needed
        while _is_2_colorable_fast(n9, edges9) and time.time() < deadline - 5:
            remaining = [e for e in all_edges_9 if e not in set(map(tuple, edges9))]
            if not remaining:
                break
            edges9.append(list(random.choice(remaining)))
        
        if not _is_2_colorable_fast(n9, edges9):
            # Minimize
            edges9 = _try_remove_edges(n9, edges9, deadline - 5)
            if not _is_2_colorable_fast(n9, edges9) and len(edges9) < best_len:
                best_edges = edges9[:]
                best_len = len(edges9)
                best_n = n9
    
    if time.time() > deadline - 5:
        return {'n': best_n, 'edges': best_edges}
    
    # Try n=7 (if possible - C(7,5)=21 edges max)
    n7 = 7
    all_edges_7 = list(combinations(range(n7), 5))
    # K_7^5 has 21 edges. Is it non-2-colorable? Let's check.
    if not _is_2_colorable_fast(n7, [list(e) for e in all_edges_7]):
        edges7 = _try_remove_edges(n7, [list(e) for e in all_edges_7], deadline - 2)
        if not _is_2_colorable_fast(n7, edges7) and len(edges7) < best_len:
            best_edges = edges7[:]
            best_len = len(edges7)
            best_n = n7
    
    # Final verification
    if best_edges is None:
        # Fallback: return K_8^5
        best_n = 8
        best_edges = [list(e) for e in combinations(range(8), 5)]
    
    # Ensure all edges are lists of ints
    best_edges = [list(e) for e in best_edges]
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
