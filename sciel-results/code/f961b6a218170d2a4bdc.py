import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    d = 7
    n_vertices = 1 << d
    total_edges = n_vertices * d // 2
    
    # Precompute all edges and their indices
    all_edges = []
    edge_to_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_index[(x, y)] = idx
                edge_to_index[(y, x)] = idx

    # Precompute for each edge, the other 3 edges that form a C4 with it
    # A C4 is formed by edges (u,v), (u',v'), (u,u'), (v,v') where u^u'=v^v'
    # For an edge (u,v) with dimension i, the other edges are:
    # For each j != i:
    #   c = u ^ (1<<j), e = v ^ (1<<j)
    #   Edges: (u,c), (v,e), (c,e)
    # If any of these 3 are present, adding (u,v) creates a C4.
    
    # Actually, it's more efficient to track counts of "potential C4s" or use a different representation.
    # Let's use a set of selected edges for fast lookup.
    
    def is_valid_addition(edge, selected_set):
        u, v = edge
        # Determine the dimension of the edge
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            c = u ^ bit_j
            e = v ^ bit_j
            
            # Check the 3 other edges
            # (u, c)
            if (min(u, c), max(u, c)) in selected_set:
                continue # This is not a violation by itself, we need all 3? 
                # Wait, the condition is "No 2-dimensional face may have all four edges selected".
                # So if I add (u,v), I create a C4 if the other 3 edges of that face are ALREADY selected.
                # The 3 edges are (u,c), (v,e), (c,e).
                # If ALL 3 are in selected_set, then adding (u,v) completes the C4.
                # My previous logic was wrong. I need to check if ALL 3 are present.
                pass
            
            e1 = (min(u, c), max(u, c))
            e2 = (min(v, e), max(v, e))
            e3 = (min(c, e), max(c, e))
            
            if e1 in selected_set and e2 in selected_set and e3 in selected_set:
                return False
        return True

    # This check is O(d) per edge. With 448 edges, it's fast.
    # But for local search, we need to be able to remove edges and re-add them.
    
    def greedy_random(seed):
        random.seed(seed)
        edges = list(range(len(all_edges)))
        random.shuffle(edges)
        
        selected = set()
        selected_indices = set()
        
        for idx in edges:
            edge = all_edges[idx]
            if is_valid_addition(edge, selected):
                selected.add(edge)
                selected_indices.add(idx)
        
        return selected, selected_indices

    def local_search(selected, selected_indices, max_iters=10000):
        # Try to improve by swapping edges: remove one, add another
        # Or just try to add an edge that was previously skipped? No, greedy took all possible.
        # Local search: try to remove an edge and see if we can add MORE than one edge back?
        # Or try to replace an edge with a different one that allows more future edges?
        # Simple local search: try to swap an edge (u,v) with (x,y) if it increases count? 
        # But count stays same in a swap.
        # Better: try to remove an edge, then greedily re-add as many as possible from the remaining unselected edges.
        # This is expensive.
        
        # Alternative: 2-opt style.
        # Let's stick to a simpler improvement: 
        # Try all pairs of (remove edge A, add edge B). If valid, do it.
        # But adding B might be blocked by A? No, removing A makes more space.
        
        current_count = len(selected)
        unselected = [e for i, e in enumerate(all_edges) if i not in selected_indices]
        
        improved = True
        while improved and (time.time() < deadline - 10.0):
            improved = False
            # Try removing each selected edge
            for rem_idx in list(selected_indices):
                rem_edge = all_edges[rem_idx]
                
                # Tentatively remove
                selected.remove(rem_edge)
                selected_indices.remove(rem_idx)
                
                # Try to add any unselected edge
                # To be fast, we can just try a few random unselected edges or all?
                # All unselected is ~448. 448 * 448 = 200k checks. Each check is O(7). 1.4M ops. 
                # This is fast enough for a few iterations.
                
                best_add_idx = -1
                # We want to add an edge that is valid.
                # Actually, we just want to increase the count.
                # If we remove 1 and add 1, count is same.
                # We need to remove 1 and add 2? Or remove 1 and add 1 that allows another?
                # This is complex.
                
                # Let's try a different local search:
                # Just try to add edges from the unselected list to the current set.
                # If we can add one, we are done (count increases).
                # But greedy already added all possible edges in a specific order.
                # If we change the order, we might get a different set.
                # So local search on a fixed set is hard to improve count without rebuilding.
                
                # Let's revert and just do multiple greedy runs with different seeds.
                # The greedy result depends heavily on the order.
                # Different orders can yield different counts.
                # The max count is likely found by trying many orders.
                
                selected.add(rem_edge)
                selected_indices.add(rem_idx)
                
                # Actually, let's just do the multi-start greedy.
                # It's simpler and often effective for this type of problem.
                # We can also try a "reverse greedy" or "random restart" approach.
                
                # Let's implement a simple hill climbing:
                # 1. Start with greedy result.
                # 2. Try to remove an edge.
                # 3. Try to add ANY edge from the remaining pool.
                # 4. If we can add an edge, we have increased count? No, we removed one.
                # 5. We need to add MORE than we removed.
                # 6. So, remove 1, try to add 2.
                
                # This is getting complicated. Let's stick to multi-start greedy.
                # And maybe a simple "add if possible" pass at the end, which greedy already does.
                
                pass
                
        return selected, selected_indices

    best_count = 0
    best_set = set()
    
    # Multi-start greedy
    num_starts = 1000
    seed = 42
    
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        selected, selected_indices = greedy_random(seed)
        count = len(selected)
        
        if count > best_count:
            best_count = count
            best_set = selected
            # If we hit the upper bound, we can stop
            if best_count >= 308:
                break
        
        seed += 1

    # Convert best_set to list of edges
    result = [list(e) for e in best_set]
    return result

# EVOLVE-BLOCK-END
