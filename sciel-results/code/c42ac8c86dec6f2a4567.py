import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit
    
    n = 40
    target_edges = 401
    
    def build_graph(seed_val):
        """
        Builds a graph based on a specific structural hypothesis.
        Returns (edges_set, adj_masks_list, max_book, max_edges_list)
        """
        random.seed(seed_val)
        edges = set()
        
        # Strategy A (Current/Inspiration 1): 
        # Partition A (9) and B (31).
        # Complete bipartite A-B.
        # Add ~122 edges inside B to keep intra-B degrees low.
        # This structure naturally limits common neighbors for intra-B edges.
        
        A = list(range(9))
        B = list(range(9, 40))
        
        # 1. Add all edges between A and B
        for a in A:
            for b in B:
                edges.add((a, b))
                
        # 2. Add 122 edges within B
        B1 = list(range(9, 24))
        B2 = list(range(24, 40))
        
        needed = 122
        added = 0
        
        deg_b1 = {v: 0 for v in B1}
        deg_b2 = {v: 0 for v in B2}
        
        # Generate all possible pairs between B1 and B2
        pairs = [(u, v) for u in B1 for v in B2]
        random.shuffle(pairs)
        
        # Add edges with a degree constraint to keep the structure "sparse" within B
        # We want to minimize common neighbors.
        # If we pick edges randomly with low degree, common neighbors are likely low.
        # Let's try to keep degree <= 8 or 9.
        max_deg = 9
        
        for u, v in pairs:
            if added == needed:
                break
            if deg_b1[u] < max_deg and deg_b2[v] < max_deg:
                edges.add((u, v))
                deg_b1[u] += 1
                deg_b2[v] += 1
                added += 1
                
        if added < needed:
            # Fallback: fill remaining ignoring constraints if necessary
            for u, v in pairs:
                if added == needed:
                    break
                if (u, v) not in edges:
                    edges.add((u, v))
                    added += 1

        # Build adjacency masks
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        # Calculate max book
        max_b = -1
        max_edges = []
        for u, v in edges:
            b = (adj_masks[u] & adj_masks[v]).bit_count()
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
                
        return edges, adj_masks, max_b, max_edges

    # --- Phase 1: Multi-start Initialization ---
    # Try a few different seeds/structures to find a good starting point.
    # The current program uses a single random shuffle. 
    # We will try 5 different random seeds for the intra-B edge placement.
    
    best_state = None
    best_max_book = float('inf')
    
    # We'll also try a slightly different partition if needed, but let's stick to the 9-31 split
    # which seems robust.
    
    seeds = [42, 123, 999, 555, 777]
    for s in seeds:
        edges, adj_masks, max_b, max_edges = build_graph(s)
        if max_b < best_max_book:
            best_max_book = max_b
            best_state = (edges, adj_masks, max_edges)
            
    # If we still have a very high book size, we might need a different structure,
    # but 9-31 with sparse intra-B usually yields ~7-9.
    
    if best_state is None:
        # Fallback to a simple random graph if something went wrong (shouldn't happen)
        edges, adj_masks, max_b, max_edges = build_graph(0)
        best_state = (edges, adj_masks, max_edges)
        best_max_book = max_b

    edges, adj_masks, current_max_edges = best_state
    current_max_book = best_max_book

    # --- Phase 2: Local Search Refinement ---
    
    def update_adj(u, v, add):
        if add:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
    def get_book_size(u, v):
        if u == v:
            return 0
        return (adj_masks[u] & adj_masks[v]).bit_count()
        
    def get_max_book_info():
        max_b = -1
        max_edges = []
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    no_improve_count = 0
    max_no_improve = 2000 # Allow more exploration
    
    # We use a "smooth" acceptance criteria:
    # 1. Always accept if max_book decreases.
    # 2. Accept if max_book stays same but count of max-book edges decreases.
    # 3. Accept if max_book stays same and count stays same with small probability (to escape plateaus).
    # 4. Reject if max_book increases.
    
    while time.time() < deadline:
        if current_max_book <= 7:
            # Theoretical lower bound is 40/6 = 6.66. So 7 is the smallest integer possible?
            # Wait, Edwards-Khadzhiivanov-Nikiforov: for n=40, m=401, book size >= 7?
            # The prompt says "score >= 40/6". 40/6 = 6.66.
            # Since book size is an integer, it must be at least 7.
            # So 7 is the theoretical minimum.
            # If we reach 7, we are at the theoretical optimum.
            # We should stop searching for lower, but we can try to stabilize or just return.
            # Actually, if we are at 7, we can't get lower.
            # So we can break immediately if we hit 7?
            # Yes, 7 is the minimum possible integer value >= 6.66.
            if current_max_book == 7:
                # We can break, but maybe there are multiple 7s? No, we just need the max to be 7.
                # The score is the max book size. So 7 is the best possible.
                break
            elif current_max_book <= 6:
                # This shouldn't happen if the theorem holds, but just in case.
                break

        if not current_max_edges:
            break
            
        # Pick a random edge from the max-book set to remove
        u, v = random.choice(current_max_edges)
        
        # Find the best non-edge to add
        # We want to minimize the book size of the new edge.
        # Heuristic: Find non-edge (x,y) with min common neighbors.
        min_book_new = float('inf')
        best_x, best_y = -1, -1
        
        # Iterate over all possible non-edges
        for x in range(n):
            for y in range(x + 1, n):
                if (1 << y) & adj_masks[x]:
                    continue
                
                b = (adj_masks[x] & adj_masks[y]).bit_count()
                if b < min_book_new:
                    min_book_new = b
                    best_x, best_y = x, y
                    if b == 0:
                        # Can't get better than 0
                        break
            if min_book_new == 0:
                break
        
        if best_x == -1:
            break
            
        # Perform the swap
        old_max = current_max_book
        old_max_count = len(current_max_edges)
        
        edges.remove((u, v))
        update_adj(u, v, False)
        
        edges.add((best_x, best_y))
        update_adj(best_x, best_y, True)
        
        new_max, new_max_edges = get_max_book_info()
        
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            else:
                # Small probability to accept to escape local minima
                # Use a decaying probability? Or fixed?
                # Fixed 0.05 seems reasonable.
                if random.random() < 0.05:
                    accepted = True
                else:
                    accepted = False
        else:
            accepted = False

        if accepted:
            current_max_book = new_max
            current_max_edges = new_max_edges
        else:
            # Rollback
            edges.add((u, v))
            update_adj(u, v, True)
            
            edges.remove((best_x, best_y))
            update_adj(best_x, best_y, False)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart?
                # We can try to perturb the graph by doing a few random swaps that don't increase max book too much?
                # Or just break if we've been stuck for a long time.
                # Given the tight time limit, breaking is safer than risky perturbations.
                # However, if we are stuck at 8, we might be able to reach 7 with a better perturbation.
                # Let's try a "kick": remove a random edge (not necessarily max book) and add a random non-edge
                # if it doesn't increase the max book.
                
                # Actually, let's just break. The multi-start already provided diversity.
                break

    result = [[u, v] for u, v in edges]
    return result
