import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def is_mono(color_mask, edge_mask):
        # Check if edge is monochromatic under coloring color_mask
        # Mono if all bits are 0 or all bits are 1
        return (color_mask & edge_mask) == 0 or (color_mask & edge_mask) == edge_mask

    def greedy_find_non_2_colorable(n):
        """
        Greedily selects edges to eliminate all 2-colorings.
        Returns a list of edges (tuples) or None if timeout.
        """
        # Generate all possible edges
        vertices = range(n)
        all_edges = list(itertools.combinations(vertices, 5))
        
        # Convert edges to masks for fast bit operations
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Initialize set of valid colorings
        # We can represent colorings as integers from 1 to 2^n - 2 (excluding 0 and all-1s is optional but they are always invalid for any edge)
        # Actually, 0 and all-1s are always mono for any edge, so they are never valid.
        # We start with all colorings and remove as we add edges.
        # To save memory and time, we can just keep a list of valid colorings.
        # For n=12, 2^12 = 4096. This is small.
        
        valid_colorings = list(range(1, (1 << n) - 1))
        
        selected_edges = []
        selected_masks = []
        
        # Map edge index to mask
        # We will iterate through all edges and pick the best one each time
        
        # Optimization: Pre-compute for each edge which colorings it kills?
        # No, just check on the fly.
        
        while valid_colorings:
            if time.time() > deadline - 5:
                return None
            
            best_edge_idx = -1
            best_kill_count = 0
            
            # Try all unselected edges
            # To make this faster, we can sample or just iterate all.
            # len(all_edges) for n=12 is 792. 792 * len(valid_colorings) checks.
            # If valid_colorings is 4000, that's 3 million checks per step.
            # We might need ~50 steps. 150 million checks. Might be too slow in Python.
            
            # Optimization: Use bitsets for valid colorings?
            # Or just sample edges if the set is large.
            # For n=10, edges=252, colorings=1024. 252*1024 = 258k checks/step. Fast.
            # For n=12, edges=792, colorings=4096. 3.2M checks/step. ~50 steps = 160M checks. ~1-2 seconds. OK.
            # For n=14, edges=2002, colorings=16384. 32M checks/step. Too slow.
            
            if n > 12:
                # Sample a subset of edges to speed up
                sample_size = min(100, len(all_edges) - len(selected_edges))
                # Get indices of unselected edges
                # This is getting complex. Let's stick to n <= 12 for the greedy part.
                # If n=12 fails, we might not find 51 edges. But 51 edges on n=10 is very likely.
                pass
            
            # Iterate through all edges to find the one that kills the most colorings
            # We can skip edges already selected
            selected_set = set(selected_edges)
            
            for i, e in enumerate(all_edges):
                if e in selected_set:
                    continue
                
                em = edge_masks[i]
                kill_count = 0
                # Count how many valid colorings are made mono by this edge
                for c in valid_colorings:
                    if is_mono(c, em):
                        kill_count += 1
                
                if kill_count > best_kill_count:
                    best_kill_count = kill_count
                    best_edge_idx = i
            
            if best_edge_idx == -1 or best_kill_count == 0:
                # No edge kills any coloring? Should not happen if valid_colorings is not empty
                # and there are edges available.
                break
                
            # Add the best edge
            best_edge = all_edges[best_edge_idx]
            best_mask = edge_masks[best_edge_idx]
            selected_edges.append(best_edge)
            selected_masks.append(best_mask)
            
            # Update valid colorings
            # Remove colorings that are mono under best_mask
            # List comprehension is fast
            valid_colorings = [c for c in valid_colorings if not is_mono(c, best_mask)]
            
            if len(selected_edges) > 100: # Safety break
                break

        if not valid_colorings:
            return selected_edges
        else:
            return None

    best_edges = None
    best_m = 200

    # Try n from 10 to 12
    # n=10 has 252 edges. Greedy should find a small set.
    # n=11 has 462 edges.
    # n=12 has 792 edges.
    
    for n in range(10, 13):
        if time.time() > deadline - 10:
            break
            
        edges = greedy_find_non_2_colorable(n)
        
        if edges is not None:
            m = len(edges)
            if m < best_m:
                best_m = m
                best_edges = edges
                
            # If we found 51 or less, we can stop?
            # The task says "Minimize". 51 is the known best.
            # If we find < 51, that's a breakthrough.
            # If we find 51, that's optimal.
            # We can stop if m <= 51.
            if m <= 51:
                break

    if best_edges is None:
        # Fallback: Random search for n=10, 11, 12 with m=51
        import random
        for n in range(10, 13):
            if time.time() > deadline - 5:
                break
            all_edges = list(itertools.combinations(range(n), 5))
            all_masks = []
            for e in all_edges:
                mask = 0
                for v in e:
                    mask |= (1 << v)
                all_masks.append(mask)
            
            for trial in range(500):
                if time.time() > deadline - 2:
                    break
                indices = random.sample(range(len(all_edges)), 51)
                edge_masks = [all_masks[i] for i in indices]
                
                # Check 2-colorability
                # Brute force is too slow for n=12 (4096 colorings * 51 edges = 200k ops, ok)
                # But we can use the same mono check
                is_2_colorable = True
                for c in range(1, (1 << n) - 1):
                    valid = True
                    for em in edge_masks:
                        if (c & em) == 0 or (c & em) == em:
                            valid = False
                            break
                    if valid:
                        is_2_colorable = True
                        break
                if not is_2_colorable:
                    best_edges = [list(all_edges[i]) for i in indices]
                    best_m = 51
                    break
            if best_edges:
                break

    if best_edges is None:
        # Absolute fallback: Complete hypergraph on 9 vertices
        # This is valid but not minimal (126 edges).
        # Better fallback: Use the greedy result even if it's > 51?
        # Or just return a known small one?
        # Let's try n=9 complete.
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_edges = edges
        best_m = len(edges)

    return {'n': n, 'edges': best_edges}
