import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute K4 edge indices
    # Edge index for (i, j) with i < j
    # idx = sum_{k=0}^{i-1} (15-k) + (j-i-1)
    # Or simply:
    # 120 edges.
    # We can build a mapping (i, j) -> index
    edge_idx = {}
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            edge_idx[(i, j)] = idx
            idx += 1
            
    # List of all K4s, each as a list of 6 edge indices
    k4_edges = []
    for combo in itertools.combinations(range(16), 4):
        a, b, c, d = combo
        edges = [
            edge_idx[(a, b)],
            edge_idx[(a, c)],
            edge_idx[(a, d)],
            edge_idx[(b, c)],
            edge_idx[(b, d)],
            edge_idx[(c, d)]
        ]
        k4_edges.append(edges)
        
    k4_edges_np = np.array(k4_edges, dtype=np.int32) # Shape (1820, 6)

    def is_valid(colours_np):
        # colours_np is a 1D array of shape (120,)
        # Get colors for each K4 edge
        # Shape (1820, 6)
        k4_colors = colours_np[k4_edges_np]
        # Sort each row
        k4_colors_sorted = np.sort(k4_colors, axis=1)
        # Check for adjacent duplicates
        # If distinct count >= 5, duplicates <= 1
        # eq = (k4_colors_sorted[:, 1:] == k4_colors_sorted[:, :-1])
        # dup_counts = np.sum(eq, axis=1)
        # return np.all(dup_counts <= 1)
        
        # Faster: check if any row has > 1 duplicate
        eq = (k4_colors_sorted[:, 1:] == k4_colors_sorted[:, :-1])
        dup_counts = np.sum(eq, axis=1)
        return np.all(dup_counts <= 1)

    # Generate seed
    def generate_seed():
        return np.array([(i + j) % 17 for i in range(16) for j in range(i + 1, 16)], dtype=np.int32)
    
    current = generate_seed()
    
    # Greedy Merging
    # We maintain a list of active colors
    # We try to merge color c1 into c2
    
    # To speed up, we can precompute which edges have which colors
    # But since we update, we just re-evaluate.
    
    # Optimization: Only try merging if it reduces color count
    # And check validity.
    
    # We can also randomize the order of merge attempts to escape local minima?
    # For now, deterministic sorted order.
    
    # Helper to get set of used colors
    def get_used_colors(colours_np):
        return set(np.unique(colours_np).tolist())

    used_colors = get_used_colors(current)
    
    # We will try to merge colors until we can't or time runs out
    # We iterate while we can make a merge
    max_iterations = 1000 # Safety bound, time is the real bound
    
    while time.time() < deadline:
        used_colors = get_used_colors(current)
        if len(used_colors) <= 13:
            break
            
        sorted_colors = sorted(used_colors)
        merged = False
        
        # Try all pairs
        for i in range(len(sorted_colors)):
            if time.time() > deadline:
                break
            c1 = sorted_colors[i]
            for j in range(i + 1, len(sorted_colors)):
                if time.time() > deadline:
                    break
                c2 = sorted_colors[j]
                
                # Create new colour array
                # mask where colour == c1
                mask = (current == c1)
                if not np.any(mask):
                    continue # Should not happen if c1 is in used_colors
                
                new_colours = current.copy()
                new_colours[mask] = c2
                
                if is_valid(new_colours):
                    current = new_colours
                    merged = True
                    break
            if merged:
                break
                
        if not merged:
            # Try random restarts or different order?
            # For now, break.
            # We could try shuffling the color order for the next attempt
            # But if no merge works in sorted order, it might work in another?
            # Merging is commutative in effect (c1->c2 is same as c2->c1 in terms of resulting set, 
            # but the "direction" matters for which color label survives. 
            # However, since we just care about the *number* of colors and validity, 
            # and the graph is symmetric, if {c1, c2} can be merged, it doesn't matter which label survives 
            # as long as we consistently relabel.
            # Wait, if I merge c1 into c2, all c1 become c2.
            # If I merge c2 into c1, all c2 become c1.
            # The resulting partition of edges is the same.
            # So if one direction works, the other works?
            # Yes, the set of colors is the same. The validity depends on the partition, not the labels.
            # So if no pair works in sorted order, no pair works at all?
            # Yes, because we are checking all pairs {c1, c2}.
            
            break
            
    return current.tolist()
