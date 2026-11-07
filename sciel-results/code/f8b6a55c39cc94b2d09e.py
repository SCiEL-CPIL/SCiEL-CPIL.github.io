import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, 5s margin
    
    n = 9
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute valid coloring masks for each edge
    # A coloring is a 9-bit integer? No, 2 colors -> 9 bits.
    # But we need to check 3-colorability too later.
    # For 2-colorability check, we use 2 colors.
    # There are 2^9 = 512 colorings.
    # We represent the set of valid colorings as a Python integer with 512 bits.
    # Bit i is 1 if coloring i is valid (i.e., edge is NOT monochromatic).
    
    edge_valid_masks = []
    for e in all_triples:
        a, b, c = e
        mask = 0
        for coloring in range(512):
            ca = (coloring >> a) & 1
            cb = (coloring >> b) & 1
            cc = (coloring >> c) & 1
            # If not all same, it's valid
            if not (ca == cb == cc):
                mask |= (1 << coloring)
        edge_valid_masks.append(mask)
        
    def is_2_colorable(edges):
        # edges is a list of indices into all_triples
        if not edges:
            return True
        current_mask = (1 << 512) - 1
        for e_idx in edges:
            current_mask &= edge_valid_masks[e_idx]
            if current_mask == 0:
                return False
        return True

    def is_3_colorable(edges):
        # Check if there is a 3-coloring
        # 3^9 = 19683 colorings
        # Represent coloring as base-3 number? Or just iterate.
        # For speed, we can precompute? No, just iterate.
        # But we only call this if is_2_colorable is False.
        # So it's rare.
        for c1 in range(3):
            for c2 in range(3):
                for c3 in range(3):
                    # This is too slow if we do nested loops in Python for 19683.
                    # Better to iterate 0..19682 and decode.
                    pass
        # Let's use a faster check:
        # We can use the same bitmask idea? No, 3^9 is too big for a simple int mask?
        # 2^19683 is huge.
        # So we just iterate.
        for val in range(19683):
            temp = val
            colors = [0] * 9
            valid = True
            for i in range(9):
                colors[i] = temp % 3
                temp //= 3
            for e_idx in edges:
                a, b, c = all_triples[e_idx]
                if colors[a] == colors[b] == colors[c]:
                    valid = False
                    break
            if valid:
                return True, colors
        return False, None

    # Search for minimum m
    best_edges = None
    best_colors = None
    
    # Try m = 5, 6, 7
    for m in [5, 6, 7]:
        if time.time() > deadline - 10:
            break
            
        # Try to find a non-2-colorable, 3-colorable hypergraph with m edges
        # Use random sampling
        # We want to bias towards "dense" structures?
        # Just random for now.
        
        attempts = 0
        max_attempts = 10000
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            # Pick m random distinct edges
            # To avoid duplicates, use a set
            chosen = set()
            while len(chosen) < m:
                chosen.add(random.randint(0, num_triples - 1))
            edges_list = list(chosen)
            
            if not is_2_colorable(edges_list):
                # It is not 2-colorable. Check if 3-colorable.
                is_3c, colors = is_3_colorable(edges_list)
                if is_3c:
                    best_edges = [list(all_triples[e_idx]) for e_idx in edges_list]
                    best_colors = colors
                    break
            # If we found one, break the while loop
            if best_edges:
                break
                
        if best_edges:
            break
            
    if best_edges is None:
        # Fallback to the known 10-edge solution
        edges = [list(e) for e in combinations(range(5), 3)]
        colors = [0, 0, 1, 1, 2, 0, 0, 0, 0]
        # Verify it's 3-colorable (it is) and not 2-colorable (it is)
        return {'edges': edges, 'colors': colors}
    else:
        return {'edges': best_edges, 'colors': best_colors}
