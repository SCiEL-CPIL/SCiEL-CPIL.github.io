import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # 1. Map edges to indices
    edge_map = {}
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[(u, v)] = idx
            edge_map[(v, u)] = idx
            edge_list.append((u, v))
            idx += 1
            
    # 2. Precompute the 48 Q3 patterns on vertices 0..7
    # Standard Q3 edges on 0..7 (3-bit)
    std_q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    # Generate the 48 unique edge sets (as frozensets of local vertex pairs)
    # We use the 48 automorphisms of the cube.
    # An automorphism is a permutation of bits (3! = 6) and a flip of bits (2^3 = 8).
    # Total 48.
    # Let's generate them explicitly.
    
    local_patterns = set()
    
    # Permutations of axes: (0,1,2) -> (x,y,z)
    axis_perms = list(itertools.permutations([0, 1, 2]))
    
    for perm in axis_perms:
        for flip_mask in range(8):
            # Define the transformation
            def transform(v):
                # v is 0..7 (3 bits)
                b0 = (v >> 0) & 1
                b1 = (v >> 1) & 1
                b2 = (v >> 2) & 1
                
                # Apply flip
                b0 ^= (flip_mask >> 0) & 1
                b1 ^= (flip_mask >> 1) & 1
                b2 ^= (flip_mask >> 2) & 1
                
                # Apply permutation
                # perm[0] is the new position for the old bit 0?
                # Let's say perm is a mapping from new axis index to old axis index.
                # Or simpler: just permute the bits.
                # Let's construct the new value.
                new_v = 0
                # The bit at position i in the new value comes from the bit at position perm[i] in the old value?
                # Actually, let's just map the basis vectors.
                # e0 -> e_perm[0], e1 -> e_perm[1], e2 -> e_perm[2].
                # So if old v has bit i, new v has bit perm[i].
                
                # Wait, standard automorphism: x' = A x.
                # If we permute axes, the bit at position i moves to position perm[i].
                
                if b0: new_v |= (1 << perm[0])
                if b1: new_v |= (1 << perm[1])
                if b2: new_v |= (1 << perm[2])
                
                return new_v
            
            # Apply to all 8 vertices
            mapped_vertices = [transform(i) for i in range(8)]
            
            # Check if this is a valid bijection (it should be)
            if len(set(mapped_vertices)) != 8:
                continue
                
            # Generate the edge set for this mapping
            # We need the edges in the image.
            # An edge (u, v) in standard Q3 maps to (transform(u), transform(v)).
            pattern_edges = set()
            for u, v in std_q3_edges:
                mu = transform(u)
                mv = transform(v)
                if mu > mv:
                    mu, mv = mv, mu
                pattern_edges.add((mu, mv))
            
            local_patterns.add(frozenset(pattern_edges))
            
    # We expect 48 patterns.
    # Convert to list of frozensets
    local_patterns_list = list(local_patterns)
    
    # 3. Map local patterns to global Q3 masks (bitmasks of 120 edges)
    all_q3_masks = []
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    # Precompute local vertex to global vertex mapping for speed?
    # We will do it inside the loop.
    
    for s in subsets_8:
        # s is a tuple of 8 global vertices
        # We need to map local 0..7 to s[0]..s[7]
        # But the pattern edges are pairs of local vertices (0..7).
        # So we map local vertex i to global vertex s[i].
        
        local_to_global = [0]*8
        for i in range(8):
            local_to_global[i] = s[i]
            
        for pattern in local_patterns_list:
            mask = 0
            valid = True
            for u, v in pattern:
                gu = local_to_global[u]
                gv = local_to_global[v]
                if gu > gv:
                    gu, gv = gv, gu
                e_idx = edge_map.get((gu, gv))
                if e_idx is None:
                    valid = False
                    break
                mask |= (1 << e_idx)
            if valid:
                all_q3_masks.append(mask)
                
    # 4. Precompute edge_to_q3_indices
    # edge_to_q3_indices[e] = list of indices i such that all_q3_masks[i] has bit e set
    edge_to_q3_indices = [[] for _ in range(total_edges)]
    for i, mask in enumerate(all_q3_masks):
        # Iterate bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            edge_to_q3_indices[e_idx].append(i)
            m ^= lsb
            
    # 5. Search
    best_mask = 0
    best_count = 0
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        current_mask = 0
        
        # Greedy Phase
        edge_order = list(range(total_edges))
        random.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() > deadline - 5:
                break
            # Check if adding e creates a Q3
            creates_q3 = False
            for q3_idx in edge_to_q3_indices[e]:
                q3_mask = all_q3_masks[q3_idx]
                # If (q3_mask without e) is subset of current_mask, then adding e completes it
                if (q3_mask ^ (1 << e)) & current_mask == (q3_mask ^ (1 << e)):
                    creates_q3 = True
                    break
            
            if not creates_q3:
                current_mask |= (1 << e)
                
        # Local Search Phase
        # Try to add edges, if stuck, remove an edge from a "full" Q3
        improved = True
        max_local_iters = 100
        for _ in range(max_local_iters):
            if not improved:
                break
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Try to add random missing edges
            missing_edges = [e for e in range(total_edges) if not (current_mask & (1 << e))]
            if missing_edges:
                random.shuffle(missing_edges)
                for e in missing_edges:
                    if time.time() > deadline - 5:
                        break
                    creates_q3 = False
                    for q3_idx in edge_to_q3_indices[e]:
                        q3_mask = all_q3_masks[q3_idx]
                        if (q3_mask ^ (1 << e)) & current_mask == (q3_mask ^ (1 << e)):
                            creates_q3 = True
                            break
                    
                    if not creates_q3:
                        current_mask |= (1 << e)
                        improved = True
                        if bin(current_mask).count('1') > best_count:
                            best_count = bin(current_mask).count('1')
                            best_mask = current_mask
            else:
                # No missing edges? Graph is complete (impossible if Q3 free)
                # Or we are stuck.
                # Find a Q3 that is "almost" complete (11 edges present)
                # We can find this by checking edges in current_mask?
                # Better: Pick a random edge e in current_mask, look at Q3s containing e.
                # If a Q3 has 11 edges in current_mask, it's a candidate for "almost full".
                # Actually, if we are stuck, it means every missing edge e creates a Q3.
                # This implies there is a Q3 Q such that for all e in Q \ current_mask, adding e completes a Q3?
                # No, it means for every missing e, there EXISTS a Q3 containing e that is "ready".
                
                # Strategy: Pick a random edge e that is "critical" (part of many ready Q3s)?
                # Simpler: Pick a random edge e from current_mask, and a random Q3 containing e.
                # If that Q3 is "full" (12 edges), we have a bug.
                # If it has 11 edges, remove a random edge from it.
                
                # Let's just pick a random edge from current_mask and a random Q3 containing it.
                # If the Q3 is not full, removing an edge from it might free up space.
                
                if current_mask == 0:
                    break
                    
                # Pick random edge in current_mask
                # Convert mask to list of indices
                current_edges_list = []
                m = current_mask
                while m:
                    lsb = m & -m
                    current_edges_list.append(lsb.bit_length() - 1)
                    m ^= lsb
                    
                if not current_edges_list:
                    break
                    
                e_rand = random.choice(current_edges_list)
                q3_candidates = edge_to_q3_indices[e_rand]
                if not q3_candidates:
                    continue
                    
                q3_idx = random.choice(q3_candidates)
                q3_mask = all_q3_masks[q3_idx]
                
                # Count edges in q3_mask that are in current_mask
                # If count == 12, we have a Q3 in our graph! This should not happen.
                # If count == 11, we can remove one of the 11 edges to break it.
                # If count < 11, removing an edge doesn't help immediately with the "stuck" state 
                # (which is caused by missing edges creating Q3s), but it might allow other additions.
                
                # Let's remove a random edge from the intersection of q3_mask and current_mask
                inter_mask = q3_mask & current_mask
                if inter_mask == 0:
                    continue
                    
                # Pick random edge from inter_mask
                m = inter_mask
                edges_in_q3 = []
                while m:
                    lsb = m & -m
                    edges_in_q3.append(lsb.bit_length() - 1)
                    m ^= lsb
                    
                if edges_in_q3:
                    e_remove = random.choice(edges_in_q3)
                    current_mask &= ~(1 << e_remove)
                    improved = True # Force another iteration
                    
        if bin(current_mask).count('1') > best_count:
            best_count = bin(current_mask).count('1')
            best_mask = current_mask
            
    # Convert best_mask to list of [u, v]
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        e_idx = lsb.bit_length() - 1
        u, v = edge_list[e_idx]
        result.append([u, v])
        m ^= lsb
        
    return result
