import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    d = 7
    n = 2 ** d
    total_edges = n * d // 2  # 448
    
    # 1. Generate all edges and map to indices
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx
    
    # 2. Precompute edge info and C4 structures
    edge_info = [] # (u, v, bit)
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        bit = diff.bit_length() - 1
        edge_info.append((u, v, bit))
        
    # List all C4s
    # A C4 is defined by 2 bits i, j and a base vertex x where bit i, j are 0.
    all_c4s = []
    c4_membership = [[] for _ in range(total_edges)]
    
    for i in range(d):
        for j in range(i+1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for x in range(n):
                if (x & mask_i) == 0 and (x & mask_j) == 0:
                    v1 = x
                    v2 = x ^ mask_i
                    v3 = x ^ mask_j
                    v4 = x ^ mask_i ^ mask_j
                    
                    e1 = edge_index.get((min(v1, v2), max(v1, v2)))
                    e2 = edge_index.get((min(v3, v4), max(v3, v4)))
                    e3 = edge_index.get((min(v1, v3), max(v1, v3)))
                    e4 = edge_index.get((min(v2, v4), max(v2, v4)))
                    
                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                        c4_idx = len(all_c4s)
                        all_c4s.append((e1, e2, e3, e4))
                        c4_membership[e1].append(c4_idx)
                        c4_membership[e2].append(c4_idx)
                        c4_membership[e3].append(c4_idx)
                        c4_membership[e4].append(c4_idx)
                        
    num_c4s = len(all_c4s)
    
    best_m = 0
    best_present = None
    
    # We will run multiple randomized greedy passes + local search
    # Estimate time per pass. 
    # Greedy deletion is O(E * C4_per_edge * log E) roughly.
    # Local search is variable.
    
    while time.time() < deadline - 20:
        present = [True] * total_edges
        c4_count = [6] * total_edges
        bad_c4s = set(range(num_c4s))
        
        # Greedy Deletion with Random Tie-Breaking
        while bad_c4s:
            if time.time() > deadline - 10:
                break
            max_count = -1
            candidates = []
            for e in range(total_edges):
                if present[e] and c4_count[e] > max_count:
                    max_count = c4_count[e]
                    candidates = [e]
                elif present[e] and c4_count[e] == max_count:
                    candidates.append(e)
            
            if not candidates:
                break # Should not happen if bad_c4s not empty
                
            # Randomly pick one
            best_e = random.choice(candidates)
            
            present[best_e] = False
            # Update counts
            for c4_idx in c4_membership[best_e]:
                if c4_idx in bad_c4s:
                    bad_c4s.remove(c4_idx)
                    e1, e2, e3, e4 = all_c4s[c4_idx]
                    for other_e in (e1, e2, e3, e4):
                        if other_e != best_e:
                            c4_count[other_e] -= 1
        
        if time.time() > deadline - 10:
            break
            
        m = sum(present)
        
        # Local Search: Try to add back edges
        # First, simple pass: add any edge that doesn't create a C4
        removed = [e for e in range(total_edges) if not present[e]]
        random.shuffle(removed)
        
        # We need a fast way to check if an edge can be added.
        # An edge e can be added if for all c4_idx in c4_membership[e],
        # not all other 3 edges in all_c4s[c4_idx] are present.
        
        # To speed up, we can maintain a "block_count" for each removed edge?
        # Or just check on the fly. Checking 6 C4s per edge is fast.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Re-shuffle removed list occasionally? Or just iterate.
            # Iterating through all removed edges every time is O(R * 6). R ~ 150. Fast.
            for e in removed:
                if present[e]:
                    continue
                # Check if e can be added
                can_add = True
                for c4_idx in c4_membership[e]:
                    e1, e2, e3, e4 = all_c4s[c4_idx]
                    # Check other 3
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    if all(present[x] for x in others):
                        can_add = False
                        break
                if can_add:
                    present[e] = True
                    m += 1
                    improved = True
                    # Remove from removed list? 
                    # Since we are iterating, just skip next time.
                    # To be efficient, we could manage the list, but simple is okay for 448 edges.
        
        # Advanced Local Search: 1-Remove, K-Add
        # Try to remove one edge 'f' and add multiple edges.
        # This is harder to implement robustly in short time.
        # Let's stick to multiple restarts of the Greedy+SimpleAdd phase.
        # The randomization in Greedy is the key.
        
        if m > best_m:
            best_m = m
            best_present = present[:]
            
    # Return the best found
    if best_present is None:
        # Fallback to current best if loop didn't run (shouldn't happen)
        best_present = [True] * total_edges
        # Need to make it valid? If time ran out during first greedy, it might be invalid.
        # But deadline guard in greedy should stop it.
        # If it stopped early, it's invalid. 
        # The problem requires a valid set.
        # If we can't guarantee validity, we should probably run a very fast greedy first.
        # But the loop above ensures we only update best_m if the greedy finished?
        # No, `bad_c4s` might not be empty if time ran out.
        # We must ensure best_present is valid.
        
        # Let's check if best_present is valid.
        # If not, we have a problem.
        # To be safe, let's force a valid solution if best is invalid.
        
        # Check validity of best_present
        is_valid = True
        for c4_idx in range(num_c4s):
            e1, e2, e3, e4 = all_c4s[c4_idx]
            if best_present[e1] and best_present[e2] and best_present[e3] and best_present[e4]:
                is_valid = False
                break
        
        if not is_valid:
             # Fallback: Run a deterministic greedy (original logic) to ensure validity
             present = [True] * total_edges
             c4_count = [6] * total_edges
             bad_c4s = set(range(num_c4s))
             while bad_c4s:
                 max_count = -1
                 best_e = -1
                 for e in range(total_edges):
                     if present[e] and c4_count[e] > max_count:
                         max_count = c4_count[e]
                         best_e = e
                 if best_e == -1: break
                 present[best_e] = False
                 for c4_idx in c4_membership[best_e]:
                     if c4_idx in bad_c4s:
                         bad_c4s.remove(c4_idx)
                         e1, e2, e3, e4 = all_c4s[c4_idx]
                         for other_e in (e1, e2, e3, e4):
                             if other_e != best_e:
                                 c4_count[other_e] -= 1
             # Simple add back
             removed = [e for e in range(total_edges) if not present[e]]
             random.shuffle(removed)
             for e in removed:
                 can_add = True
                 for c4_idx in c4_membership[e]:
                     e1, e2, e3, e4 = all_c4s[c4_idx]
                     others = [x for x in (e1, e2, e3, e4) if x != e]
                     if all(present[x] for x in others):
                         can_add = False
                         break
                 if can_add:
                     present[e] = True
             best_present = present
             best_m = sum(best_present)

    out = []
    for idx, (u, v) in enumerate(all_edges):
        if best_present[idx]:
            out.append([u, v])
            
    return out
# EVOLVE-BLOCK-END
