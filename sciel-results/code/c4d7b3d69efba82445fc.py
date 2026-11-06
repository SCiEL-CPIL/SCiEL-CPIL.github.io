"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    d = 7
    n_vertices = 1 << d
    # Precompute all edges
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # Map edge to index for faster lookup
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    n_edges = len(all_edges)
    
    # Precompute conflict graph: for each edge, which other edges form a C4 with it?
    # Two edges (u,v) and (u',v') form a C4 if they are opposite sides or adjacent sides?
    # A 4-cycle in Q_d is a 2-dimensional face.
    # Edges in a face: (x, x^e_i), (x^e_j, x^e_i^e_j), (x^e_j, x), (x^e_i^e_j, x^e_i) ? No.
    # Let's characterize conflicts directly.
    # Edge e1 = (a, b) where b = a ^ (1<<i).
    # Edge e2 = (c, d) where d = c ^ (1<<j).
    # They form a C4 if they are part of the same 2D face.
    # This happens if:
    # 1. i != j
    # 2. The four vertices a, b, c, d form a square.
    # Specifically, if e1 and e2 are parallel (i==j), they don't form a C4 directly, 
    # but they might be opposite sides of a C4? No, opposite sides of a C4 are parallel.
    # Wait, a C4 has 4 edges. Two edges are in a C4 if they are either:
    # - Adjacent in the C4 (share a vertex)
    # - Opposite in the C4 (disjoint vertices)
    
    # Let's just check: do edges e1 and e2 belong to the same 2D face?
    # A 2D face is defined by two bit positions i, j and a fixed assignment of the other d-2 bits.
    # The 4 vertices are:
    # v0 = x
    # v1 = x ^ (1<<i)
    # v2 = x ^ (1<<j)
    # v3 = x ^ (1<<i) ^ (1<<j)
    # The 4 edges are:
    # (v0, v1), (v1, v3), (v3, v2), (v2, v0)
    
    # So, for a given edge e = (u, v) with diff bit i:
    # It belongs to faces defined by any other bit j != i.
    # The face is determined by u (or v) and the bit j.
    # The other edges in this face are:
    # Let u' = u ^ (1<<j), v' = v ^ (1<<j).
    # The other edges are (u, u'), (v, v'), and (u', v').
    # So e=(u,v) conflicts with (u,u'), (v,v'), and (u',v').
    # Note: (u,u') and (v,v') share vertices with e. (u',v') is opposite.
    
    conflicts = [[] for _ in range(n_edges)]
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i: continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            # The 3 other edges in this face
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            for e in (e1, e2, e3):
                if e in edge_to_idx:
                    c_idx = edge_to_idx[e]
                    if c_idx != idx:
                        conflicts[idx].append(c_idx)
    
    # Remove duplicates in conflicts just in case (though logic above seems unique per j)
    for i in range(n_edges):
        conflicts[i] = list(set(conflicts[i]))
    
    # Helper: check if adding edge idx to current set is valid
    # current_set is a boolean array or set of indices
    # For speed, we maintain a count of how many conflicts of an edge are in the set.
    # conflict_count[e] = number of edges in current_set that conflict with e.
    # e can be added if conflict_count[e] == 0.
    
    def run_greedy(randomize_order=True):
        order = list(range(n_edges))
        if randomize_order:
            random.shuffle(order)
        
        in_set = [False] * n_edges
        conflict_count = [0] * n_edges
        count = 0
        chosen_indices = []
        
        for idx in order:
            if conflict_count[idx] == 0:
                # Add idx
                in_set[idx] = True
                count += 1
                chosen_indices.append(idx)
                # Update conflict counts for all neighbors
                for c in conflicts[idx]:
                    conflict_count[c] += 1
        return chosen_indices, in_set, conflict_count, count

    def local_improve(chosen_indices, in_set, conflict_count, count):
        improved = True
        while improved:
            improved = False
            # Try to add any edge
            for idx in range(n_edges):
                if not in_set[idx] and conflict_count[idx] == 0:
                    in_set[idx] = True
                    count += 1
                    chosen_indices.append(idx)
                    for c in conflicts[idx]:
                        conflict_count[c] += 1
                    improved = True
            
            if improved:
                continue
            
            # Try "1-out, k-in" moves
            # For each edge in set, remove it, see how many can be added.
            # To speed up, we can limit the search or do it efficiently.
            best_gain = 0
            best_move = None # (remove_idx, [add_indices])
            
            # Optimization: Only consider edges that are "critical" or just sample?
            # With 448 edges, 448 iterations of "remove and greedy add" is feasible.
            # Greedy add after removal: 
            # When we remove u, conflict_count for all neighbors of u decreases by 1.
            # Then we can greedily add any edge with conflict_count 0.
            
            for u in chosen_indices:
                # Simulate removal of u
                # We need to know which edges become available.
                # An edge v is available if conflict_count[v] == 1 and the only conflict was u?
                # Or if conflict_count[v] becomes 0 after removing u.
                
                # Let's just do the full simulation for the top few candidates or all if fast enough.
                # 448 * (448 + updates) ~ 200k ops. Very fast.
                
                # Store old state to rollback? No, we can just calculate the potential.
                # To add, we need a list of candidates.
                
                # Let's try a simpler heuristic: 
                # Remove u.
                # Find all v not in set such that conflict_count[v] == 1 and u is the one blocking it?
                # Actually, conflict_count[v] counts how many neighbors are in set.
                # If we remove u, conflict_count[v] drops by 1 if v conflicts with u.
                # So v becomes addable if conflict_count[v] was 1 AND v conflicts with u.
                # OR if conflict_count[v] was 0 (already addable, but we checked that).
                
                candidates = []
                for v in conflicts[u]:
                    if not in_set[v]:
                        if conflict_count[v] == 1:
                            candidates.append(v)
                
                if not candidates:
                    continue
                
                # Greedily add from candidates
                # We need to be careful: adding one candidate might block others.
                # Let's simulate.
                new_count = count - 1
                new_in_set = in_set[:] # Copy is expensive? 448 bools. OK.
                new_conf_count = conflict_count[:]
                
                # Remove u
                new_in_set[u] = False
                for c in conflicts[u]:
                    new_conf_count[c] -= 1
                
                # Now greedily add from candidates (and potentially others that became 0?)
                # Only candidates became 0 from 1. Others were >1 or 0.
                # If they were 0, they would have been added in the previous "add any" loop.
                # So we only need to consider candidates.
                
                # Sort candidates by degree or just random?
                random.shuffle(candidates)
                
                for v in candidates:
                    if new_in_set[v]: continue # Might have been added? No, we only add if not in set.
                    if new_conf_count[v] == 0:
                        # Add v
                        new_in_set[v] = True
                        new_count += 1
                        for c in conflicts[v]:
                            new_conf_count[c] += 1
                
                gain = new_count - count
                if gain > best_gain:
                    best_gain = gain
                    best_move = (u, candidates, new_in_set, new_conf_count, new_count)
                    # Early exit if gain is large?
                    if best_gain > 1:
                        # Apply immediately? Or keep looking for best?
                        # Let's keep looking for best in this pass to avoid local traps.
                        pass
            
            if best_gain > 0 and best_move:
                u, candidates, new_in_set, new_conf_count, new_count = best_move
                # Apply move
                in_set = new_in_set
                conflict_count = new_conf_count
                count = new_count
                # Rebuild chosen_indices
                chosen_indices = [i for i, val in enumerate(in_set) if val]
                improved = True
        
        return chosen_indices, in_set, conflict_count, count

    deadline = time.time() + 110
    best_count = 0
    best_result = []
    
    # Initial random greedy
    while time.time() < deadline:
        chosen, in_set, conf_count, count = run_greedy(True)
        
        # Local improve
        chosen, in_set, conf_count, count = local_improve(chosen, in_set, conf_count, count)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_edges[i]) for i in chosen]
            
        # If we are close to upper bound, we can stop or continue
        if best_count >= 308:
            break

    return best_result
# EVOLVE-BLOCK-END
