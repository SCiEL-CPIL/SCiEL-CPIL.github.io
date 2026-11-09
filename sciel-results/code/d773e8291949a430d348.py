import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    d = 7
    n = 2 ** d
    
    # Precompute edges and their 2D face partners
    # Edge is (u, v) with u < v
    # For an edge in dimension i, the 4-cycle involves flipping bit j.
    # The other 3 edges are:
    # 1. (u ^ (1<<j), v ^ (1<<j))  -- parallel edge
    # 2. (u, u ^ (1<<j))           -- left connector
    # 3. (v, v ^ (1<<j))           -- right connector
    # Note: u and v differ in bit i. j != i.
    
    edges = []
    edge_to_idx = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v, i))
                edge_to_idx[(u, v)] = idx
    
    total_edges = len(edges)
    
    # Precompute for each edge, the list of (other_edge_1, other_edge_2, other_edge_3)
    # that would complete a C4 if this edge were added.
    # Actually, it's easier to check validity on the fly or precompute conflicts.
    # Conflict: Edge A conflicts with Edge B if they are in the same 2D face and 
    # together with 2 others form a C4? No, the constraint is "No 2D face may have ALL 4 edges".
    # So an edge is invalid to add if there exists a 2D face containing it where the other 3 are present.
    
    # Let's precompute for each edge e, the set of "potential blockers" or rather, 
    # the sets of 3 edges that, if all present, would make e invalid.
    # For edge e = (u, v, i), and any j != i:
    # The face is defined by bits i and j.
    # Vertices: u, v=u^(1<<i), c=u^(1<<j), e2=v^(1<<j)
    # Edges in face:
    # 1. (u, v) -> e
    # 2. (c, e2) -> parallel
    # 3. (u, c) -> connector 1
    # 4. (v, e2) -> connector 2
    
    # So e is blocked by face (i,j) if parallel, connector1, connector2 are all present.
    
    # Precompute for each edge index, a list of tuples of 3 other edge indices.
    # If all 3 are in 'selected', the edge cannot be added.
    edge_blockers = [[] for _ in range(total_edges)]
    
    for idx, (u, v, i) in enumerate(edges):
        for j in range(d):
            if j == i:
                continue
            c = u ^ (1 << j)
            e2 = v ^ (1 << j)
            
            # Get indices of the other 3 edges
            # Parallel: (c, e2)
            p1, p2 = min(c, e2), max(c, e2)
            idx_par = edge_to_idx.get((p1, p2))
            
            # Connector 1: (u, c)
            p3, p4 = min(u, c), max(u, c)
            idx_c1 = edge_to_idx.get((p3, p4))
            
            # Connector 2: (v, e2)
            p5, p6 = min(v, e2), max(v, e2)
            idx_c2 = edge_to_idx.get((p5, p6))
            
            if idx_par is not None and idx_c1 is not None and idx_c2 is not None:
                edge_blockers[idx].append((idx_par, idx_c1, idx_c2))
    
    def is_valid_addition(idx, selected_set):
        """Check if edge idx can be added given current selected_set."""
        for b1, b2, b3 in edge_blockers[idx]:
            if b1 in selected_set and b2 in selected_set and b3 in selected_set:
                return False
        return True
    
    def count_violations(selected_set):
        """Count number of C4s that are fully selected."""
        # A C4 is fully selected if all 4 edges are in selected_set.
        # We can iterate through all 2D faces.
        # Number of 2D faces in Q7: C(7,2) * 2^(7-2) = 21 * 32 = 672 faces.
        # Each face has 4 edges.
        # We can precompute the 4 edge indices for each face.
        # But doing this every time is expensive.
        # Alternative: Just return the list of violated faces?
        # Or better: In local search, we just need to find ONE violated edge to remove.
        # So we can just pick a random edge in selected_set and check if it's part of a violation?
        # No, we need to remove an edge that is part of a violation.
        # Let's just return a list of edges that are "critical" (part of a full C4).
        pass

    # Precompute faces for violation checking
    # Face is defined by pair of dimensions (i, j) and a base vertex with bits i,j = 0.
    # Actually, simpler: iterate all 2D subcubes.
    # There are C(7,2) = 21 pairs of dimensions.
    # For each pair (i, j), there are 2^(7-2) = 32 faces.
    # Total 672 faces.
    
    face_edges = []
    for i in range(d):
        for j in range(i+1, d):
            # Iterate over all combinations of other bits
            # Mask for other bits
            mask = 0
            for k in range(d):
                if k != i and k != j:
                    mask |= (1 << k)
            
            # Iterate over all subsets of mask
            # There are 2^(7-2) = 32 such subsets
            # We can iterate m from 0 to 2^7-1 and check if (m & ~mask) == 0?
            # Or just iterate all u and check if u has 0 in i and j?
            # No, a face is a specific 2D subcube.
            # Let's just iterate all u, and for each u that is the "min" corner of a face?
            # Simpler: Precompute all 4 edges for every 2D face.
            
            for u in range(n):
                # Check if u is the corner with bits i and j set to 0
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u ^ (1 << i)
                    v2 = u ^ (1 << j)
                    v3 = u ^ (1 << i) ^ (1 << j)
                    
                    # Edges:
                    # (u, v1) - dim i
                    # (u, v2) - dim j
                    # (v1, v3) - dim j
                    # (v2, v3) - dim i
                    
                    e1 = edge_to_idx.get((min(u, v1), max(u, v1)))
                    e2 = edge_to_idx.get((min(u, v2), max(u, v2)))
                    e3 = edge_to_idx.get((min(v1, v3), max(v1, v3)))
                    e4 = edge_to_idx.get((min(v2, v3), max(v2, v3)))
                    
                    if None not in (e1, e2, e3, e4):
                        face_edges.append((e1, e2, e3, e4))
    
    # Local Search Strategy:
    # 1. Start with a greedy solution (fast).
    # 2. Loop while time permits:
    #    a. Try to add random edges.
    #    b. If stuck (no edges can be added), remove a random edge from a violated C4?
    #       Wait, the greedy solution is always valid. So we start with 0 violations.
    #       If we only add valid edges, we never have violations.
    #       The problem is that greedy gets stuck in a local optimum where no single edge can be added.
    #       To escape, we must REMOVE some edges, creating "space", then ADD others.
    #       This is a 2-opt style move or simulated annealing.
        
    def greedy_init():
        sel = set()
        # Randomize order of edges for diversity
        order = list(range(total_edges))
        random.shuffle(order)
        for idx in order:
            if is_valid_addition(idx, sel):
                sel.add(idx)
        return sel
    
    def get_violated_edges(sel):
        """Return a list of edges that are part of at least one fully selected C4."""
        violated = set()
        for f in face_edges:
            if all(e in sel for e in f):
                violated.update(f)
        return list(violated)
    
    best_sel = greedy_init()
    best_size = len(best_sel)
    
    while time.time() < deadline:
        # Try to improve current best_sel
        cur_sel = set(best_sel)
        
        # Phase 1: Try to add edges
        # We can just run a greedy add on the remaining edges?
        # Or random attempts.
        # Let's try random attempts first, then greedy fill.
        
        # Random Add Phase
        # Try to add random edges from the complement
        complement = [i for i in range(total_edges) if i not in cur_sel]
        random.shuffle(complement)
        
        added_any = False
        for idx in complement:
            if is_valid_addition(idx, cur_sel):
                cur_sel.add(idx)
                added_any = True
                # Don't break, keep adding if possible?
                # Actually, adding one might block another. 
                # Greedy fill is more efficient.
                break # Just do one random add then re-evaluate? 
                     # No, let's just do a full greedy pass on the complement.
        
        # Full Greedy Pass on complement
        # This is fast: O(N * checks)
        # Checks are O(1) amortized if we just check blockers.
        # Blockers are precomputed.
        
        # Let's re-run greedy on the current state to see if we can add more
        # Actually, if we just did a greedy init, we are already at a local max for that order.
        # To escape, we need to REMOVE.
        
        # Phase 2: Removal and Re-add
        # Pick a random edge to remove? No, that decreases size.
        # Pick a random SET of edges to remove?
        # Heuristic: Remove 1-3 random edges, then run greedy fill again.
        
        # Let's try removing 1 random edge
        if cur_sel:
            remove_idx = random.choice(list(cur_sel))
            cur_sel.remove(remove_idx)
            
            # Now try to add edges again
            # We can try to add the removed edge back? No, it was valid before.
            # The space opened up allows OTHER edges to be added.
            
            # Greedy fill
            # We need to be efficient.
            # Let's just iterate through all non-selected edges and add if valid.
            # This is O(N * B) where B is number of blockers.
            # N=448, B~14. Very fast.
            
            # To avoid bias, shuffle the order of non-selected edges
            non_sel = [i for i in range(total_edges) if i not in cur_sel]
            random.shuffle(non_sel)
            
            for idx in non_sel:
                if is_valid_addition(idx, cur_sel):
                    cur_sel.add(idx)
            
            if len(cur_sel) > best_size:
                best_sel = set(cur_sel)
                best_size = len(cur_sel)
        
        # If we are stuck at the same size for a while, maybe try removing 2 edges?
        # For now, single edge removal + greedy refill is a decent local search.
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    # Convert best_sel to list of [u, v]
    result = []
    for idx in best_sel:
        u, v, _ = edges[idx]
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
