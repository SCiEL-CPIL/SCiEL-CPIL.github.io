import random
import time

def construct():
    deadline = time.time() + 110.0  # safety margin before 150s kill
    d = 7
    n = 1 << d
    # Precompute all edges and their "opposite" pairs for C4 detection
    # An edge (u,v) with direction bit i: the two other edges of the face
    # through this edge are determined by the other bit j.
    # For C4 detection: edge (u,v) along dim i creates a C4 iff there exists j != i
    # such that edges (u, u^bit_j), (v, v^bit_j), and (u^bit_j, v^bit_j) are all present.
    
    # Represent edges as (min(u,v), max(u,v)) for set membership
    # Precompute for each edge its two "partner" edge pairs for each other dimension
    # Actually, for speed, let's use a different approach:
    # Use a boolean array indexed by edge id.
    
    # Assign each edge a unique id
    # Edge (u, v) where v = u | (1<<i) and u has bit i = 0
    # id = u * d + i  (since u < v always when bit i is set in v)
    
    # For C4 check: adding edge (u, i) [meaning u to u^(1<<i)] creates a C4
    # if for some j != i, all of:
    #   edge (u, j) exists, edge (u^(1<<i), j) exists, edge (u^(1<<j), i) exists
    #   (where edge (x, k) means x to x^(1<<k), requiring bit k of x is 0)
    
    # Let's precompute: for each edge (u, i), the list of (j, e1, e2, e3) where
    # e1 = (u, j), e2 = (u^(1<<i), j), e3 = (u^(1<<j), i)
    # But we need to be careful: edge (x, k) is only valid if bit k of x is 0.
    
    # Simpler: use a set of frozensets or a 2D boolean array.
    # Let's use a flat array: edge_present[u * d + i] where we store it only when bit i of u is 0.
    # Actually, let's just use a set of (u, v) with u < v.
    
    # For speed, let's use numpy or a bytearray indexed by edge_id.
    # edge_id for edge between u and v (u < v): 
    #   diff = u ^ v, which is a power of 2. i = bit_length - 1.
    #   If bit i of u is 0, then u < v, edge_id = u * d + i
    #   If bit i of u is 1, then v < u, edge_id = v * d + i
    # So edge_id = min(u,v) * d + i where i = (u^v).bit_length()-1
    
    def edge_id(u, v):
        diff = u ^ v
        i = diff.bit_length() - 1
        a, b = (u, v) if u < v else (v, u)
        return a * d + i
    
    def edge_from_id(eid):
        u = eid // d
        i = eid % d
        v = u | (1 << i)
        return u, v
    
    # Precompute C4 partners for each edge
    # For edge (u, i) where bit i of u is 0:
    #   For each j != i:
    #     e1 = edge (u, j) -> valid only if bit j of u is 0
    #     e2 = edge (u^(1<<i), j) -> valid only if bit j of u^(1<<i) is 0
    #     e3 = edge (u^(1<<j), i) -> valid only if bit i of u^(1<<j) is 0
    # Note: if bit j of u is 0, then bit j of u^(1<<i) is also 0 (since i != j)
    #       and bit i of u^(1<<j) is 0 (since j != i and bit i of u is 0)
    # So all three are valid edge ids when bit j of u is 0.
    # If bit j of u is 1, then we'd need to use the "other" representation.
    # Let me reconsider: the C4 has vertices u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j)
    # The four edges are:
    #   (u, u^(1<<i)) - dim i
    #   (u^(1<<i), u^(1<<i)^(1<<j)) - dim j
    #   (u^(1<<j), u^(1<<i)^(1<<j)) - dim i
    #   (u, u^(1<<j)) - dim j
    # For the check when adding edge (u, u^(1<<i)):
    #   We need edges (u, u^(1<<j)), (u^(1<<i), u^(1<<i)^(1<<j)), (u^(1<<j), u^(1<<i)^(1<<j))
    
    # Let me just precompute for each edge_id the list of 3-tuples of edge_ids that would form a C4.
    # There are 448 edges. For each, up to 6 other dimensions.
    
    c4_partners = [[] for _ in range(n * d)]
    for u in range(n):
        for i in range(d):
            if u & (1 << i):
                continue  # not the canonical representation
            eid = u * d + i
            for j in range(d):
                if j == i:
                    continue
                # Edge (u, u^(1<<j)): canonical if bit j of u is 0
                if u & (1 << j):
                    # canonical rep is (u^(1<<j), j)
                    e1 = (u ^ (1 << j)) * d + j
                else:
                    e1 = u * d + j
                # Edge (u^(1<<i), u^(1<<i)^(1<<j)): canonical if bit j of u^(1<<i) is 0
                v1 = u ^ (1 << i)
                if v1 & (1 << j):
                    e2 = (v1 ^ (1 << j)) * d + j
                else:
                    e2 = v1 * d + j
                # Edge (u^(1<<j), u^(1<<i)^(1<<j)): canonical if bit i of u^(1<<j) is 0
                v2 = u ^ (1 << j)
                if v2 & (1 << i):
                    e3 = (v2 ^ (1 << i)) * d + i
                else:
                    e3 = v2 * d + i
                c4_partners[eid].append((e1, e2, e3))
    
    # All edge ids
    all_edge_ids = [u * d + i for u in range(n) for i in range(d) if not (u & (1 << i))]
    
    def greedy_random(seed):
        rng = random.Random(seed)
        order = all_edge_ids[:]
        rng.shuffle(order)
        present = bytearray(n * d)  # present[eid] = 1 if edge is selected
        count = 0
        for eid in order:
            ok = True
            for (e1, e2, e3) in c4_partners[eid]:
                if present[e1] and present[e2] and present[e3]:
                    ok = False
                    break
            if ok:
                present[eid] = 1
                count += 1
        return present, count
    
    def local_improve(present, count, deadline):
        """Try to improve by removing edges that are in many C4s and re-adding."""
        # For each edge, count how many C4s it participates in (among present edges)
        # Actually, let's do a simpler local search:
        # Try removing each edge and see if we can add more edges back.
        
        # First, let's try a simple approach: for each edge in the set, 
        # remove it, then greedily add back all rejected edges.
        # But that's expensive. Let's try a targeted approach.
        
        # Actually, let's try: randomly remove a few edges, then re-greedy the rest.
        # Or: try all single-edge removals + re-greedy.
        
        best_present = present
        best_count = count
        
        # Try removing each edge and re-greedy
        present_list = list(present)
        
        for eid in all_edge_ids:
            if time.time() > deadline - 5:
                break
            if not present_list[eid]:
                continue
            # Remove eid
            present_list[eid] = 0
            cur_count = count - 1
            
            # Re-greedy: try to add back all edges not in present_list
            # in random order
            rng = random.Random(eid)  # deterministic per eid
            rejected = [e for e in all_edge_ids if not present_list[e]]
            rng.shuffle(rejected)
            
            for e in rejected:
                ok = True
                for (e1, e2, e3) in c4_partners[e]:
                    if present_list[e1] and present_list[e2] and present_list[e3]:
                        ok = False
                        break
                if ok:
                    present_list[e] = 1
                    cur_count += 1
            
            if cur_count > best_count:
                best_count = cur_count
                best_present = bytearray(present_list)
            
            # Restore
            present_list[eid] = 1
        
        return best_present, best_count
    
    # Multi-restart greedy
    best_present = None
    best_count = 0
    seed = 42
    
    # Do several random greedy restarts
    while time.time() < deadline - 20:
        present, count = greedy_random(seed)
        if count > best_count:
            best_count = count
            best_present = present
        seed += 1
    
    # Local improvement on the best
    if best_present is not None and time.time() < deadline - 5:
        best_present, best_count = local_improve(best_present, best_count, deadline)
    
    # Convert to output format
    out = []
    for eid in all_edge_ids:
        if best_present[eid]:
            u, v = edge_from_id(eid)
            out.append([u, v])
    
    return out
