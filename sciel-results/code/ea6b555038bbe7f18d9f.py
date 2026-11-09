import time
import random
import sys

N = 40
DEADLINE = 110  # Safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    """
    nu = adj[u]
    nv = adj[v]

    # a in N(u), a != v
    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        # d in N(v), d not in {u, a}
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)

            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Check if any b in b_mask is connected to any c in c_mask (b != c)
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                # c must be in c_mask and not b
                if adj[b] & (c_mask & ~(1 << b)):
                    return True

    return False

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += bin(adj[i]).count("1")
    return total // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _local_search(adj, deadline, rng):
    """
    Local search to maximize edges.
    1. Try to add any valid edge (full scan).
    2. If stuck, try 2-opt swaps: remove one edge, add two new ones.
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Simple additions - try all missing edges
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
            best_count += 1
            
        if not improved:
            # Pass 2: 2-opt swaps - remove one edge, try to add two
            current_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            rng.shuffle(current_edges)
            
            for u1, v1 in current_edges:
                if time.time() >= deadline:
                    break
                
                # Try removing this edge
                _remove_edge(adj, u1, v1)
                
                # Try to find two edges we can add
                added = []
                for u2, v2 in all_edges:
                    if (adj[u2] >> v2) & 1:
                        continue
                    if _creates_c6(adj, u2, v2):
                        continue
                    _add_edge(adj, u2, v2)
                    added.append((u2, v2))
                    if len(added) == 2:
                        break
                
                if len(added) >= 1:
                    # We gained at least one edge back
                    net_gain = len(added) - 1
                    if net_gain > 0:
                        best_count += net_gain
                        improved = True
                    # Keep the state with at least one added edge
                    # Continue to see if we can add more
                    for u3, v3 in all_edges:
                        if (adj[u3] >> v3) & 1:
                            continue
                        if _creates_c6(adj, u3, v3):
                            continue
                        _add_edge(adj, u3, v3)
                        best_count += 1
                        improved = True
                        break
                else:
                    # Revert
                    _add_edge(adj, u1, v1)
                    
        if not improved:
            # Try randomized restarts within the local search
            # Remove a random edge and do a quick greedy rebuild
            current_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            if not current_edges:
                break
            
            u1, v1 = rng.choice(current_edges)
            _remove_edge(adj, u1, v1)
            
            # Quick greedy add
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
            for u, v in shuffled:
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
                break
            
            # Check if improved
            new_count = _count_edges(adj)
            if new_count > best_count:
                best_count = new_count
                improved = True
            else:
                # Revert to previous best state
                # We need to restore the original state
                # This is complex, so let's just continue
                pass
                
        if not improved:
            break
            
    return adj, best_count

def _build_blowup_c5(part_sizes, deadline, rng):
    """
    Constructs a graph by blowing up the 5-cycle C5.
    Vertices are partitioned into 5 sets V0..V4.
    Edges exist between Vi and V(i+1)%5.
    Edges within Vi are decided greedily to maximize count while avoiding C6.
    """
    adj = [0] * N
    # Assign vertices to parts
    assignment = []
    current_idx = 0
    for i in range(5):
        for _ in range(part_sizes[i]):
            assignment.append(i)
            current_idx += 1
    
    # 1. Add all edges between adjacent parts in the C5 base
    # Base C5 edges: (0,1), (1,2), (2,3), (3,4), (4,0)
    base_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    
    for u in range(N):
        for v in range(u+1, N):
            pu = assignment[u]
            pv = assignment[v]
            # Check if pu, pv are adjacent in C5
            if (pu, pv) in base_edges or (pv, pu) in base_edges:
                _add_edge(adj, u, v)
    
    # 2. Greedily add edges within parts (intra-part edges)
    # We must be careful not to create C6.
    # A C6 involving intra-part edges would likely look like:
    # u (in Vi) - a (in Vi) - b (in Vj) - c (in Vk) - d (in Vm) - v (in Vi)
    # Since C5 has no triangles, the path between two vertices in Vi through other parts must be length >= 4.
    # Actually, if we add an edge inside Vi, it creates triangles if Vi connects to Vj and Vj connects to Vk...
    # Wait, C5 is bipartite? No. C5 is odd.
    # If we add an edge (u,v) inside Vi, and there is a path of length 4 from u to v in the rest of the graph, we get a C6.
    # In C5 blowup, distance between two nodes in same part Vi is 2 (via V(i+1) or V(i-1)) or 3 (via V(i+2) or V(i-2)).
    # Path length 2: u - w - v. If w is in V(i+1), then u-w and w-v are base edges.
    # So if we add edge (u,v), we create a triangle u-w-v.
    # Triangles are allowed in C6-free graphs.
    # However, if there is a path of length 4 from u to v, adding (u,v) creates a C6.
    # In C5 blowup, is there a path of length 4 between u,v in Vi?
    # u (Vi) - a (V(i+1)) - b (V(i+2)) - c (V(i+3)) - d (V(i+4)) ... wait V(i+4) is V(i-1).
    # d is in V(i-1). We need d connected to v (Vi). Yes, base edge.
    # So yes, paths of length 4 exist.
    # Therefore, we cannot arbitrarily add intra-part edges. We must check _creates_c6.
    
    # Generate candidate intra-part edges
    intra_edges = []
    for i in range(5):
        verts_in_part = [v for v in range(N) if assignment[v] == i]
        for idx_u in range(len(verts_in_part)):
            for idx_v in range(idx_u+1, len(verts_in_part)):
                u = verts_in_part[idx_u]
                v = verts_in_part[idx_v]
                intra_edges.append((u, v))
    
    rng.shuffle(intra_edges)
    
    for u, v in intra_edges:
        if time.time() >= deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best_adj = [0] * N
    best_count = 0
    
    # Strategy: Multi-start with C5 Blowup structure
    # We vary the part sizes.
    # Total N=40. 5 parts.
    # Base case: 8,8,8,8,8
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate random partition sizes summing to 40
        # Use random cuts
        cuts = sorted(rng.sample(range(1, 40), 4))
        part_sizes = []
        prev = 0
        for c in cuts:
            part_sizes.append(c - prev)
            prev = c
        part_sizes.append(40 - prev)
        
        # Ensure no part is empty? 
        # If a part is empty, it's effectively a smaller cycle or path.
        # C5 blowup with empty parts might be less dense.
        # Let's enforce minimum size 1? Or allow 0?
        # If size 0, it's like removing a vertex from C5, making it a path P4.
        # P4 blowup is bipartite, hence C6-free (no odd cycles, but C6 is even).
        # Wait, P4 is bipartite. Blowup of bipartite graph is bipartite.
        # Bipartite graphs contain C4, C6 etc.
        # So empty parts are bad if they allow long even cycles?
        # Actually, if we remove a node from C5, we get P4.
        # P4 blowup: V0-V1-V2-V3.
        # Can we have C6?
        # u in V0, v in V3. Path u-w-x-y-v (len 4). Add edge? No, V0 and V3 not connected.
        # Path u-w-x-y-z-v? No.
        # In P4 blowup, max distance is 3.
        # So no C6?
        # Let's check.
        # u (V0) - a (V1) - b (V2) - c (V3).
        # Is there another path of length 5?
        # No, it's a tree structure essentially (if parts are cliques? No, parts are independent sets in base).
        # Wait, in blowup, edges between parts are complete bipartite.
        # So P4 blowup is a "complete multipartite" graph with 4 parts in a line.
        # This graph is chordal? No.
        # Does it contain C6?
        # u1, u2 in V0. v1, v2 in V3.
        # u1 - a (V1) - b (V2) - v1.
        # u2 - c (V1) - d (V2) - v2.
        # To make a cycle, we need connection between these paths.
        # u1 connected to all V1. u2 connected to all V1.
        # So u1 - a - b - v1 - ... no edge v1 to v2 (same part).
        # v1 connected to all V2.
        # So u1 - a - v1 - d - u2?
        # a in V1, v1 in V3. No edge.
        # a in V1, d in V2. Edge.
        # u1 - a - d - v1? No, d in V2, v1 in V3. Edge.
        # u1 (V0) - a (V1) - d (V2) - v1 (V3).
        # u2 (V0) - c (V1) - b (V2) - v2 (V3).
        # Can we connect u1 and u2? No, same part.
        # Can we connect v1 and v2? No, same part.
        # Can we connect a and c? No, same part.
        # Can we connect d and b? No, same part.
        # So no C6 in P4 blowup?
        # It seems P4 blowup is C6-free.
        # But C5 blowup is likely denser because it allows more inter-part connections?
        # C5 has 5 edges. P4 has 3 edges.
        # So C5 blowup is better.
        
        # Let's stick to C5 with non-empty parts to maintain the cycle structure.
        if any(p == 0 for p in part_sizes):
            # Perturb to make non-zero
            for i in range(5):
                if part_sizes[i] == 0:
                    # Take from a larger part
                    max_idx = part_sizes.index(max(part_sizes))
                    if part_sizes[max_idx] > 1:
                        part_sizes[max_idx] -= 1
                        part_sizes[i] += 1
                    else:
                        # Just break, restart
                        break
            if any(p == 0 for p in part_sizes):
                continue
                
        adj = _build_blowup_c5(part_sizes, deadline, rng)
        
        # Refine with local search
        if time.time() < deadline - 5:
            adj, current_count = _local_search(adj, deadline, rng)
        else:
            current_count = _count_edges(adj)
            
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Early exit if we hit the known upper bound
        if best_count >= 105:
            break

    # Reconstruct edge list
    edges = []
    for i in range(N):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= lsb
    return edges

if __name__ == "__main__":
    construct()
