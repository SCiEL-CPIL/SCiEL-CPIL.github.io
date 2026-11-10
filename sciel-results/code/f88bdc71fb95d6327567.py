import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    
    # Precompute all 8-subsets
    # Storing as tuples for fast hashing/indexing if needed, but we iterate
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    # Precompute partitions of 8 indices into two sets of 4
    # Indices are relative to the subset: 0..7
    partitions = []
    for a in itertools.combinations(range(8), 4):
        a_set = set(a)
        b_tuple = tuple(i for i in range(8) if i not in a_set)
        partitions.append((a, b_tuple))
    # There are 70 such partitions
    
    def has_q3(adj):
        """
        Check if the graph represented by adj (list of 16 bitmasks) contains Q3.
        Returns True if Q3 is found.
        """
        for verts in subsets_8:
            # Quick check: count edges within induced subgraph
            # If < 12, cannot contain Q3
            edge_count = 0
            for i in range(8):
                u = verts[i]
                # Count bits in adj[u] that are in verts
                mask = 0
                for j in range(8):
                    if j > i: # Avoid double counting? No, just check all
                        v = verts[j]
                        if adj[u] >> v & 1:
                            edge_count += 1
                # Simpler: just sum degrees and divide by 2?
                # But we need to check specific structure.
                pass
            
            # Let's just run the partition check. It's fast enough.
            # Extract induced adjacencies for the 8 vertices
            # We need to check for each partition (A_idx, B_idx)
            for a_idx, b_idx in partitions:
                # A and B are tuples of relative indices
                # Map relative indices to actual vertices
                # A_verts = [verts[i] for i in a_idx]
                # B_verts = [verts[i] for i in b_idx]
                
                # Check if non-edges between A and B form a matching
                used_a = set()
                used_b = set()
                is_matching = True
                # Iterate all pairs (a_rel, b_rel) in A_idx x B_idx
                for a_rel in a_idx:
                    u = verts[a_rel]
                    for b_rel in b_idx:
                        v = verts[b_rel]
                        # Check if edge (u, v) is NOT in graph
                        if not (adj[u] >> v & 1):
                            # This is a non-edge. It must be part of the "missing matching"
                            if a_rel in used_a or b_rel in used_b:
                                is_matching = False
                                break
                            used_a.add(a_rel)
                            used_b.add(b_rel)
                    if not is_matching:
                        break
                
                if is_matching:
                    return True
        return False

    def find_q3_edges(adj):
        """
        Find one Q3 in the graph and return its edges.
        """
        for verts in subsets_8:
            for a_idx, b_idx in partitions:
                used_a = set()
                used_b = set()
                is_matching = True
                missing_pairs = []
                for a_rel in a_idx:
                    u = verts[a_rel]
                    for b_rel in b_idx:
                        v = verts[b_rel]
                        if not (adj[u] >> v & 1):
                            if a_rel in used_a or b_rel in used_b:
                                is_matching = False
                                break
                            used_a.add(a_rel)
                            used_b.add(b_rel)
                            missing_pairs.append((a_rel, b_rel))
                    if not is_matching:
                        break
                
                if is_matching:
                    # Construct the Q3 edges
                    # The Q3 is K_{4,4} minus the missing pairs.
                    # Edges are all (u,v) where u in A, v in B, AND (a_rel,b_rel) not in missing_pairs
                    q3_edges = []
                    for a_rel in a_idx:
                        u = verts[a_rel]
                        for b_rel in b_idx:
                            v = verts[b_rel]
                            if (a_rel, b_rel) not in missing_pairs:
                                q3_edges.append((u, v))
                    return q3_edges
        return []

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def has_q3_with_edge(adj, u, v):
        """
        Check if adding edge (u,v) creates a Q3.
        Only checks 8-subsets containing u and v.
        """
        # Find all 8-subsets containing u and v
        # Other 6 vertices from remaining 14
        other = [x for x in range(n) if x != u and x != v]
        
        # Optimize: Precompute subsets?
        # We can just iterate combinations of other
        for six in itertools.combinations(other, 6):
            verts = tuple(sorted([u, v] + list(six)))
            # We need to check if this specific set of 8 vertices, WITH the new edge (u,v),
            # contains a Q3.
            # Since we are adding (u,v), we must include it in the potential Q3.
            # However, the Q3 might use other edges too.
            # We can temporarily add the edge to a copy? No, that's slow.
            # We can just check the condition on the graph where (u,v) is present.
            
            # We know (u,v) is present.
            # We need to check if there is a partition of these 8 vertices into A,B
            # such that the non-edges in A x B form a matching.
            # Note: The "non-edges" are defined by the current graph PLUS the new edge (u,v).
            
            for a_idx, b_idx in partitions:
                # Map verts to relative indices 0..7
                # We need to know which relative index u and v have
                # But we can just iterate all partitions of the 8 vertices
                # The relative indices in `partitions` are fixed 0..7.
                # We need to map the actual vertices `verts` to relative 0..7.
                # `verts` is a sorted tuple.
                
                used_a = set()
                used_b = set()
                is_matching = True
                for a_rel in a_idx:
                    u_act = verts[a_rel]
                    for b_rel in b_idx:
                        v_act = verts[b_rel]
                        
                        # Check if edge is missing
                        # Edge exists if:
                        # 1. It is in adj
                        # 2. OR it is the new edge (u,v)
                        
                        is_new = (u_act == u and v_act == v) or (u_act == v and v_act == u)
                        
                        if is_new:
                            # Edge is present
                            pass
                        else:
                            # Check adj
                            if not (adj[u_act] >> v_act & 1):
                                # Non-edge
                                if a_rel in used_a or b_rel in used_b:
                                    is_matching = False
                                    break
                                used_a.add(a_rel)
                                used_b.add(b_rel)
                    if not is_matching:
                        break
                if is_matching:
                    return True
        return False

    # Main Search Loop
    best_edges = []
    best_count = 0
    seed = 42
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        random.seed(seed)
        seed += 1
        
        # Initialize random graph
        # Start with high density
        p = 0.7
        adj = [0] * n
        for u, v in all_edges:
            if random.random() < p:
                add_edge(adj, u, v)
        
        current_count = count_edges(adj)
        
        # Phase 1: Remove edges to break all Q3s
        # To avoid infinite loop if stuck, limit iterations
        for _ in range(5000):
            if not has_q3(adj):
                break
            q3_edges = find_q3_edges(adj)
            if not q3_edges:
                break # Should not happen
            # Remove a random edge from the Q3
            u, v = random.choice(q3_edges)
            remove_edge(adj, u, v)
            current_count -= 1
            
        # Phase 2: Local Search (Add edges)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding all missing edges in random order
            missing = [(u, v) for u, v in all_edges if not (adj[u] >> v & 1)]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                if not has_q3_with_edge(adj, u, v):
                    add_edge(adj, u, v)
                    current_count += 1
                    improved = True
                    
        # Phase 3: Kick (Escape local optimum)
        # Remove 5 random edges
        for _ in range(5):
            # Pick a random existing edge
            existing = [(u, v) for u, v in all_edges if (adj[u] >> v & 1)]
            if existing:
                u, v = random.choice(existing)
                remove_edge(adj, u, v)
                current_count -= 1
        
        # Run Local Search again after kick
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = [(u, v) for u, v in all_edges if not (adj[u] >> v & 1)]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                if not has_q3_with_edge(adj, u, v):
                    add_edge(adj, u, v)
                    current_count += 1
                    improved = True
        
        if current_count > best_count:
            best_count = current_count
            best_edges = [(u, v) for u, v in all_edges if (adj[u] >> v & 1)]

    return best_edges
