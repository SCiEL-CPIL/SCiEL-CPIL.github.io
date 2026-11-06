import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 16
    # adj[u] is a bitmask of neighbors
    adj = [0] * n
    
    # Precompute subsets of size 8
    # Map subset index to list of vertices
    subsets = []
    subset_indices_by_edge = { (u,v): [] for u in range(n) for v in range(u+1, n) }
    # Also need reverse mapping or just iterate all subsets for full check
    
    idx = 0
    for S in itertools.combinations(range(n), 8):
        subsets.append(S)
        # For each pair in S, add this subset index to their list
        for i in range(8):
            for j in range(i+1, 8):
                u, v = S[i], S[j]
                if u > v: u, v = v, u
                subset_indices_by_edge[(u,v)].append(idx)
        idx += 1
        
    # Precompute permutation masks for Q3
    # Q3 vertices 0..7
    # Edges of Q3:
    # 0-1, 0-2, 0-4, 0-5
    # 1-3, 1-5, 1-6
    # 2-3, 2-6, 2-7
    # 4-5, 4-7, 4-6
    # 3-7, 3-6
    # Let's generate them programmatically
    q3_edges = []
    for i in range(8):
        # i is binary 000..111
        # neighbors differ by 1 bit
        for b in range(3):
            if not (i & (1 << b)):
                j = i ^ (1 << b)
                if i < j:
                    q3_edges.append((i, j))
    # q3_edges has 12 edges? No, 8 vertices * 3 / 2 = 12 edges?
    # Wait. Q3 has 8 vertices. Each degree 3. Total edges = 8*3/2 = 12.
    # My previous thought of 8 edges was wrong. K_4,4 has 16. Q3 is a subgraph.
    # Let's re-verify Q3 structure.
    # Cube has 12 edges.
    
    # So we need to check for 12 edges.
    # Permutations: 8! = 40320.
    # Aut(Q3) size is 48.
    # Distinct mappings: 40320 / 48 = 840.
    
    perm_masks = []
    # Generate all permutations of 0..7
    # This might be slow to generate 40320 permutations in Python at startup?
    # 40320 is small.
    for p in itertools.permutations(range(8)):
        # Check if p is a valid isomorphism?
        # No, we just need the set of edges induced by p on Q3 edges.
        # Actually, any permutation of vertices defines a mapping.
        # We want the mask of edges in the target graph (0..7) that correspond to Q3 edges.
        # If we map Q3 vertex i to p[i], then edge (u,v) in Q3 becomes (p[u], p[v]).
        # We want to check if the induced graph on S contains these edges.
        # So for a given permutation p, we compute the mask of edges (p[u], p[v]) for (u,v) in Q3.
        mask = 0
        for u, v in q3_edges:
            pu, pv = p[u], p[v]
            if pu > pv: pu, pv = pv, pu
            # Encode edge (pu, pv) in 28-bit mask
            # Index: pu * 8 + pv? No, total 28 edges.
            # Let's use a standard encoding: idx = pu * 16 + pv? No.
            # Map (a,b) with a<b to a*8 + b? Max 7*8+7 = 63. Too big for 28 bits?
            # 28 edges. We can map (a,b) to a specific bit.
            # Precompute edge_bit[(a,b)]
            pass
            
    # Let's precompute edge_bit for all 28 pairs in 0..7
    edge_bit = {}
    bit_idx = 0
    for a in range(8):
        for b in range(a+1, 8):
            edge_bit[(a,b)] = 1 << bit_idx
            bit_idx += 1
            
    for p in itertools.permutations(range(8)):
        mask = 0
        for u, v in q3_edges:
            pu, pv = p[u], p[v]
            if pu > pv: pu, pv = pv, pu
            mask |= edge_bit[(pu, pv)]
        # Add to set to remove duplicates (automorphisms produce same mask)
        if mask not in perm_masks:
            perm_masks.append(mask)
            
    # perm_masks should have 840 entries?
    # Actually, distinct masks might be fewer if some permutations result in same edge set?
    # No, the edge set of Q3 is unique for each isomorphism class.
    # Wait, if p1 and p2 are related by an automorphism of Q3, they produce the same mask?
    # Yes. So the number of unique masks is 840.
    
    # Function to check if graph (adj) on subset S contains Q3
    def has_q3_in_subset(adj, S):
        # Map S to 0..7
        # S is a tuple of 8 vertices
        # Build mask of edges in G[S]
        mask = 0
        for i in range(8):
            for j in range(i+1, 8):
                u, v = S[i], S[j]
                if adj[u] & (1 << v):
                    # Edge exists
                    # Need to map (S[i], S[j]) to (i, j) in local coords?
                    # No, the mask is defined on local indices 0..7.
                    # So if edge S[i]-S[j] exists, set bit for (i,j)
                    mask |= edge_bit[(i, j)]
        # Check if any perm mask is a subset of mask
        for pm in perm_masks:
            if (mask & pm) == pm:
                return True
        return False
        
    # Function to check if adding edge (u,v) creates Q3
    def is_valid_add(adj, u, v):
        # Only check subsets containing u and v
        for idx in subset_indices_by_edge.get((u,v), []):
            S = subsets[idx]
            if has_q3_in_subset(adj, S):
                return False
        return True
        
    # Initialize graph
    # Start with random graph p=0.5
    random.seed(42)
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # Cleanup: Remove edges to make it valid
    # We need a function to find ANY Q3 in the whole graph
    def find_q3(adj):
        for idx, S in enumerate(subsets):
            if has_q3_in_subset(adj, S):
                return S
        return None
        
    # Iteratively remove edges until valid
    while time.time() < deadline - 5.0:
        S = find_q3(adj)
        if S is None:
            break
        # Pick a random edge in S to remove
        # Get edges present in S
        edges_in_S = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = S[i], S[j]
                if adj[u] & (1 << v):
                    edges_in_S.append((u, v))
        if not edges_in_S:
            break # Should not happen
        u, v = random.choice(edges_in_S)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Local Search
    best_edges = sum(bin(adj[u]).count('1') for u in range(n)) // 2
    current_edges = best_edges
    
    while time.time() < deadline:
        # Try to add a random non-edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        if adj[u] & (1 << v):
            continue # Already edge
            
        # Check validity
        if is_valid_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
        else:
            # Failed to add
            # Maybe remove a random edge to unblock?
            # Only do this if we've been failing a lot?
            # Simple heuristic: with small probability, remove a random edge
            if random.random() < 0.05:
                # Pick random edge
                eu = random.randint(0, n-1)
                ev = random.randint(0, n-1)
                if eu != ev and adj[eu] & (1 << ev):
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
                    current_edges -= 1

    # Return list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    return edges
