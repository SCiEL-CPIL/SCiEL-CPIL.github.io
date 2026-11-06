import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    
    # Precompute Q3 masks for K_4,4
    # Vertices 0-3 in A, 4-7 in B
    # Edge (i, j) where i in [0,3], j in [4,7] maps to bit i*4 + (j-4)
    # Total 16 bits.
    
    def is_q3_mask(mask):
        # Check if the bipartite graph defined by mask is Q3
        # Q3 is 3-regular
        deg_a = [0]*4
        deg_b = [0]*4
        for i in range(4):
            for j in range(4):
                if mask & (1 << (i*4 + j)):
                    deg_a[i] += 1
                    deg_b[j] += 1
        if all(d == 3 for d in deg_a) and all(d == 3 for d in deg_b):
            return True
        return False

    q3_masks = []
    all_edges = list(range(16))
    for combo in itertools.combinations(all_edges, 12):
        mask = 0
        for idx in combo:
            mask |= (1 << idx)
        if is_q3_mask(mask):
            q3_masks.append(mask)
            
    # adj[i] is a bitmask of neighbors
    adj = [0] * n
    
    # Initialize with Complement of Shrikhande graph
    # Shrikhande graph is 5-regular (40 edges).
    # It is constructed from K_4 square K_4 (6-regular, 48 edges) by removing a perfect matching.
    # The complement is 10-regular (80 edges).
    
    def idx(i, j):
        return i * 4 + j
        
    # Build full grid graph K_4 square K_4
    for i in range(4):
        for j in range(4):
            u = idx(i, j)
            # Row neighbors
            for j2 in range(4):
                if j2 != j:
                    v = idx(i, j2)
                    if u < v:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            # Col neighbors
            for i2 in range(4):
                if i2 != i:
                    v = idx(i2, j)
                    if u < v:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
    # Now remove a perfect matching to get Shrikhande (5-regular)
    # Matching: (0,0)-(1,1), (0,1)-(1,0), (2,0)-(3,1), (2,1)-(3,0),
    #           (0,2)-(1,3), (0,3)-(1,2), (2,2)-(3,3), (2,3)-(3,2)
    matching = [
        (idx(0,0), idx(1,1)),
        (idx(0,1), idx(1,0)),
        (idx(2,0), idx(3,1)),
        (idx(2,1), idx(3,0)),
        (idx(0,2), idx(1,3)),
        (idx(0,3), idx(1,2)),
        (idx(2,2), idx(3,3)),
        (idx(2,3), idx(3,2))
    ]
    for u, v in matching:
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Now take the COMPLEMENT to get a dense graph (10-regular, 80 edges)
    for u in range(n):
        for v in range(u+1, n):
            if not (adj[u] & (1 << v)):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # Precompute checks for Q3 detection
    # For each 8-subset, and each 4-split
    precomputed_checks = []
    subsets_8 = list(itertools.combinations(range(n), 8))
    for sub in subsets_8:
        for A in itertools.combinations(sub, 4):
            B = tuple(v for v in sub if v not in A)
            pairs = []
            for i in range(4):
                a = A[i]
                for j in range(4):
                    b = B[j]
                    pairs.append((a, b, i*4 + j))
            precomputed_checks.append(pairs)
            
    # Shuffle checks for randomization in finding Q3
    # We will create a list of indices and shuffle it in find_q3
    
    def find_q3_edge(adj):
        # Returns an edge (u, v) in a Q3, or None
        # Use random indices to find Q3 faster on average
        indices = list(range(len(precomputed_checks)))
        random.shuffle(indices)
        
        for idx in indices:
            pairs = precomputed_checks[idx]
            mask = 0
            # Quick degree check?
            # We can build mask and then check.
            for a, b, mb in pairs:
                if adj[a] & (1 << b):
                    mask |= (1 << mb)
            
            # Check if mask contains any Q3
            # Optimization: Check degrees first?
            # deg_a[0] = popcount(mask & 0b1111000000000000) >> 12? No.
            # Bit 0..3: a=0. Bit 4..7: a=1. etc.
            # a=0 bits: 0, 4, 8, 12.
            # a=1 bits: 1, 5, 9, 13.
            # a=2 bits: 2, 6, 10, 14.
            # a=3 bits: 3, 7, 11, 15.
            
            # If any degree < 3, skip.
            # deg_a[i] = bin(mask & (0b1111 << (i*4))).count('1')? No, bits are i, i+4, i+8, i+12.
            # Mask for a=0: 0b1111000000000000? No.
            # Bit i*4 + j.
            # For a=0 (i=0), bits are 0, 4, 8, 12.
            # Mask0 = (1<<0)|(1<<4)|(1<<8)|(1<<12) = 0x1111
            # For a=1 (i=1), bits are 1, 5, 9, 13.
            # Mask1 = 0x2222
            # For a=2 (i=2), bits are 2, 6, 10, 14.
            # Mask2 = 0x4444
            # For a=3 (i=3), bits are 3, 7, 11, 15.
            # Mask3 = 0x8888
            
            ma = mask & 0x1111
            if bin(ma).count('1') < 3: continue
            ma = mask & 0x2222
            if bin(ma).count('1') < 3: continue
            ma = mask & 0x4444
            if bin(ma).count('1') < 3: continue
            ma = mask & 0x8888
            if bin(ma).count('1') < 3: continue
            
            # Check B degrees
            # b=0 (j=0), bits 0, 1, 2, 3. Mask 0x000F
            # b=1 (j=1), bits 4, 5, 6, 7. Mask 0x00F0
            # b=2 (j=2), bits 8, 9, 10, 11. Mask 0x0F00
            # b=3 (j=3), bits 12, 13, 14, 15. Mask 0xF000
            
            mb = mask & 0x000F
            if bin(mb).count('1') < 3: continue
            mb = mask & 0x00F0
            if bin(mb).count('1') < 3: continue
            mb = mask & 0x0F00
            if bin(mb).count('1') < 3: continue
            mb = mask & 0xF000
            if bin(mb).count('1') < 3: continue
            
            # Now check against q3_masks
            for qm in q3_masks:
                if (mask & qm) == qm:
                    # Found Q3
                    # Return an edge from this Q3
                    for a, b, mbit in pairs:
                        if qm & (1 << mbit):
                            return (a, b)
                    return (pairs[0][0], pairs[0][1])
                    
        return None

    # Phase 1: Remove edges to make Q3-free
    max_iter = 1000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        edge = find_q3_edge(adj)
        if edge is None:
            break
        u, v = edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Phase 2: Addition
    # Try to add missing edges
    missing_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if not (adj[u] & (1 << v)):
                missing_edges.append((u, v))
                
    # Multiple passes of addition
    # We will do more passes if time permits, or until no more edges can be added
    while time.time() < deadline - 5:
        random.shuffle(missing_edges)
        added_any = False
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if Q3 exists
            edge_in_q3 = find_q3_edge(adj)
            
            if edge_in_q3 is not None:
                # Q3 found. Remove the new edge.
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Keep the edge
                added_any = True
        if not added_any:
            break

    # Convert adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges

# Run once to ensure no syntax errors
# construct()
