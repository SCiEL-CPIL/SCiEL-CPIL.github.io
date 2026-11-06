import time
import random

def construct():
    n = 20
    deadline = time.time() + 140.0
    best_edges = 0
    best_masks = [0] * n
    
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        # Random initialization
        p = random.uniform(0.4, 0.8)
        masks = [0] * n
        for u, v in pairs:
            if random.random() < p:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
        # Repair phase
        # We need to remove edges to eliminate all violations.
        # To avoid infinite loops, limit iterations or just break if no progress.
        max_repair_steps = 1000
        for _ in range(max_repair_steps):
            # Find a violation
            violation_found = False
            # Shuffle triples to randomize repair order
            random.shuffle(triples)
            for u, v, w in triples:
                common = masks[u] & masks[v] & masks[w]
                if common.bit_count() > 2:
                    violation_found = True
                    # Pick a random common neighbor c
                    # And a random vertex from {u, v, w}
                    # Remove edge (vertex, c)
                    # Actually, removing any edge between U and C works.
                    # Let's pick a random edge from the set of edges connecting U and C.
                    # Edges are (u, c), (v, c), (w, c) for c in C.
                    # We need to ensure the edge exists.
                    # Since c is a common neighbor, all these edges exist.
                    
                    # Get list of common neighbors
                    common_list = []
                    temp = common
                    while temp:
                        lsb = temp & -temp
                        c = lsb.bit_length() - 1
                        common_list.append(c)
                        temp ^= lsb
                    
                    c = random.choice(common_list)
                    vertex = random.choice([u, v, w])
                    
                    # Remove edge (vertex, c)
                    masks[vertex] &= ~(1 << c)
                    masks[c] &= ~(1 << vertex)
                    break
            
            if not violation_found:
                break
        
        # Greedy addition phase
        # Try to add all missing edges
        # Shuffle pairs
        current_pairs = pairs[:]
        random.shuffle(current_pairs)
        for u, v in current_pairs:
            if masks[u] & (1 << v):
                continue # Edge already exists
            
            mask_u_new = masks[u] | (1 << v)
            mask_v_new = masks[v] | (1 << u)
            
            valid = True
            
            # Check triples containing u
            for x in range(n):
                if x == u: continue
                for y in range(x+1, n):
                    if y == u: continue
                    common = mask_u_new & masks[x] & masks[y]
                    if common.bit_count() > 2:
                        valid = False
                        break
                if not valid: break
            
            if not valid:
                continue
                
            # Check triples containing v but not u
            for x in range(n):
                if x == v or x == u: continue
                for y in range(x+1, n):
                    if y == v or y == u: continue
                    common = mask_v_new & masks[x] & masks[y]
                    if common.bit_count() > 2:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                masks[u] = mask_u_new
                masks[v] = mask_v_new
                
        # Count edges
        edge_count = 0
        for i in range(n):
            edge_count += masks[i].bit_count()
        edge_count //= 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_masks = masks[:]
            
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_masks[i] & (1 << j):
                edges.append([i, j])
                
    return edges
