import random
import time
import math

def construct():
    d = 7
    n = 2 ** d
    total_edges = n * d // 2
    
    # Generate all edges and map them to indices for faster state representation
    edges = []
    edge_to_idx = {}
    idx_to_edge = []
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                idx_to_edge.append((x, y))
    
    E = len(edges)
    
    # Precompute for each edge, the list of other edges that form a C4 with it.
    # If edge e is in the set, and any of its C4-partners are also in the set, 
    # we must ensure the *other two* edges of the C4 are NOT in the set?
    # No, the condition is "No 2-dimensional face may have all four edges selected".
    # So for each C4 face F = {e1, e2, e3, e4}, we cannot have all 4 in the set.
    # This is a 4-uniform hypergraph independent set problem.
    
    # To speed up checking, for each edge e, we can store the list of C4s it belongs to.
    # A C4 is identified by the two bit positions (i, j) and the base vertex.
    # Actually, simpler: For each edge e=(u,v) flipping bit k, 
    # it belongs to d-1 faces. Each face is determined by another bit i.
    # The face contains edges:
    # e_k: (u, u^ek)
    # e_i: (u, u^ei)
    # e_k': (u^ei, u^ei^ek)
    # e_i': (u^ek, u^ek^ei)
    
    # Let's precompute the "conflict groups".
    # For each edge index e, store a list of tuples (face_id, edge_indices_in_face).
    # Or simpler: For each edge e, store a list of lists of 3 other edges that form a C4 with e.
    # If e is selected, and any one of these "triples" is fully selected, we have a C4.
    # Wait, if e is selected, a C4 is formed if the other 3 edges of a specific face are selected.
    
    c4_partners = [[] for _ in range(E)]
    
    for x in range(n):
        for k in range(d):
            v = x ^ (1 << k)
            if x >= v: continue
            e_idx = edge_to_idx[(x, v)]
            
            for i in range(d):
                if i == k: continue
                w = x ^ (1 << i)
                z = v ^ (1 << i)
                
                # The face edges are:
                # (x, v) -> e_idx
                # (x, w)
                # (v, z)
                # (w, z)
                
                e1 = edge_to_idx[(min(x, w), max(x, w))]
                e2 = edge_to_idx[(min(v, z), max(v, z))]
                e3 = edge_to_idx[(min(w, z), max(w, z))]
                
                # Store these 3 edges as partners for e_idx
                # We need to check if e1, e2, e3 are ALL selected.
                c4_partners[e_idx].append((e1, e2, e3))
    
    # State: list of booleans
    # Score: number of True bits
    
    start_time = time.time()
    max_time = 115.0
    
    # Initialize with a random valid set or empty set?
    # Greedy initialization is usually better to start near the optimum.
    # Let's do a fast greedy init first.
    
    def greedy_init():
        state = [False] * E
        # Random order
        indices = list(range(E))
        random.shuffle(indices)
        for idx in indices:
            u, v = idx_to_edge[idx]
            # Check if adding idx creates a C4
            # It creates a C4 if for any partner tuple (a,b,c) in c4_partners[idx],
            # state[a] and state[b] and state[c] are all True.
            ok = True
            for a, b, c in c4_partners[idx]:
                if state[a] and state[b] and state[c]:
                    ok = False
                    break
            if ok:
                state[idx] = True
        return state
    
    best_state = greedy_init()
    best_score = sum(best_state)
    
    # Simulated Annealing
    current_state = best_state[:]
    current_score = best_score
    
    T = 1.0
    cooling = 0.999
    
    # Moves:
    # 1. Add a random False edge (if valid)
    # 2. Remove a random True edge
    # 3. Swap: Remove True, Add False (if valid)
    
    # To make moves efficient, we can just pick a random edge and toggle it?
    # Toggling an edge e:
    # If turning ON: check if it creates a C4. (Check c4_partners[e])
    # If turning OFF: always valid.
    
    # We need to track the number of True edges to know score.
    
    while time.time() - start_time < max_time:
        # Pick a random move
        # Bias towards adding edges if we are below best, or exploring?
        # Standard SA: pick random edge, toggle it.
        idx = random.randint(0, E - 1)
        
        if current_state[idx]:
            # Turn OFF
            new_state = current_state[:] # Copy is expensive? E=448. 448 bytes. Fast enough.
            new_state[idx] = False
            new_score = current_score - 1
            delta = new_score - current_score # -1
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_state = new_state
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_state = current_state[:]
        else:
            # Turn ON
            # Check validity
            valid = True
            for a, b, c in c4_partners[idx]:
                if current_state[a] and current_state[b] and current_state[c]:
                    valid = False
                    break
            
            if valid:
                new_state = current_state[:]
                new_state[idx] = True
                new_score = current_score + 1
                delta = new_score - current_score # +1
                # Always accept improvements
                current_state = new_state
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_state = current_state[:]
            # If invalid, we cannot turn it on. 
            # In SA, we usually propose a move and check if it's allowed. 
            # If the move is "Add edge" and it's invalid, the move is rejected.
            # This means we only propose valid adds or any removes.
            # This is a "Valid-Only" SA variant.
            
        T *= cooling
        if T < 0.01:
            # Reheat occasionally? Or just let it converge.
            # Given time limit, simple cooling is fine.
            pass

    # Convert best_state to list of edges
    result = []
    for i, is_selected in enumerate(best_state):
        if is_selected:
            result.append(list(idx_to_edge[i]))
            
    return result
