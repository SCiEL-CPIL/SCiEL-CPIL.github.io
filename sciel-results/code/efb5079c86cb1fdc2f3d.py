import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports) # 330
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # 2. Build conflict graph for supports
    # Conflict if intersection size is 3
    # We build the adjacency list for the conflict graph G_S
    # We will find Max Independent Set in G_S
    adj = [set() for _ in range(N_S)]
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            # Check intersection size
            # supports[i] and supports[j] are tuples
            inter = len(set(supports[i]) & set(supports[j]))
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Find Max Independent Set in G_S
    # Using Local Search (Hill Climbing with Swaps)
    
    best_set = set()
    best_size = 0
    
    rng = np.random.RandomState(42)
    
    def greedy_init():
        avail = list(range(N_S))
        rng.shuffle(avail)
        s = set()
        for v in avail:
            # Check if v conflicts with any in s
            # Since s is small, we can just check
            # Or maintain a list of blocked nodes
            # For simplicity, check against s
            # Optimization: if any neighbor in s, skip
            # We can precompute neighbors
            if not any(nb in s for nb in adj[v]):
                s.add(v)
        return s

    def local_search(initial_set):
        current_set = set(initial_set)
        current_size = len(current_set)
        
        # Precompute degrees or just use adj
        # We want to improve by adding, removing, or swapping
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # 1. Try to add any compatible node
            # Candidates: nodes not in set, not adjacent to any node in set
            # Efficiently: 
            # blocked = union of neighbors of nodes in current_set
            # candidates = all_nodes - current_set - blocked
            
            # For N=330, this is fast enough to do simply
            in_set = np.zeros(N_S, dtype=bool)
            for v in current_set:
                in_set[v] = True
                
            blocked = np.zeros(N_S, dtype=bool)
            for v in current_set:
                for nb in adj[v]:
                    blocked[nb] = True
            
            candidates = np.where(~in_set & ~blocked)[0]
            
            if len(candidates) > 0:
                # Pick one randomly or the one with lowest degree?
                # Just pick first or random
                v_add = candidates[rng.randint(len(candidates))]
                current_set.add(v_add)
                current_size += 1
                improved = True
                continue
            
            # 2. Try 1-swap: remove v, add u
            # For each v in current_set:
            #   Remove v.
            #   Find compatible u not in set (excluding v's neighbors? No, u just needs to be compatible with new set).
            #   If found, swap.
            
            # To speed up, we can try swaps that increase size by 1 (remove 1, add 2)
            # But standard hill climbing is 1-swap (remove 1, add 1) which keeps size same,
            # or remove 1, add 0 (decrease), or add 1 (increase).
            # Since we already tried adding 1, we try swaps that might lead to better local optima.
            
            # Let's try 1-swap (remove 1, add 1) to escape local minima
            # And 2-add (remove 1, add 2) to increase size
            
            # 2a. Remove 1, Add 2
            # For each v in current_set:
            #   Temporarily remove v
            #   Find 2 compatible nodes u1, u2
            #   If found, add them
            
            # This is expensive.
            # Let's stick to simple 1-swap and random restarts for now.
            # With 330 nodes, greedy is likely very good.
            
            # Actually, let's just do Random Restarts with Greedy + Local Add.
            # If we can't add, break.
            break
            
        return current_set, current_size

    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        init_set = greedy_init()
        final_set, size = local_search(init_set)
        
        if size > best_size:
            best_size = size
            best_set = final_set
            
        # Optional: Try to improve best_set further with more aggressive local search?
        # For now, rely on restarts.
        
    # 4. Construct vectors from best_set supports
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        # Generate all 16 sign patterns
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        # Fallback if something went wrong
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    # 5. Verify
    # Check min distance >= max norm
    # Norm squared = 4
    # Min dist squared >= 4
    # dist^2 = 8 - 2 dot >= 4 => dot <= 2
    # We ensured supports are independent (intersection <= 2), so dot <= 2 is guaranteed?
    # If intersection <= 2, max dot is 2.
    # If intersection < 2, max dot is < 2.
    # So yes, valid.
    
    return final_vectors
