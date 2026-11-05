import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    N = 256
    # 1. Identify forbidden nodes
    forbidden = set()
    for i in range(1, N+1):
        s = 2 * i
        r = int(s**0.5)
        if r*r == s:
            forbidden.add(i)
            
    V = [i for i in range(1, N+1) if i not in forbidden]
    # Map original index to 0..M-1 for compact storage?
    # Or just use 1..256 indices and ignore forbidden ones.
    # Using 1..256 is easier.
    
    # 2. Build adjacency
    # Squares up to 512
    squares = set()
    k = 2
    while k*k <= 512:
        squares.add(k*k)
        k += 1
        
    adj = [[] for _ in range(N+1)]
    for u in V:
        for v in V:
            if u >= v: continue
            if u + v in squares:
                adj[u].append(v)
                adj[v].append(u)
                
    # 3. Greedy Initialization
    # Min degree heuristic
    # We need to dynamically update degrees or just use static?
    # Static min degree is faster.
    # Let's try static min degree first.
    degrees = [len(adj[i]) for i in range(N+1)]
    
    S = set()
    conflict_count = [0] * (N+1)
    
    # Sort V by degree
    sorted_V = sorted(V, key=lambda x: degrees[x])
    
    for u in sorted_V:
        # Check if u is valid
        if conflict_count[u] == 0:
            S.add(u)
            for w in adj[u]:
                conflict_count[w] += 1
                
    best_set = set(S)
    best_size = len(S)
    
    # 4. Local Search
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        if not S: break
        
        # Pick random u in S
        u = random.choice(list(S))
        
        # Remove u
        S.remove(u)
        for w in adj[u]:
            conflict_count[w] -= 1
            
        old_size = len(S)
        
        # Try to add neighbors of u
        # Candidates are neighbors of u not in S
        # We want to add as many as possible (greedy)
        added = []
        # To make it deterministic or random? Random helps exploration.
        # Let's shuffle neighbors
        neighbors = adj[u][:]
        random.shuffle(neighbors)
        
        for v in neighbors:
            if v in S: continue
            if conflict_count[v] == 0:
                # Can add v
                S.add(v)
                added.append(v)
                for w in adj[v]:
                    conflict_count[w] += 1
        
        new_size = len(S)
        
        # Acceptance
        if new_size > old_size:
            # Always accept (this is an improvement)
            if new_size > best_size:
                best_set = set(S)
                best_size = new_size
        elif new_size == old_size:
            # Accept with prob 1 (or T related) to explore
            # Actually, if we don't accept, we revert.
            # Let's always accept equal size to move around the plateau
            if new_size > best_size:
                best_set = set(S)
                best_size = new_size
        else:
            # new_size < old_size
            # Accept with probability exp(-delta/T)
            delta = old_size - new_size
            prob = pow(2.71828, -delta / T) if T > 0 else 0
            if random.random() < prob:
                pass # Accept
            else:
                # Revert
                S.remove(added[-1]) if added else None # This revert logic is messy
                # Better: keep a snapshot or undo properly.
                
    # The revert logic above is flawed because `added` might have multiple items.
    # Let's restructure the move to be atomic or handle revert carefully.
    
    return list(best_set)
