import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build adjacency list
    adj = [[] for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) == 3:
                adj[i].append(j)
                adj[j].append(i)
                
    # 1. Greedy Initialization (Multiple restarts)
    def greedy_is():
        remaining = list(range(n_sup))
        current_set = set()
        random.shuffle(remaining)
        for v in remaining:
            # Check if v conflicts with any in current_set
            compatible = True
            for u in current_set:
                if u in adj[v]: # O(1) if adj was sets, but adj is lists now.
                    # We need fast lookup.
                    pass
            # This is slow. Let's use the conflict_count logic even for greedy.
            pass
        return current_set

    # Better Greedy using conflict counts
    def greedy_is_fast():
        in_set = [False] * n_sup
        conflict_count = [0] * n_sup
        current_size = 0
        
        # Random order
        order = list(range(n_sup))
        random.shuffle(order)
        
        for v in order:
            # If v has no conflicts with current set, add it
            if conflict_count[v] == 0:
                in_set[v] = True
                current_size += 1
                for u in adj[v]:
                    conflict_count[u] += 1
        
        # Convert to set for return
        return set(i for i, val in enumerate(in_set) if val)

    best_set = set()
    best_size = 0
    start_time = time.time()
    time_limit = 115.0 # Leave 5s for construction
    
    # Run greedy multiple times
    while time.time() - start_time < 5.0:
        s = greedy_is_fast()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # 2. Optimized Simulated Annealing
    # State: in_set (bool list), conflict_count (int list)
    in_set = [False] * n_sup
    conflict_count = [0] * n_sup
    current_size = 0
    
    # Initialize from best_set
    for v in best_set:
        in_set[v] = True
        current_size += 1
        for u in adj[v]:
            conflict_count[u] += 1
            
    # Precompute neighbors as lists for fast iteration
    # adj is already list of lists
    
    T = 1.0
    cooling = 0.9999
    # Number of steps
    steps = 0
    max_steps = 2000000 # Adjust based on speed
    
    # We need a list of candidates not in S?
    # Or just pick random v and check.
    
    while time.time() - start_time < time_limit:
        # Pick random v
        v = random.randint(0, n_sup - 1)
        
        if in_set[v]:
            # Remove v
            # Delta = -1
            # Always accept? Or probabilistic?
            # Removing reduces size. Only useful if it allows adding many later.
            # In SA, we usually accept worsening moves.
            # Let's accept with prob exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                in_set[v] = False
                current_size -= 1
                for u in adj[v]:
                    conflict_count[u] -= 1
        else:
            # v not in S
            # Find neighbors in S
            # N_S(v)
            # We can just check conflict_count[v]?
            # No, conflict_count[v] is the number of neighbors of v that are IN S.
            # So N_S(v) is exactly the set of neighbors u where in_set[u] is True.
            # And len(N_S(v)) == conflict_count[v].
            
            n_conflicts = conflict_count[v]
            
            if n_conflicts == 0:
                # Add v
                in_set[v] = True
                current_size += 1
                for u in adj[v]:
                    conflict_count[u] += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(i for i, val in enumerate(in_set) if val)
            else:
                # Move: Remove N_S(v), Add v
                # Delta = 1 - n_conflicts
                delta = 1 - n_conflicts
                
                if delta > 0:
                    accept = True
                else:
                    # delta <= 0
                    # p = exp(delta / T)
                    # Use log for stability if T is small?
                    # T starts at 1.0.
                    if delta == 0:
                        accept = random.random() < 0.5
                    else:
                        # delta < 0
                        # Calculate exp(delta/T)
                        # To avoid underflow/overflow issues and speed up, 
                        # we can compare log(p) vs log(random)
                        # log(p) = delta/T
                        # log(random) is hard to generate directly.
                        # Just use np.exp
                        p = np.exp(delta / T)
                        accept = random.random() < p
                
                if accept:
                    # Remove all u in N_S(v)
                    for u in adj[v]:
                        if in_set[u]:
                            in_set[u] = False
                            current_size -= 1
                            for w in adj[u]:
                                conflict_count[w] -= 1
                    # Add v
                    in_set[v] = True
                    current_size += 1
                    for u in adj[v]:
                        conflict_count[u] += 1
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(i for i, val in enumerate(in_set) if val)
                        
        T *= cooling
        steps += 1

    # Construct vectors from best_set
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
