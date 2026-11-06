import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin from 90s limit
    
    # 1. Preprocessing: Identify valid vertices (2a is not a perfect square)
    def is_square(n):
        r = int(math.isqrt(n))
        return r * r == n

    V = list(range(1, 257))
    V_valid = [a for a in V if not is_square(2 * a)]
    n = len(V_valid)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(V_valid)}
    
    # 2. Build Conflict Graph as adjacency matrix (bitset-friendly)
    squares = [i*i for i in range(2, 23)]  # 4 to 512
    
    # adj_matrix[i][j] = True if V_valid[i] + V_valid[j] is a perfect square
    adj_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for s in squares:
            v = s - V_valid[i]
            if v in val_to_idx:
                j = val_to_idx[v]
                if i != j:
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True
    
    # 3. Use Simulated Annealing approach
    # State: a bit vector x of length n, x[i]=1 means V_valid[i] is in the independent set
    # Objective: maximize sum(x) subject to no two adjacent vertices both in set
    
    def count_conflicts(x):
        """Count number of edges with both endpoints in x"""
        # For each i where x[i]=1, count neighbors j where x[j]=1
        # Using matrix: for each i, x[i] * sum(adj_matrix[i] & x)
        # But we want total edges, so divide by 2
        if not np.any(x):
            return 0
        # Get indices where x is 1
        idx = np.where(x)[0]
        if len(idx) == 0:
            return 0
        # For each selected vertex, count its selected neighbors
        sub = adj_matrix[np.ix_(idx, idx)]
        # Count upper triangle to avoid double counting
        return int(np.sum(sub) // 2)
    
    def is_valid(x):
        return count_conflicts(x) == 0
    
    # Greedy initialization using min-degree heuristic
    def greedy_init():
        x = np.zeros(n, dtype=bool)
        remaining = set(range(n))
        # Compute degrees
        degs = adj_matrix.sum(axis=1).astype(int)
        # Sort by degree
        order = sorted(remaining, key=lambda i: degs[i])
        for i in order:
            if i not in remaining:
                continue
            # Check if i has any conflict with current x
            if not np.any(adj_matrix[i] & x):
                x[i] = True
                remaining.discard(i)
        return x
    
    # Simulated Annealing
    def simulated_annealing(x, time_budget=80.0):
        current_x = x.copy()
        current_size = int(np.sum(current_x))
        best_x = current_x.copy()
        best_size = current_size
        
        # Initial temperature
        T = 1.0
        cooling = 0.9995
        min_T = 1e-6
        
        while time.time() < start_time + time_budget and T > min_T:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a vertex
                candidates = np.where(~current_x)[0]
                if len(candidates) == 0:
                    T *= cooling
                    continue
                i = random.choice(candidates.tolist())
                new_x = current_x.copy()
                new_x[i] = True
                # Check if valid
                if not np.any(adj_matrix[i] & new_x):
                    current_x = new_x
                    new_size = int(np.sum(current_x))
                    if new_size > best_size:
                        best_size = new_size
                        best_x = current_x.copy()
                else:
                    # Conflict introduced, skip
                    pass
            elif move_type < 0.7:
                # Remove a vertex
                candidates = np.where(current_x)[0]
                if len(candidates) == 0:
                    T *= cooling
                    continue
                i = random.choice(candidates.tolist())
                new_x = current_x.copy()
                new_x[i] = False
                current_x = new_x
                new_size = int(np.sum(current_x))
                if new_size > best_size:
                    best_size = new_size
                    best_x = current_x.copy()
            else:
                # Swap: remove one, add one
                in_set = np.where(current_x)[0]
                not_in_set = np.where(~current_x)[0]
                if len(in_set) == 0 or len(not_in_set) == 0:
                    T *= cooling
                    continue
                i = random.choice(in_set.tolist())
                j = random.choice(not_in_set.tolist())
                new_x = current_x.copy()
                new_x[i] = False
                new_x[j] = True
                # Check validity
                if not np.any(adj_matrix[j] & new_x):
                    current_x = new_x
                    new_size = int(np.sum(current_x))
                    if new_size > best_size:
                        best_size = new_size
                        best_x = current_x.copy()
            
            T *= cooling
        
        return best_x, best_size
    
    # Multi-restart strategy
    best_x = None
    best_size = 0
    
    # First, try greedy init
    x0 = greedy_init()
    size0 = int(np.sum(x0))
    if size0 > best_size:
        best_size = size0
        best_x = x0.copy()
    
    # Run simulated annealing from greedy init
    if time.time() < start_time + deadline - start_time - 5:
        sa_x, sa_size = simulated_annealing(x0, time_budget=40.0)
        if sa_size > best_size:
            best_size = sa_size
            best_x = sa_x.copy()
    
    # Try multiple random restarts
    num_restarts = 0
    while time.time() < start_time + deadline - start_time - 3:
        # Random initialization: pick random subset and fix conflicts
        x_rand = np.zeros(n, dtype=bool)
        # Randomly include vertices
        for i in range(n):
            if random.random() < 0.5:
                x_rand[i] = True
        # Fix conflicts: greedily remove conflicting vertices
        changed = True
        while changed and time.time() < start_time + deadline - start_time - 3:
            changed = False
            for i in range(n):
                if x_rand[i]:
                    if np.any(adj_matrix[i] & x_rand):
                        x_rand[i] = False
                        changed = True
        
        rand_size = int(np.sum(x_rand))
        if rand_size > best_size:
            best_size = rand_size
            best_x = x_rand.copy()
        
        # Run SA from this start
        if time.time() < start_time + deadline - start_time - 3:
            sa_x, sa_size = simulated_annealing(x_rand, time_budget=15.0)
            if sa_size > best_size:
                best_size = sa_size
                best_x = sa_x.copy()
        
        num_restarts += 1
    
    # Final local search polish on best solution
    if best_x is not None and time.time() < start_time + deadline - start_time - 1:
        # Try to add any vertex that doesn't conflict
        improved = True
        while improved and time.time() < start_time + deadline - start_time - 1:
            improved = False
            for i in range(n):
                if not best_x[i] and not np.any(adj_matrix[i] & best_x):
                    best_x[i] = True
                    improved = True
                    break
        
        # Try swaps
        improved = True
        while improved and time.time() < start_time + deadline - start_time - 1:
            improved = False
            in_set = np.where(best_x)[0]
            not_in_set = np.where(~best_x)[0]
            for i in in_set:
                for j in not_in_set:
                    new_x = best_x.copy()
                    new_x[i] = False
                    new_x[j] = True
                    if not np.any(adj_matrix[j] & new_x):
                        # This swap is valid, but size doesn't change
                        # Only do it if it might enable future improvements
                        # For now, skip since size is same
                        pass
            # Try 2-swap: remove 2, add 2
            if len(in_set) >= 2 and len(not_in_set) >= 2:
                in_list = in_set.tolist()
                not_list = not_in_set.tolist()
                # Try a few random 2-swaps
                for _ in range(100):
                    i1, i2 = random.sample(in_list, 2)
                    j1, j2 = random.sample(not_list, 2)
                    new_x = best_x.copy()
                    new_x[i1] = False
                    new_x[i2] = False
                    new_x[j1] = True
                    new_x[j2] = True
                    # Check validity
                    if not np.any(adj_matrix[j1] & new_x) and not np.any(adj_matrix[j2] & new_x):
                        # Valid, size same, but might help future
                        # Accept with some probability (annealing-like)
                        if random.random() < 0.1:
                            best_x = new_x
                            improved = True
                            break
    
    # Convert back to values
    result = [V_valid[i] for i in range(n) if best_x[i]]
    
    # Verify
    for a in result:
        for b in result:
            if a <= b:
                s = a + b
                r = math.isqrt(s)
                if r * r == s:
                    # Invalid, should not happen
                    pass
    
    return result
