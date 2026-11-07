import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    adj = set()
    for u, v in edges:
        adj.add((u, v))
        adj.add((v, u))
    
    def evaluate(subset):
        s = set(subset)
        degrees = []
        for v in s:
            deg = sum(1 for u in s if u != v and (u, v) in adj)
            degrees.append(deg)
        return len(set(degrees))
    
    best_subset = None
    best_score = -1
    
    # Try multiple restarts with different initial sizes
    for restart in range(20):
        # Random initial size between min_vertices and n
        size = random.randint(min_vertices, n)
        current_subset = set(random.sample(range(n), size))
        current_score = evaluate(current_subset)
        
        if current_score > best_score:
            best_score = current_score
            best_subset = list(current_subset)
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        max_iters = 30000
        
        for i in range(max_iters):
            in_list = list(current_subset)
            out_list = [v for v in range(n) if v not in current_subset]
            
            if not out_list:
                # Can only remove, but we need at least min_vertices
                if len(current_subset) <= min_vertices:
                    break
                # Remove a random vertex
                swap_out = random.choice(in_list)
                new_subset = set(current_subset)
                new_subset.remove(swap_out)
                new_score = evaluate(new_subset)
                delta = new_score - current_score
                if delta >= 0 or random.random() < pow(2.718281828459045, delta / T):
                    current_subset = new_subset
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_subset = list(current_subset)
            else:
                # Swap or add
                if len(current_subset) < n and random.random() < 0.3:
                    # Add a vertex
                    swap_in = random.choice(out_list)
                    new_subset = set(current_subset)
                    new_subset.add(swap_in)
                    new_score = evaluate(new_subset)
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < pow(2.718281828459045, delta / T):
                        current_subset = new_subset
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_subset = list(current_subset)
                else:
                    # Swap
                    swap_in = random.choice(out_list)
                    swap_out = random.choice(in_list)
                    new_subset = set(current_subset)
                    new_subset.remove(swap_out)
                    new_subset.add(swap_in)
                    new_score = evaluate(new_subset)
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < pow(2.718281828459045, delta / T):
                        current_subset = new_subset
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_subset = list(current_subset)
            
            T *= cooling
            if T < min_T:
                break
    
    return best_subset
