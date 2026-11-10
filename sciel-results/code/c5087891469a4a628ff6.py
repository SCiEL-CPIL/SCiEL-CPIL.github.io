import random
import math

def construct():
    N = 256
    # Squares up to 2*N
    max_sum = 2 * N
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # conflicts[i] will store the set of j != i such that i+j is a square
    # We also need to handle the diagonal constraint: 2*i is a square
    conflicts = [set() for _ in range(N + 1)]
    invalid = [False] * (N + 1)
    
    for i in range(1, N + 1):
        # Check diagonal
        if 2 * i in squares:
            invalid[i] = True
            continue
            
        for j in range(i + 1, N + 1):
            if i + j in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # We want to find a large independent set.
    # Strategy: Greedy with degree heuristic, multiple randomized restarts.
    
    best_set = []
    best_size = 0
    
    # Valid nodes are those not invalid
    valid_nodes = [i for i in range(1, N + 1) if not invalid[i]]
    
    # Pre-compute degree for sorting heuristic
    # Using the number of conflicts as a static heuristic
    degrees = [len(conflicts[i]) for i in range(N + 1)]
    
    # Run multiple trials to handle local optima
    trials = 100
    
    for _ in range(trials):
        # Create a list of (degree, node)
        # Add random noise to break ties and explore different orderings
        nodes_to_consider = []
        for i in valid_nodes:
            # Add small random noise to degree to randomize order slightly
            # But keep the general trend of low degree first
            noise = random.uniform(0, 1)
            nodes_to_consider.append((degrees[i] + noise, i))
            
        # Sort by the noisy degree
        nodes_to_consider.sort(key=lambda x: x[0])
        
        current_set = set()
        blocked = set()
        
        for _, node in nodes_to_consider:
            if node in blocked:
                continue
            # Check if node is in current_set? No, we add it if not blocked.
            # But we must ensure it doesn't conflict with existing set members.
            # The 'blocked' set contains all neighbors of nodes in current_set.
            # So if node is not in blocked, it is safe to add.
            
            # Wait, does 'blocked' contain the node itself?
            # We add neighbors to blocked.
            # If we add node to current_set, we add its neighbors to blocked.
            
            current_set.add(node)
            for neighbor in conflicts[node]:
                blocked.add(neighbor)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
    return best_set
