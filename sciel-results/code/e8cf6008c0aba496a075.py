import random

def construct():
    N = 256
    
    # 1. Build the conflict graph
    # adj[i] contains indices j such that (i+1) and (j+1) conflict
    adj = [set() for _ in range(N)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                adj[i - 1].add(j - 1)
                adj[j - 1].add(i - 1)
    
    def greedy_independent_set(strategy):
        remaining = set(range(N))
        selected = []
        
        while remaining:
            if strategy == 'min_degree':
                # Pick node with minimum degree in the remaining subgraph
                # This is a standard heuristic for Maximum Independent Set
                best_node = min(remaining, key=lambda x: len(adj[x] & remaining))
            elif strategy == 'max_degree':
                best_node = max(remaining, key=lambda x: len(adj[x] & remaining))
            elif strategy == 'random':
                best_node = random.choice(list(remaining))
            else:
                # Default: arbitrary (e.g., smallest index)
                best_node = min(remaining)
            
            selected.append(best_node + 1)
            remaining.discard(best_node)
            for neighbor in adj[best_node]:
                remaining.discard(neighbor)
        
        return selected

    # Try multiple strategies to find the best set
    best_set = []
    
    # 1. Min degree (usually best for sparse graphs)
    s1 = greedy_independent_set('min_degree')
    if len(s1) > len(best_set):
        best_set = s1
        
    # 2. Max degree
    s2 = greedy_independent_set('max_degree')
    if len(s2) > len(best_set):
        best_set = s2
        
    # 3. Smallest index (deterministic baseline)
    s3 = greedy_independent_set('smallest')
    if len(s3) > len(best_set):
        best_set = s3
        
    # 4. Random restarts to escape local optima of the greedy choice
    for _ in range(20):
        s_rand = greedy_independent_set('random')
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
    if not best_set:
        return [1]
        
    return best_set
