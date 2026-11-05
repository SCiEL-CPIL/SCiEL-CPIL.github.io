import math

def construct():
    N = 256
    # Nodes are 2..256
    # Weights are 1/i
    # Conflict if lcm(a,b) <= 256
    
    # Precompute conflicts
    # adj[i] is a set of numbers that conflict with i
    # Indices 0..254 correspond to 2..256
    nodes = list(range(2, N + 1))
    adj = [set() for _ in range(len(nodes))]
    
    for i in range(len(nodes)):
        a = nodes[i]
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            if math.lcm(a, b) <= N:
                adj[i].add(j)
                adj[j].add(i)
                
    # Greedy Initialization
    # Sort by weight descending (which is index ascending)
    selected = [False] * len(nodes)
    current_weight = 0.0
    
    for i in range(len(nodes)):
        # Check if i conflicts with any selected node
        can_add = True
        for j in adj[i]:
            if selected[j]:
                can_add = False
                break
        if can_add:
            selected[i] = True
            current_weight += 1.0 / nodes[i]
            
    # Local Search: 1-swap improvement
    # We try to replace a set of conflicting neighbors with a single unselected node
    # If weight(new) > sum(weights(conflicts)), do it.
    
    improved = True
    while improved:
        improved = False
        for i in range(len(nodes)):
            if selected[i]:
                continue
                
            # Find selected neighbors
            conflicts = []
            for j in adj[i]:
                if selected[j]:
                    conflicts.append(j)
                    
            if not conflicts:
                # If no conflicts, we should have added it in greedy. 
                # But maybe greedy order prevented it? No, greedy adds if no conflict.
                # So if it's not selected and has no selected neighbors, it's an error in greedy or state.
                # Actually, if it has no selected neighbors, we can just add it.
                selected[i] = True
                current_weight += 1.0 / nodes[i]
                improved = True
                continue
                
            sum_conflict_weight = 0.0
            for j in conflicts:
                sum_conflict_weight += 1.0 / nodes[j]
                
            weight_i = 1.0 / nodes[i]
            
            if weight_i > sum_conflict_weight:
                # Swap
                for j in conflicts:
                    selected[j] = False
                selected[i] = True
                current_weight = current_weight - sum_conflict_weight + weight_i
                improved = True
                
    # Extract result
    result = []
    for i in range(len(nodes)):
        if selected[i]:
            result.append(nodes[i])
            
    if not result:
        return [2]
        
    return result
