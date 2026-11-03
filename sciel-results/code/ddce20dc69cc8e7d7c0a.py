def _optimized_construct(n):
    perms = list(itertools.permutations(range(1, n)))
    num_perms = len(perms)
    
    costs = [[0] * num_perms for _ in range(num_perms)]
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            max_k = min(len(a), len(b))
            k = 0
            for k in range(max_k, -1, -1):
                if k == 0 or a[len(a)-k:] == b[:k]:
                    break
            costs[i][j] = 2 * (n - 1) - k
            
    # Greedy NN
    path = [0]
    visited = [False] * num_perms
    visited[0] = True
    current = 0
    for _ in range(num_perms - 1):
        next_node = -1
        min_cost = float('inf')
        for j in range(num_perms):
            if not visited[j] and costs[current][j] < min_cost:
                min_cost = costs[current][j]
                next_node = j
        path.append(next_node)
        visited[next_node] = True
        current = next_node
        
    # 2-Opt
    def path_cost(p):
        total = 0
        for i in range(len(p) - 1):
            total += costs[p[i]][p[i+1]]
        return total
        
    improved = True
    start_t = time.time()
    while improved and time.time() < start_t + 100:
        improved = False
        for i in range(num_perms - 1):
            for j in range(i + 1, num_perms):
                # Check swap i+1..j
                # Edges affected: (i, i+1), (j, j+1) [if j+1 exists]
                # New edges: (i, j), (i+1, j+1) [if j+1 exists]
                
                p_i = path[i]
                p_i1 = path[i+1]
                p_j = path[j]
                
                if j + 1 < num_perms:
                    p_j1 = path[j+1]
                    cost_before = costs[p_i][p_i1] + costs[p_j][p_j1]
                    cost_after = costs[p_i][p_j] + costs[p_i1][p_j1]
                else:
                    # j is last
                    cost_before = costs[p_i][p_i1]
                    cost_after = costs[p_i][p_j]
                    
                if cost_after < cost_before:
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    improved = True
                    # Continue searching
                    
    # Construct word
    word = []
    for idx, p_idx in enumerate(path):
        p = perms[p_idx]
        block = list(p) + [n] + list(p)
        if idx == 0:
            word.extend(block)
        else:
            max_k = min(len(word), len(block))
            k = 0
            for k in range(max_k, -1, -1):
                if k == 0 or word[len(word)-k:] == block[:k]:
                    break
            word.extend(block[k:])
            
    return "".join(map(str, word))
