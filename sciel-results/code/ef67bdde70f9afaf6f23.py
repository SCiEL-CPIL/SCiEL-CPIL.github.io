    def search(target_n, time_limit):
        edges = [set() for _ in range(target_n)]
        end_time = time.time() + time_limit
        
        def backtrack(k):
            if time.time() > end_time:
                return False
            if k == target_n:
                return True
            
            attempts = 0
            while attempts < 500:
                attempts += 1
                if time.time() > end_time:
                    return False
                
                size = k // 2 + random.randint(-k//4, k//4)
                if size < 0: size = 0
                if size > k: size = k
                neighbors = set(random.sample(range(k), size))
                
                valid = True
                for u in neighbors:
                    for v in neighbors:
                        if u < v and v in edges[u]:
                            for w in neighbors:
                                if w != u and w != v and w in edges[u] and w in edges[v]:
                                    valid = False
                                    break
                    if not valid:
                        break
                if not valid:
                    continue
                
                non_neighbors = set(range(k)) - neighbors
                if has_independent_set_size_5(list(non_neighbors), edges):
                    continue
                
                for v in neighbors:
                    edges[k].add(v)
                    edges[v].add(k)
                
                if backtrack(k + 1):
                    return True
                
                for v in neighbors:
                    edges[k].remove(v)
                    edges[v].remove(k)
            
            return False
        
        if backtrack(0):
            return edges
        return None

    # Try 36
    res = search(36, 50)
    if res:
        edge_list = []
        for i in range(36):
            for j in res[i]:
                if i < j:
                    edge_list.append([i, j])
        return {'n': 36, 'edges': edge_list}

    # Try 35
    remaining = deadline - time.time()
    if remaining > 5:
        res = search(35, remaining - 5)
        if res:
            edge_list = []
            for i in range(35):
                for j in res[i]:
                    if i < j:
                        edge_list.append([i, j])
            return {'n': 35, 'edges': edge_list}

    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
