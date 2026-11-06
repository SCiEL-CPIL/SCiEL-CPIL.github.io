import time
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit

    n = 256
    max_sum = 2 * n

    # Identify perfect squares up to max_sum
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Filter valid numbers: 2*a must not be a perfect square
    valid_numbers = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_numbers.append(a)

    num_valid = len(valid_numbers)
    val_to_idx = {v: idx for idx, v in enumerate(valid_numbers)}

    # Partition into odd and even (bipartite graph)
    # Odd numbers: odd + odd = even (can be square), odd + even = odd (cannot be square)
    # Even numbers: even + even = even (can be square)
    # So edges only exist between same-parity pairs? No wait...
    # Actually: odd + even = odd, which is never a perfect square (squares of integers >= 1 are 1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256)
    # Wait, 1 is odd! So odd+even CAN be a square if the sum is 1? No, min sum is 1+2=3.
    # Let me reconsider: squares are 1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256
    # Odd squares: 1, 9, 25, 49, 81, 121, 169, 225
    # Even squares: 4, 16, 36, 64, 100, 144, 196, 256
    # So odd+even = odd CAN be a square (e.g., 2+7=9, 1+8=9)
    # This is NOT bipartite by parity!
    
    # Let me reconsider the bipartite structure.
    # Actually, the graph where edges connect a,b if a+b is a square:
    # This is NOT necessarily bipartite. Let me check for odd cycles.
    # 1+3=4, 3+13=16, 13+12=25, 12+4=16... wait 4+12=16, 12+13=25, 13+3=16, 3+1=4
    # 1-3-13-12-4-1? 1+3=4✓, 3+13=16✓, 13+12=25✓, 12+4=16✓, 4+1=5✗
    # Let me try: 2+7=9, 7+9=16, 9+7=16... 
    # 1+8=9, 8+17=25, 17+19=36, 19+17=36...
    # 2+7=9, 7+9=16, 9+7=16 (self-loop? no, 7!=9)
    # 2+7=9, 7+9=16, 9+27=36, 27+25=52 no...
    # Let me just check if the graph is bipartite by trying 2-coloring.
    
    # Build adjacency list
    neighbors = [[] for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b > a and b in val_to_idx:
                j = val_to_idx[b]
                neighbors[i].append(j)
                neighbors[j].append(i)
    
    # Try to 2-color the graph to check bipartiteness
    color = [-1] * num_valid
    is_bipartite = True
    for start in range(num_valid):
        if color[start] == -1:
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in neighbors[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
    
    if is_bipartite:
        # Use Hopcroft-Karp to find maximum matching, then min vertex cover, then max independent set
        left_nodes = [i for i in range(num_valid) if color[i] == 0]
        right_nodes = [i for i in range(num_valid) if color[i] == 1]
        
        # Build mapping
        left_idx = {node: idx for idx, node in enumerate(left_nodes)}
        right_idx = {node: idx for idx, node in enumerate(right_nodes)}
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        
        # Adjacency from left to right
        adj_lr = [[] for _ in range(n_left)]
        for i in range(n_left):
            u = left_nodes[i]
            for v in neighbors[u]:
                if v in right_idx:
                    adj_lr[i].append(right_idx[v])
        
        # Hopcroft-Karp algorithm
        INF = float('inf')
        pair_u = [-1] * n_left   # match for left node u
        pair_v = [-1] * n_right  # match for right node v
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj_lr[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == INF:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found
        
        def dfs(u):
            for v in adj_lr[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        # Find minimum vertex cover using the matching
        # Unmatched left nodes are in the cover
        # Then follow alternating paths from unmatched left nodes
        visited_left = [False] * n_left
        visited_right = [False] * n_right
        
        # Start from unmatched left nodes
        for u in range(n_left):
            if pair_u[u] == -1:
                visited_left[u] = True
        
        # BFS/DFS from unmatched left nodes following alternating paths
        # Edge from left to right: if matched, go from right to left
        # Edge from left to right: if unmatched, go from left to right
        changed = True
        while changed:
            changed = False
            # From visited left nodes, go to right nodes via unmatched edges
            for u in range(n_left):
                if visited_left[u]:
                    for v in adj_lr[u]:
                        if not visited_right[v] and pair_u[u] != v:
                            visited_right[v] = True
                            changed = True
            # From visited right nodes, go to left nodes via matched edges
            for v in range(n_right):
                if visited_right[v]:
                    if pair_v[v] != -1:
                        u = pair_v[v]
                        if not visited_left[u]:
                            visited_left[u] = True
                            changed = True
        
        # Minimum vertex cover:
        # Left nodes: those NOT visited (i.e., matched left nodes not reachable from unmatched left)
        # Right nodes: those visited
        vertex_cover_left = set()
        vertex_cover_right = set()
        for u in range(n_left):
            if not visited_left[u]:
                vertex_cover_left.add(left_nodes[u])
        for v in range(n_right):
            if visited_right[v]:
                vertex_cover_right.add(right_nodes[v])
        
        vertex_cover = vertex_cover_left | vertex_cover_right
        
        # Maximum independent set = complement of minimum vertex cover
        independent_set = set(range(num_valid)) - vertex_cover
        
        result = [valid_numbers[i] for i in independent_set]
        if not result:
            return [1]
        return result
    else:
        # Graph is not bipartite, fall back to greedy + local search
        # Greedy by degree
        static_degrees = [len(neighbors[i]) for i in range(num_valid)]
        sorted_indices = sorted(range(num_valid), key=lambda x: static_degrees[x])
        
        best_mask = 0
        remaining_mask = (1 << num_valid) - 1
        for v in sorted_indices:
            if (remaining_mask & (1 << v)):
                best_mask |= (1 << v)
                remove = (1 << v)
                for nb in neighbors[v]:
                    remove |= (1 << nb)
                remaining_mask &= ~remove
        
        # Add fitting
        def add_fitting(s_mask):
            for j in range(num_valid):
                if not (s_mask & (1 << j)):
                    fits = True
                    for nb in neighbors[j]:
                        if s_mask & (1 << nb):
                            fits = False
                            break
                    if fits:
                        s_mask |= (1 << j)
            return s_mask
        
        best_mask = add_fitting(best_mask)
        
        # Local search with swaps
        if time.time() - start_time < deadline - 5:
            improved = True
            while improved and (time.time() - start_time) < deadline - 3:
                improved = False
                new_mask = add_fitting(best_mask)
                if new_mask != best_mask:
                    best_mask = new_mask
                    improved = True
                    continue
                
                s_indices = [i for i in range(num_valid) if (best_mask & (1 << i))]
                for i in s_indices:
                    if time.time() - start_time > deadline - 3:
                        break
                    rem_mask = best_mask & ~(1 << i)
                    candidate_mask = add_fitting(rem_mask)
                    if bin(candidate_mask).count('1') > bin(best_mask).count('1'):
                        best_mask = candidate_mask
                        improved = True
                        break
        
        result = [valid_numbers[i] for i in range(num_valid) if (best_mask & (1 << i))]
        if not result:
            return [1]
        return result
# EVOLVE-BLOCK-END
