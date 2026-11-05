import time
from math import lcm
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 256

    # Precompute conflicts: conflicts[i] is a set of j such that lcm(i, j) <= 256
    conflicts = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if lcm(i, j) <= N:
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Precompute reciprocal weights
    weights = {i: 1.0 / i for i in range(2, N + 1)}

    def is_compatible(x, current_set):
        for y in conflicts[x]:
            if y in current_set:
                return False
        return True

    # Check if conflict graph is bipartite
    def is_bipartite():
        color = [-1] * (N + 1)
        for start in range(2, N + 1):
            if color[start] != -1:
                continue
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in conflicts[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False, color
        return True, color

    bipartite, color = is_bipartite()

    # If bipartite, use min-cut for exact maximum weight independent set
    if bipartite:
        # Build flow network for min-cut
        # Left side: color 0, Right side: color 1
        # Source -> left nodes (capacity = weight)
        # Right nodes -> Sink (capacity = weight)
        # For each edge (u,v) where u in left, v in right: u -> v (infinite capacity)
        
        INF = float('inf')
        num_nodes = N + 2  # 0..N for numbers, N+1 for source, N+2 for sink
        source = N + 1
        sink = N + 2
        
        # Adjacency list for residual graph
        adj = [[] for _ in range(num_nodes)]
        cap = [[0.0] * num_nodes for _ in range(num_nodes)]
        
        def add_edge(u, v, c):
            adj[u].append(v)
            adj[v].append(u)
            cap[u][v] += c
        
        # Add edges
        for i in range(2, N + 1):
            if color[i] == 0:
                add_edge(source, i, weights[i])
            else:
                add_edge(i, sink, weights[i])
        
        for i in range(2, N + 1):
            for j in conflicts[i]:
                if i < j:  # avoid duplicate edges
                    if color[i] == 0 and color[j] == 1:
                        add_edge(i, j, INF)
                    elif color[i] == 1 and color[j] == 0:
                        add_edge(j, i, INF)
        
        # Edmonds-Karp for max flow
        def bfs():
            parent = [-1] * num_nodes
            parent[source] = -2
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if parent[v] == -1 and cap[u][v] > 0:
                        parent[v] = u
                        if v == sink:
                            return parent
                        queue.append(v)
            return None
        
        total_flow = 0.0
        while time.time() < deadline - 5:
            parent = bfs()
            if parent is None:
                break
            # Find bottleneck
            path_flow = INF
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, cap[u][v])
                v = u
            # Update residual capacities
            v = sink
            while v != source:
                u = parent[v]
                cap[u][v] -= path_flow
                cap[v][u] += path_flow
                v = u
            total_flow += path_flow
        
        # Find min-cut: nodes reachable from source in residual graph
        visited = [False] * num_nodes
        queue = deque([source])
        visited[source] = True
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v] and cap[u][v] > 0:
                    visited[v] = True
                    queue.append(v)
        
        # Selected set: left nodes NOT reachable from source, right nodes reachable from source
        # Wait, let me reconsider. In max weight independent set via min-cut:
        # We want to maximize sum of weights of selected nodes.
        # The min-cut separates source side from sink side.
        # Left nodes on source side: not selected (we pay their weight)
        # Right nodes on sink side: not selected (we pay their weight)
        # So selected = left nodes on sink side + right nodes on source side
        
        selected = set()
        for i in range(2, N + 1):
            if color[i] == 0:
                if not visited[i]:  # on sink side
                    selected.add(i)
            else:
                if visited[i]:  # on source side
                    selected.add(i)
        
        if selected:
            return list(selected)

    # Fallback: local search if not bipartite or min-cut fails
    import random

    # Initial solution: greedy by weight
    current_set = set()
    current_score = 0.0
    
    # Try multiple initial solutions
    best_set = set()
    best_score = 0.0

    # Initial solution 1: all numbers in [129, 256]
    init1 = set(range(129, N + 1))
    s1 = sum(weights[x] for x in init1)
    if s1 > best_score:
        best_score = s1
        best_set = init1

    # Initial solution 2: greedy
    greedy_set = set()
    greedy_score = 0.0
    for x in range(2, N + 1):
        if is_compatible(x, greedy_set):
            greedy_set.add(x)
            greedy_score += weights[x]
    if greedy_score > best_score:
        best_score = greedy_score
        best_set = greedy_set

    current_set = set(best_set)
    current_score = best_score

    # Local search: 1-add, 1-swap, 2-swap
    iteration = 0
    while time.time() < deadline:
        improved = False

        # 1. Try single add
        best_add = None
        best_add_val = 0.0
        for x in range(2, N + 1):
            if x not in current_set:
                if is_compatible(x, current_set):
                    val = weights[x]
                    if val > best_add_val:
                        best_add_val = val
                        best_add = x

        if best_add is not None:
            current_set.add(best_add)
            current_score += best_add_val
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            improved = True
            continue

        # 2. Try 1-swap
        best_swap_gain = 0.0
        best_swap_x = None
        best_swap_y = None

        for y in list(current_set):
            for x in range(2, N + 1):
                if x in current_set:
                    continue
                valid = True
                for z in conflicts[x]:
                    if z in current_set and z != y:
                        valid = False
                        break
                if valid:
                    gain = weights[x] - weights[y]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_x = x
                        best_swap_y = y

        if best_swap_x is not None:
            current_set.remove(best_swap_y)
            current_set.add(best_swap_x)
            current_score += best_swap_gain
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            improved = True
            continue

        # 3. Try 2-swap (limited)
        best_2swap_gain = 0.0
        best_2swap = None

        current_list = list(current_set)
        n = len(current_list)

        if n >= 2:
            # Sample pairs to remove
            num_samples = min(100, n * (n - 1) // 2)
            random.seed(iteration)
            
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            sampled_pairs = random.sample(all_pairs, min(num_samples, len(all_pairs)))
            
            for i, j in sampled_pairs:
                rem_i = current_list[i]
                rem_j = current_list[j]
                remaining = set(current_list) - {rem_i, rem_j}

                candidates = []
                for x in range(2, N + 1):
                    if x in current_set:
                        continue
                    if is_compatible(x, remaining):
                        candidates.append(x)

                if len(candidates) < 2:
                    continue

                # Try a few candidate pairs
                num_cand_samples = min(30, len(candidates) * (len(candidates) - 1) // 2)
                all_cand_pairs = [(a, b) for a in range(len(candidates)) for b in range(a + 1, len(candidates))]
                sampled_cand_pairs = random.sample(all_cand_pairs, min(num_cand_samples, len(all_cand_pairs)))
                
                for a_idx, b_idx in sampled_cand_pairs:
                    a = candidates[a_idx]
                    b = candidates[b_idx]
                    if is_compatible(a, remaining | {b}) and is_compatible(b, remaining | {a}):
                        gain = weights[a] + weights[b] - weights[rem_i] - weights[rem_j]
                        if gain > best_2swap_gain:
                            best_2swap_gain = gain
                            best_2swap = (rem_i, rem_j, a, b)

        if best_2swap is not None:
            rem_i, rem_j, a, b = best_2swap
            current_set.remove(rem_i)
            current_set.remove(rem_j)
            current_set.add(a)
            current_set.add(b)
            current_score += best_2swap_gain
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            improved = True
            continue

        # 4. Random perturbation if stuck
        if len(current_set) >= 2:
            r1, r2 = random.sample(list(current_set), 2)
            rem = set(current_set) - {r1, r2}
            cands = [x for x in range(2, N + 1) if x not in current_set and is_compatible(x, rem)]
            if len(cands) >= 2:
                a, b = random.sample(cands, 2)
                if is_compatible(a, rem | {b}) and is_compatible(b, rem | {a}):
                    current_set.remove(r1)
                    current_set.remove(r2)
                    current_set.add(a)
                    current_set.add(b)
                    current_score = sum(weights[v] for v in current_set)
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                    improved = True

        if not improved:
            iteration += 1

    return list(best_set)
