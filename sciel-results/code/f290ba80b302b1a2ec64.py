import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 105  # 45s safety margin

    if k != 4 or g != 10:
        # Fallback for other parameters: return a random regular graph or raise
        # For this task, we assume k=4, g=10
        # If not, we can try a generic search, but let's focus on the specific case
        # to ensure it runs.
        pass

    # Voltage-graph approach over K4,4
    # K4,4: U={0,1,2,3}, V={0,1,2,3}
    # Lift size n. Total vertices 8n.
    # We want total vertices ~ 320, so n=40.
    
    def get_nb_walks(length):
        walks = []
        for start in range(8):
            # DFS with non-backtracking constraint
            stack = [(start, -1, [start])]
            while stack:
                curr, prev, path = stack.pop()
                if len(path) == length + 1:
                    if path[-1] == start:
                        # Verify non-backtracking
                        valid = True
                        for i in range(2, len(path)):
                            if path[i] == path[i-2]:
                                valid = False
                                break
                        if valid:
                            walks.append(path)
                    continue
                for nxt in range(8):
                    if (curr < 4 and nxt >= 4) or (curr >= 4 and nxt < 4):
                        if nxt != prev:
                            stack.append((nxt, curr, path + [nxt]))
        return walks

    walks_4 = get_nb_walks(4)
    walks_6 = get_nb_walks(6)
    walks_8 = get_nb_walks(8)
    
    # Precompute voltage terms for each walk
    def precompute_walk_info(walks):
        info = []
        for w in walks:
            terms = []
            for i in range(len(w) - 1):
                u, v = w[i], w[i+1]
                if u < 4:
                    terms.append((u, v - 4, 1))
                else:
                    terms.append((v - 4, u, -1))
            info.append(terms)
        return info

    walk_info_4 = precompute_walk_info(walks_4)
    walk_info_6 = precompute_walk_info(walks_6)
    walk_info_8 = precompute_walk_info(walks_8)

    all_walk_infos = walk_info_4 + walk_info_6 + walk_info_8

    def count_violations(x, n):
        """Count number of zero-sum walks mod n."""
        count = 0
        for terms in all_walk_infos:
            s = 0
            for u, v, sign in terms:
                s += sign * x[u][v]
            if s % n == 0:
                count += 1
        return count

    def construct_graph(x, n):
        edges = []
        for u in range(4):
            for v in range(4):
                volt = x[u][v]
                for i in range(n):
                    u_id = u * n + i
                    v_id = 4 * n + v * n + ((i + volt) % n)
                    edges.append((u_id, v_id))
        return 8 * n, edges

    def verify_girth(n_graph, edges, required_girth):
        adj = [[] for _ in range(n_graph)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        min_girth = float('inf')
        for start in range(n_graph):
            dist = [-1] * n_graph
            parent = [-1] * n_graph
            q = deque([start])
            dist[start] = 0
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        parent[nei] = curr
                        q.append(nei)
                    elif nei != parent[curr]:
                        cycle_len = dist[curr] + dist[nei] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < required_girth:
                                return False
        return min_girth >= required_girth

    best_result = None
    
    # Try n=40 first (320 vertices)
    # Then n=41, 42...
    for n in range(40, 50):
        if time.time() > deadline:
            break
        
        # Simulated Annealing
        best_x = [[random.randint(0, n-1) for _ in range(4)] for _ in range(4)]
        best_score = count_violations(best_x, n)
        
        temp = 10.0
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Pick a random edge to change
            u = random.randint(0, 3)
            v = random.randint(0, 3)
            old_val = best_x[u][v]
            new_val = random.randint(0, n-1)
            if new_val == old_val:
                continue
            
            best_x[u][v] = new_val
            new_score = count_violations(best_x, n)
            
            delta = new_score - best_score
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                best_score = new_score
            else:
                best_x[u][v] = old_val
            
            temp *= 0.99
            
            if best_score == 0:
                break
        
        if best_score == 0:
            n_graph, edges = construct_graph(best_x, n)
            if verify_girth(n_graph, edges, g):
                if best_result is None or n_graph < best_result[0]:
                    best_result = (n_graph, edges)
                break # Found a valid one, stop searching for larger n

    if best_result is not None:
        return best_result

    # Fallback: Return a valid graph if we didn't find one via voltage search.
    # This should not happen for n=40, but if it does, we return a larger random graph?
    # No, random graphs have short cycles.
    # We must return *something*.
    # If we fail, we return an empty list? No, must be valid.
    # Let's try a larger n with random search again?
    
    # Actually, if the search fails, it's better to return a known valid graph if we have one.
    # Since we don't have a hardcoded one, we try n=45 with more iterations.
    
    for n in range(45, 60):
        if time.time() > deadline:
            break
        # Quick random search
        for _ in range(100):
            x = [[random.randint(0, n-1) for _ in range(4)] for _ in range(4)]
            if count_violations(x, n) == 0:
                n_graph, edges = construct_graph(x, n)
                if verify_girth(n_graph, edges, g):
                    return n_graph, edges
    return 0, [] # This will score 0, but it's a last resort.

