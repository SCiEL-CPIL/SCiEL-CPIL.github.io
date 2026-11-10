<<<<<<< SEARCH
import time
import random
import numpy as np

def construct():
    N = 40
    deadline = time.time() + 115  # Safety margin (evaluator kills at 150s)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_edges_list = []
    best_count = 0
    
    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)
        
        # Adjacency as bitmasks and sets
        adj_mask = [0] * N  # bitmask of neighbors
        adj_set = [set() for _ in range(N)]
        edges = []
        
        def add_edge(u, v):
            if (adj_mask[u] >> v) & 1:
                return False
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_set[u].add(v)
            adj_set[v].add(u)
            edges.append((u, v))
            return True
        
        def remove_edge(u, v):
            if not ((adj_mask[u] >> v) & 1):
                return False
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_set[u].discard(v)
            adj_set[v].discard(u)
            try:
                edges.remove((u, v))
            except ValueError:
                edges.remove((v, u))
            return True
        
        def get_blocking_c6(u, v):
            """
            Returns a list of edges forming a C6 with (u,v), or None.
            Uses bitmask BFS to check if a path u-a-b-c-d-v exists.
            """
            # We need a simple path of length 5 from u to v (not using edge u-v directly)
            # Path: u -> n1 -> n2 -> n3 -> n4 -> v
            # So we need n4 to be reachable from u in exactly 4 steps (without using v or revisiting),
            # and n4 adjacent to v.
            
            # BFS layers from u, avoiding v
            # layers[d] = bitmask of nodes reachable in exactly d steps from u,
            #            not using v, and not revisiting nodes in earlier layers (simple path constraint is tricky)
            # For a quick check, we ignore the "no revisit" constraint (over-approximation).
            # If the over-approximation says no, then definitely no C6.
            # If it says yes, we do a targeted DFS to confirm.
            
            def neighbors_mask_excl(node, excl):
                m = adj_mask[node]
                m &= ~(1 << excl)  # exclude 'excl'
                return m
            
            layers = [0] * 5  # layers[0] to layers[4]
            layers[0] = 1 << u
            
            for d in range(1, 5):
                mask = 0
                prev = layers[d-1]
                m = prev
                while m:
                    low = m & (-m)
                    node = low.bit_length() - 1
                    m ^= low
                    mask |= neighbors_mask_excl(node, v)
                # Remove nodes in earlier layers to enforce simple path (approximate)
                # Actually, for C6 detection, we just need *some* path.
                # The bitmask BFS without simple path constraint might find a path that revisits nodes,
                # which wouldn't form a simple C6. But it's a good filter.
                # Let's keep it simple: just OR neighbors.
                layers[d] = mask
            
            # Check if any node in layers[4] is adjacent to v
            v_neighbors = adj_mask[v] & ~(1 << u)  # exclude u (start of path)
            if (layers[4] & v_neighbors) == 0:
                return None  # No C6 possible
            
            # Targeted DFS to find the actual simple path
            # State: (current_node, depth, path_so_far)
            # We want depth 4, and current_node adjacent to v.
            stack = [(u, 0, [u])]
            while stack:
                node, depth, path = stack.pop()
                if depth == 4:
                    if (adj_mask[node] >> v) & 1:
                        # Found path: path + [v]
                        c6_edges = []
                        for i in range(len(path) - 1):
                            c6_edges.append((path[i], path[i+1]))
                        c6_edges.append((node, v))
                        return c6_edges
                    continue
                # Explore neighbors
                nm = adj_mask[node]
                # Exclude v (we only go to v at the end)
                nm &= ~(1 << v)
                # Exclude nodes already in path (simple path)
                for p in path:
                    nm &= ~(1 << p)
                
                m = nm
                while m:
                    low = m & (-m)
                    nb = low.bit_length() - 1
                    m ^= low
                    stack.append((nb, depth + 1, path + [nb]))
            return None
        
        def is_c6_free(u, v):
            return get_blocking_c6(u, v) is None
        
        # --- Greedy phase with heuristic ranking ---
        def compute_priority():
            """
            Rank missing edges by a heuristic score.
            Prefer edges that connect high-degree nodes but have few common neighbors (to avoid C4s, which can lead to C6s).
            """
            missing = [(u, v) for u, v in all_pairs if not ((adj_mask[u] >> v) & 1)]
            if not missing:
                return []
            
            scores = []
            for u, v in missing:
                du = bin(adj_mask[u]).count('1')
                dv = bin(adj_mask[v]).count('1')
                common = bin(adj_mask[u] & adj_mask[v]).count('1')
                # Score: balance degree and common neighbors
                score = (du + dv) * 1.0 - common * 3.0
                score += random.random() * 0.5
                scores.append((score, u, v))
            
            scores.sort(reverse=True)
            return [(u, v) for _, u, v in scores]
        
        # Initial graph: random sparse graph or empty
        # Let's start with a few random edges to break symmetry
        for _ in range(20):
            u, v = random.sample(range(N), 2)
            if is_c6_free(u, v):
                add_edge(u, v)
        
        local_best = len(edges)
        local_best_adj = [set(s) for s in adj_set]
        
        # Local search loop
        while time.time() < deadline:
            # Get ranked missing edges
            ranked_missing = compute_priority()
            if not ranked_missing:
                break
            
            # Shuffle for diversity
            random.shuffle(ranked_missing)
            
            added_in_pass = 0
            for u, v in ranked_missing:
                if time.time() > deadline:
                    break
                if (adj_mask[u] >> v) & 1:
                    continue
                
                if is_c6_free(u, v):
                    add_edge(u, v)
                    added_in_pass += 1
                    if len(edges) > local_best:
                        local_best = len(edges)
                        local_best_adj = [set(s) for s in adj_set]
            
            if added_in_pass == 0:
                # Stuck: perturbation
                if ranked_missing:
                    target_u, target_v = random.choice(ranked_missing)
                    block_c6 = get_blocking_c6(target_u, target_v)
                    if block_c6:
                        # Remove 1-3 random edges from the blocking C6
                        num_removals = random.randint(1, 3)
                        for _ in range(num_removals):
                            if edges:
                                e_to_remove = random.choice(block_c6)
                                remove_edge(e_to_remove[0], e_to_remove[1])
                    elif edges:
                        e = random.choice(edges)
                        remove_edge(e[0], e[1])
        
        # Update global best
        if local_best > best_count:
            best_count = local_best
            best_edges_list = []
            for u in range(N):
                for v in local_best_adj[u]:
                    if u < v:
                        best_edges_list.append([u, v])
        
        return local_best
    
    # Multi-restart
    seeds = [42, 123, 7, 999, 31415, 271, 808, 555, 111, 222]
    for i, s in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        run_once(s)
    
    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
=======
import time
import random
import numpy as np

def construct():
    N = 40
    deadline = time.time() + 115  # Safety margin (evaluator kills at 150s)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_edges_list = []
    best_count = 0
    
    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)
        
        # Adjacency as bitmasks
        adj_mask = [0] * N
        degrees = [0] * N
        edges = []
        
        def add_edge(u, v):
            if (adj_mask[u] >> v) & 1:
                return False
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            edges.append((u, v))
            return True
        
        def remove_edge(u, v):
            if not ((adj_mask[u] >> v) & 1):
                return False
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            degrees[u] -= 1
            degrees[v] -= 1
            edges.remove((u, v))
            return True
        
        def get_blocking_c6(u, v):
            """
            Returns a list of edges forming a C6 with (u,v), or None.
            Fast bitmask BFS filter, then targeted DFS with bitmasks for path.
            """
            # Quick BFS filter: can we reach a neighbor of v in exactly 4 steps from u (avoiding v)?
            layers = [0] * 5
            layers[0] = 1 << u
            
            for d in range(1, 5):
                mask = 0
                prev = layers[d-1]
                m = prev
                while m:
                    low = m & (-m)
                    node = low.bit_length() - 1
                    m ^= low
                    mask |= adj_mask[node] & ~(1 << v)
                layers[d] = mask
            
            v_neighbors = adj_mask[v] & ~(1 << u)
            if (layers[4] & v_neighbors) == 0:
                return None
            
            # Targeted DFS using bitmask for visited path
            # State: (node, depth, visited_mask)
            # We want to find path u -> n1 -> n2 -> n3 -> n4 -> v where n4 adj v
            stack = [(u, 0, 1 << u)]
            while stack:
                node, depth, visited = stack.pop()
                if depth == 4:
                    if (adj_mask[node] >> v) & 1:
                        # Reconstruct path by backtracking (we'll store parent pointers)
                        # For now, just return a dummy - we need the actual edges
                        # Let's use a different approach: store parent
                        pass
                    continue
                nm = adj_mask[node] & ~(1 << v) & ~visited
                m = nm
                while m:
                    low = m & (-m)
                    nb = low.bit_length() - 1
                    m ^= low
                    stack.append((nb, depth + 1, visited | (1 << nb)))
            
            # More careful DFS that tracks the path
            parent = [-1] * N
            depth_arr = [-1] * N
            stack = [u]
            depth_arr[u] = 0
            parent[u] = -1
            
            found = False
            while stack and not found:
                node = stack.pop()
                if depth_arr[node] == 4:
                    if (adj_mask[node] >> v) & 1:
                        # Reconstruct path
                        path = []
                        cur = node
                        while cur != -1:
                            path.append(cur)
                            cur = parent[cur]
                        path.reverse()
                        # path is [u, n1, n2, n3, n4]
                        c6_edges = []
                        for i in range(len(path) - 1):
                            c6_edges.append((path[i], path[i+1]))
                        c6_edges.append((node, v))
                        return c6_edges
                    continue
                if depth_arr[node] >= 4:
                    continue
                nm = adj_mask[node] & ~(1 << v)
                for nb in range(N):
                    if (nm >> nb) & 1:
                        if depth_arr[nb] == -1:
                            depth_arr[nb] = depth_arr[node] + 1
                            parent[nb] = node
                            stack.append(nb)
            return None
        
        def is_c6_free(u, v):
            return get_blocking_c6(u, v) is None
        
        def compute_priority():
            """
            Rank missing edges by heuristic score.
            """
            missing = []
            for u, v in all_pairs:
                if not ((adj_mask[u] >> v) & 1):
                    missing.append((u, v))
            if not missing:
                return []
            
            scores = []
            for u, v in missing:
                du = degrees[u]
                dv = degrees[v]
                common = bin(adj_mask[u] & adj_mask[v]).count('1')
                score = (du + dv) * 1.0 - common * 3.0
                score += random.random() * 0.5
                scores.append((score, u, v))
            
            scores.sort(reverse=True)
            return [(u, v) for _, u, v in scores]
        
        # Initial graph: start with a structured base
        # Use a random regular-ish graph to break symmetry
        for _ in range(15):
            u, v = random.sample(range(N), 2)
            if is_c6_free(u, v):
                add_edge(u, v)
        
        local_best = len(edges)
        local_best_adj = [0] * N
        
        def snapshot():
            return [m for m in adj_mask]
        
        best_snapshot = snapshot()
        
        # Local search loop
        passes = 0
        while time.time() < deadline:
            passes += 1
            ranked_missing = compute_priority()
            if not ranked_missing:
                break
            
            random.shuffle(ranked_missing)
            
            added_in_pass = 0
            for u, v in ranked_missing:
                if time.time() > deadline:
                    break
                if (adj_mask[u] >> v) & 1:
                    continue
                
                if is_c6_free(u, v):
                    add_edge(u, v)
                    added_in_pass += 1
                    if len(edges) > local_best:
                        local_best = len(edges)
                        best_snapshot = snapshot()
            
            if added_in_pass == 0:
                # Perturbation: remove edges to unblock
                if edges:
                    # Try removing from a blocking C6 if we can find one
                    perturbed = False
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        if ranked_missing:
                            tu, tv = random.choice(ranked_missing)
                            if not ((adj_mask[tu] >> tv) & 1):
                                block = get_blocking_c6(tu, tv)
                                if block:
                                    # Remove one edge from the blocking cycle
                                    e = random.choice(block)
                                    remove_edge(e[0], e[1])
                                    perturbed = True
                                    break
                    if not perturbed and edges:
                        e = random.choice(edges)
                        remove_edge(e[0], e[1])
        
        # Restore best snapshot if current is worse
        if len(edges) < local_best:
            for i in range(N):
                adj_mask[i] = best_snapshot[i]
            for i in range(N):
                degrees[i] = bin(adj_mask[i]).count('1')
            edges.clear()
            for u in range(N):
                m = adj_mask[u]
                while m:
                    low = m & (-m)
                    v = low.bit_length() - 1
                    m ^= low
                    if u < v:
                        edges.append((u, v))
        
        # Update global best
        if local_best > best_count:
            best_count = local_best
            best_edges_list = []
            for u in range(N):
                m = adj_mask[u]
                while m:
                    low = m & (-m)
                    v = low.bit_length() - 1
                    m ^= low
                    if u < v:
                        best_edges_list.append([u, v])
        
        return local_best
    
    # Multi-restart with more diverse seeds
    seeds = [42, 123, 7, 999, 31415, 271, 808, 555, 111, 222, 333, 444, 5555, 6666, 7777]
    for i, s in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        run_once(s)
    
    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
>>>>>>> REPLACE
