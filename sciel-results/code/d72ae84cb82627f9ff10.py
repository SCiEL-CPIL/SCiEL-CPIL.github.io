import random
import time

def construct():
    n = 20
    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def is_valid(edge_set):
        # Build adjacency masks
        masks = [0] * n
        for u, v in edge_set:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        
        # Check for K3,3
        # For each pair u, v, let C = N(u) & N(v)
        # If |C| >= 3, check if any w has |N(w) & C| >= 3
        for u in range(n):
            for v in range(u + 1, n):
                common_uv = masks[u] & masks[v]
                if bin(common_uv).count('1') < 3:
                    continue
                # Iterate w
                for w in range(n):
                    if w == u or w == v:
                        continue
                    common_w = masks[w] & common_uv
                    if bin(common_w).count('1') >= 3:
                        return False
        return True

    def find_violation(edge_set):
        # Returns a list of edges forming a K3,3 if found, else None
        masks = [0] * n
        for u, v in edge_set:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
        for u in range(n):
            for v in range(u + 1, n):
                common_uv = masks[u] & masks[v]
                if bin(common_uv).count('1') < 3:
                    continue
                for w in range(n):
                    if w == u or w == v:
                        continue
                    common_w = masks[w] & common_uv
                    if bin(common_w).count('1') >= 3:
                        # Found K3,3 with parts {u,v,w} and common_w
                        # Edges are between {u,v,w} and common_w
                        edges_in_k33 = []
                        common_vertices = [i for i in range(n) if (common_w >> i) & 1]
                        for x in [u, v, w]:
                            for y in common_vertices:
                                if x < y:
                                    edges_in_k33.append((x, y))
                                else:
                                    edges_in_k33.append((y, x))
                        return edges_in_k33
        return None

    def repair(edge_set):
        # Remove edges until valid
        while True:
            violation = find_violation(edge_set)
            if violation is None:
                break
            # Remove a random edge from the violation
            edge_to_remove = random.choice(violation)
            edge_set.discard(edge_to_remove)
        return edge_set

    def greedy_add(edge_set):
        # Try to add all possible edges
        added = True
        while added:
            added = False
            # Generate all possible edges not in edge_set
            possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edge_set]
            random.shuffle(possible_edges)
            for u, v in possible_edges:
                new_set = edge_set | {(u, v)}
                if is_valid(new_set):
                    edge_set.add((u, v))
                    added = True
        return edge_set

    def turan_init(k):
        # Create Turan graph T(n, k)
        # Partition n vertices into k parts as equal as possible
        parts = [[] for _ in range(k)]
        for i in range(n):
            parts[i % k].append(i)
        
        edges = set()
        for i in range(k):
            for j in range(i + 1, k):
                for u in parts[i]:
                    for v in parts[j]:
                        if u < v:
                            edges.add((u, v))
                        else:
                            edges.add((v, u))
        return edges

    # Main loop
    while time.time() - start_time < time_limit:
        # Try different initializations
        # 1. Turan graphs
        for k in [3, 4, 5, 6]:
            if time.time() - start_time >= time_limit:
                break
            init_edges = turan_init(k)
            # Repairing Turan might be effective
            current_edges = repair(set(init_edges))
            current_edges = greedy_add(current_edges)
            
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = [list(e) for e in current_edges]
        
        # 2. Random graphs
        for _ in range(5):
            if time.time() - start_time >= time_limit:
                break
            num_edges = random.randint(80, 120)
            all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_possible)
            init_edges = set(all_possible[:num_edges])
            current_edges = repair(init_edges)
            current_edges = greedy_add(current_edges)
            
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = [list(e) for e in current_edges]

    return best_edges
