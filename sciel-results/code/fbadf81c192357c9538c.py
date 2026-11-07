import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin for 60s limit
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Precompute edge map
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[idx] = (u, v)
            idx += 1
            
    # Reverse map: (u, v) -> index
    edge_index = {}
    for i, (u, v) in edge_map.items():
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    def is_valid(edge_set):
        """
        Fast validity check using degree grouping.
        """
        if len(edge_set) == 0:
            return True
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        for e in edge_set:
            u, v = edge_map[e]
            
            s_by_deg = {}
            for s in adj[u]:
                if s == v:
                    continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            t_by_deg = {}
            for t in adj[v]:
                if t == u:
                    continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    if not (len(S_set) == 1 and len(T_set) == 1 and S_set == T_set):
                        return False
                        
        return True

    def find_violation(edge_set):
        """Find one violating path or return None."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        for e in edge_set:
            u, v = edge_map[e]
            s_by_deg = {}
            for s in adj[u]:
                if s == v:
                    continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            t_by_deg = {}
            for t in adj[v]:
                if t == u:
                    continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    for s in S_set:
                        for t in T_set:
                            if s != t:
                                return (s, u, v, t)
        return None

    best_edges_set = set()
    best_count = 0

    def try_improve(current_edges, remaining_time_fn):
        """Greedy add + local search on current_edges. Returns improved set."""
        if not is_valid(current_edges):
            return current_edges
        
        improved = current_edges.copy()
        
        # Greedy add: try all missing edges in random order
        missing = [i for i in range(total_edges) if i not in improved]
        random.shuffle(missing)
        
        for idx in missing:
            if remaining_time_fn() < 5.0:
                break
            new_set = improved | {idx}
            if is_valid(new_set):
                improved = new_set
        
        # Local search: try edge swaps (remove one, add one or two)
        for _ in range(3):
            if remaining_time_fn() < 5.0:
                break
            progress = False
            existing = list(improved)
            random.shuffle(existing)
            for rem in existing:
                if remaining_time_fn() < 5.0:
                    break
                trial = improved - {rem}
                # Try adding each missing edge
                missing2 = [i for i in range(total_edges) if i not in trial]
                random.shuffle(missing2)
                for add in missing2:
                    if remaining_time_fn() < 5.0:
                        break
                    candidate = trial | {add}
                    if len(candidate) > len(improved) and is_valid(candidate):
                        improved = candidate
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break
        
        return improved

    # Try multiple bipartite splits
    splits_to_try = [
        (list(range(7)), list(range(7, 15))),  # 7-8
        (list(range(6)), list(range(6, 15))),  # 6-9
        (list(range(5)), list(range(5, 15))),  # 5-10
        (list(range(8)), list(range(8, 15))),  # 8-7
    ]
    
    # Also try random splits
    for restart in range(20):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 42)
        
        if restart < len(splits_to_try):
            A, B = splits_to_try[restart]
        else:
            # Random split
            A = random.sample(range(n), random.randint(3, 12))
            B = [v for v in range(n) if v not in A]
        
        current_edges = set()
        for u in A:
            for v in B:
                current_edges.add(edge_index[(u, v)])
        
        if not is_valid(current_edges):
            # Fallback: star
            current_edges = set()
            for v in range(1, n):
                current_edges.add(edge_index[(0, v)])
        
        improved = try_improve(current_edges, lambda: deadline - time.time())
        
        if len(improved) > best_count:
            best_count = len(improved)
            best_edges_set = improved.copy()
        
        if best_count >= 60:  # Near theoretical max for 15 vertices
            break

    # Final check
    if not is_valid(best_edges_set):
        best_edges_set = set()
        for v in range(1, n):
            best_edges_set.add(edge_index[(0, v)])

    result = []
    for e in best_edges_set:
        u, v = edge_map[e]
        result.append([int(u), int(v)])
        
    return result
