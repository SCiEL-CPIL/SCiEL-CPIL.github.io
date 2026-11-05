import random
from collections import defaultdict, deque

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        
        # Phase 1: Initial Decomposition using Randomized Greedy Walks
        paths = []
        vertices = list(range(max(adj.keys()) + 1)) if adj else []
        
        while remaining_edges:
            # Find a vertex with remaining edges
            start_candidates = [v for v in vertices if adj[v] and any(e in remaining_edges for e in [(v, n) for n in adj[v]] + [(n, v) for n in adj[v]])]
            if not start_candidates:
                # Fallback, should not happen
                u, v = next(iter(remaining_edges))
                paths.append([u, v])
                remaining_edges.discard(tuple(sorted((u, v))))
                continue
                
            start = random.choice(start_candidates)
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Find neighbors connected by remaining edges
                neighbors = [n for n in adj[current] if tuple(sorted((current, n))) in remaining_edges and n not in visited]
                if not neighbors:
                    break
                
                # Heuristic: Prefer neighbors with higher degree in remaining graph (Warnsdorff)
                # But to keep it distinct and simple, let's just pick one with min remaining degree to avoid trapping?
                # Or max? Let's use a simple random choice with slight bias.
                # Actually, for initialization, we just need valid paths.
                # Let's pick the neighbor with the smallest degree in the remaining graph to try to avoid dead ends?
                # Or largest?
                # Let's stick to a simple greedy: pick the first available, or random.
                # To make it robust, let's pick the neighbor that leads to the longest potential path? Too slow.
                # Let's just pick a random one.
                next_node = random.choice(neighbors)
                
                # Remove edge
                e = tuple(sorted((current, next_node)))
                remaining_edges.discard(e)
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
                
            paths.append(path)
            
        # Phase 2: Merging Paths
        # Data structure:
        # paths: list of lists
        # We will repeatedly try to merge paths.
        
        # Helper to check if two paths can be merged
        # P1 ends at v, P2 starts at v.
        # Merge if set(P1) U set(P2) has size len(P1) + len(P2) - 1.
        
        def can_merge(p1, p2, shared_vertex):
            # p1 ends with shared_vertex, p2 starts with shared_vertex
            if p1[-1] != shared_vertex or p2[0] != shared_vertex:
                return False
            # Check for vertex intersection other than shared_vertex
            s1 = set(p1)
            s2 = set(p2)
            if s1 & s2 != {shared_vertex}:
                return False
            return True

        # We will use a loop that runs until no more merges are possible.
        # To avoid infinite loops or high complexity, we limit iterations.
        # Since N is small, this will converge quickly.
        
        max_iterations = 1000
        for _ in range(max_iterations):
            merged_any = False
            # Iterate through all pairs of paths
            # To be efficient, we can group by endpoints.
            # But N is small, so O(K^2) where K is number of paths is fine.
            
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    p1 = paths[i]
                    p2 = paths[j]
                    
                    # Check if p1 ends at p2[0]
                    if p1[-1] == p2[0]:
                        if can_merge(p1, p2, p2[0]):
                            # Merge p1 and p2
                            new_path = p1 + p2[1:]
                            # Remove i and j, add new_path
                            # Be careful with indices
                            paths.pop(j)
                            paths.pop(i)
                            paths.append(new_path)
                            merged_any = True
                            break
                    # Check if p2 ends at p1[0]
                    if p2[-1] == p1[0]:
                        if can_merge(p2, p1, p1[0]):
                            # Merge p2 and p1
                            new_path = p2 + p1[1:]
                            paths.pop(j)
                            paths.pop(i)
                            paths.append(new_path)
                            merged_any = True
                            break
                    # Check if p1 starts at p2[-1] (reverse p1? No, paths are directed? 
                    # The problem doesn't specify direction, but a path is a sequence.
                    # [1,2,3] is the same as [3,2,1] for edge coverage.
                    # We can reverse paths to allow more merges.
                    # Let's allow merging if p1[0] == p2[-1] by reversing p1?
                    # If we reverse p1, it becomes [3,2,1]. Then it ends at 1.
                    # So we should check both orientations.
                    
                    # Actually, let's just try to merge in both directions.
                    # If p1[0] == p2[-1]:
                    #   Reverse p1 -> new_p1
                    #   Check merge(new_p1, p2, p2[-1])
                    # This is equivalent to checking if p1 and p2 can be joined at the other end.
                    
                    # Let's simplify: 
                    # A merge is possible if the union of vertices is simple.
                    # We can join P1 and P2 if:
                    # 1. P1[-1] == P2[0] and sets are disjoint except P1[-1]
                    # 2. P1[0] == P2[-1] and sets are disjoint except P1[0] (Result: P2 + P1[1:]) -> Wait, if P1[0]==P2[-1], we can form P2 + P1[1:]? No, P2 ends at P1[0]. So P2 + P1[1:] is valid if P2[-1]==P1[0].
                    #    Wait, P2 = [u, ..., v], P1 = [v, ..., w].
                    #    Merged: [u, ..., v, ..., w]. This is P2 + P1[1:].
                    #    This is the same as case 1 if we swap P1 and P2?
                    #    Case 1: P1 ends at v, P2 starts at v. Merged: P1 + P2[1:].
                    #    Case 2: P2 ends at v, P1 starts at v. Merged: P2 + P1[1:].
                    #    These are symmetric.
                    
                    # So we just need to check:
                    # if p1[-1] == p2[0]: try merge p1, p2
                    # if p2[-1] == p1[0]: try merge p2, p1 (which is p2 + p1[1:])
                    
                if merged_any:
                    break
            if not merged_any:
                break
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
