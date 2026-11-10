"""Direct construction for dense odd-order graphs avoiding equal-degree path endpoints."""
import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80.0

    def is_valid(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for (u, v) in edge_set:
            degs_u = set(deg[s] for s in adj[u] if s != v)
            degs_v = set(deg[t] for t in adj[v] if t != u)
            if degs_u & degs_v:
                return False
        return True

    def havel_hakimi(deg_seq):
        """Try to construct a simple graph from a degree sequence using Havel-Hakimi.
        Returns edge set or None."""
        n = len(deg_seq)
        remaining = list(deg_seq)
        edges = []
        
        for _ in range(n):
            # Find vertex with max remaining degree
            if max(remaining) < 0:
                return None
            max_idx = max(range(n), key=lambda i: remaining[i])
            d = remaining[max_idx]
            if d < 0:
                return None
            if d > n - 1:
                return None
            
            if d == 0:
                continue
            
            # Get d other vertices with highest remaining degrees
            others = [(remaining[i], i) for i in range(n) if i != max_idx]
            others.sort(reverse=True)
            
            if len([x for x in others if x[0] > 0]) < d:
                return None
            
            for j in range(d):
                v = others[j][1]
                if remaining[v] <= 0:
                    return None
                u = max_idx
                if u < v:
                    edges.append((u, v))
                else:
                    edges.append((v, u))
                remaining[u] -= 1
                remaining[v] -= 1
            
            # After connecting, remaining[max_idx] should be 0
            if remaining[max_idx] != 0:
                return None
        
        # Verify all remaining are 0
        if any(r != 0 for r in remaining):
            return None
        
        # Verify no duplicate edges
        edge_set = set()
        for e in edges:
            if e in edge_set:
                return None
            edge_set.add(e)
        
        # Verify degrees
        deg = [0] * n
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
        if sorted(deg) != sorted(deg_seq):
            return None
        
        return edge_set

    best_edges = None
    best_count = 0

    # Try degree sequence [2,3,4,5,6,7,8,9,10,11,12,13,13,13,13] (63 edges)
    base_target = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 13]
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        shuffled = base_target[:]
        random.shuffle(shuffled)
        edges = havel_hakimi(shuffled)
        if edges is not None:
            if is_valid(edges, n):
                if len(edges) > best_count:
                    best_edges = edges
                    best_count = len(edges)
                    if best_count >= 63:
                        break
    
    # Fallback: K_{7,8} (56 edges)
    if best_count < 56:
        A = list(range(7))
        B = list(range(7, 15))
        k78 = set()
        for a in A:
            for b in B:
                k78.add((a, b))
        if is_valid(k78, n):
            best_edges = k78
            best_count = 56

    # Try to add more edges
    if best_edges is not None and time.time() < deadline:
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        missing = [e for e in all_possible if e not in best_edges]
        
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for e in missing[:]:
                if time.time() > deadline:
                    break
                if e in best_edges:
                    continue
                trial = best_edges | {e}
                if is_valid(trial, n):
                    best_edges = trial
                    missing.remove(e)
                    added_any = True
                    best_count = len(best_edges)

    if best_edges is None:
        A = list(range(7))
        B = list(range(7, 15))
        best_edges = set()
        for a in A:
            for b in B:
                best_edges.add((a, b))

    result = [list(e) for e in sorted(best_edges)]
    final = []
    for e in result:
        final.append([int(e[0]), int(e[1])])
    return final
