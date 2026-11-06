import itertools
import random
import sys

def construct():
    """
    Constructs a C4-free graph on up to 43 vertices with independence number <= 10.
    Starts with the polarity graph of PG(2,5) and greedily adds vertices.
    """
    q = 5
    
    # 1. Generate base graph (Polarity graph of PG(2,5))
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n_base = len(points)
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    edges = []
    # Adjacency list
    adj = [set() for _ in range(n_base)]
    
    for i in range(n_base):
        for j in range(i + 1, n_base):
            u = points[i]
            v = points[j]
            dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                
    current_n = n_base
    current_adj = [set(s) for s in adj] # Copy
    
    # Helper: Check if a set S is C4-compatible in the current graph
    def is_c4_compatible(S):
        S_list = list(S)
        for i in range(len(S_list)):
            u = S_list[i]
            for j in range(i+1, len(S_list)):
                v = S_list[j]
                if current_adj[u].intersection(current_adj[v]):
                    return False
        return True

    # Helper: Check if G[V_set] has an independent set of size at least k
    # Returns True if such a set exists
    def has_independent_set(V_set, k):
        if len(V_set) < k:
            return False
        if k == 0:
            return True
        
        # Simple recursive search
        # Pick a vertex, include or exclude
        # To speed up, pick vertex with max degree in induced subgraph
        V_list = list(V_set)
        
        # Heuristic: Order vertices by degree in V_set
        # deg(v) = |N(v) \cap V_set|
        degrees = {v: len(current_adj[v].intersection(V_set)) for v in V_list}
        # Sort by degree descending? Or ascending?
        # For independent set, high degree vertices are "expensive" to include (remove many neighbors).
        # Usually, branch on high degree vertices first for pruning.
        
        V_sorted = sorted(V_list, key=lambda x: degrees[x], reverse=True)
        
        def dfs(remaining, count):
            if count >= k:
                return True
            if not remaining:
                return False
            if count + len(remaining) < k:
                return False
            
            # Pick the first vertex in sorted order (highest degree)
            v = V_sorted[0]
            if v not in remaining:
                # Find a valid vertex
                for u in V_sorted:
                    if u in remaining:
                        v = u
                        break
                else:
                    return False
            
            # Option 1: Include v
            # Remove v and its neighbors from remaining
            neighbors_in_rem = current_adj[v].intersection(remaining)
            new_remaining = remaining - neighbors_in_rem - {v}
            if dfs(new_remaining, count + 1):
                return True
            
            # Option 2: Exclude v
            if dfs(remaining - {v}, count):
                return True
            
            return False

        return dfs(set(V_list), 0)

    # Helper: Find Maximum Independent Set in Conflict Graph H
    # H is defined on vertices 0..current_n-1
    # u ~_H v iff N(u) \cap N(v) != empty
    # We want a large set S such that no two in S share a neighbor.
    # This is MIS in H.
    def find_max_c4_compatible_set():
        # Build conflict graph H
        # We can represent H by checking intersections
        # For efficiency, we can precompute intersections?
        # n is small (~31-43), so O(n^2 * deg) is fine.
        
        # We will use a simple branch and bound for MIS in H
        # Vertices: 0..current_n-1
        # Edge in H if current_adj[u] & current_adj[v]
        
        best_set = set()
        
        # Order vertices by "conflict degree" (number of conflicts)
        # High conflict degree first?
        
        def dfs(remaining, current_set):
            nonlocal best_set
            if not remaining:
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                return
            
            # Pruning: if len(current_set) + len(remaining) <= len(best_set), return
            if len(current_set) + len(remaining) <= len(best_set):
                return
            
            # Pick a vertex v from remaining
            # Heuristic: pick vertex with max conflict degree in remaining
            # Conflict degree of v in remaining = |{u in remaining : N(v) & N(u) != empty}|
            
            v = None
            max_conf = -1
            for u in remaining:
                conf = 0
                for w in remaining:
                    if u == w: continue
                    if current_adj[u].intersection(current_adj[w]):
                        conf += 1
                if conf > max_conf:
                    max_conf = conf
                    v = u
            
            if v is None:
                return

            # Option 1: Include v in S
            # This means we cannot include any u in remaining that conflicts with v
            # u conflicts with v if N(u) & N(v) != empty
            valid_remaining = set()
            for u in remaining:
                if u == v: continue
                if not current_adj[u].intersection(current_adj[v]):
                    valid_remaining.add(u)
            
            dfs(valid_remaining, current_set | {v})
            
            # Option 2: Exclude v
            dfs(remaining - {v}, current_set)

        dfs(set(range(current_n)), set())
        return best_set

    # Greedy addition loop
    for _ in range(43 - current_n):
        # Find the maximum C4-compatible set S
        S = find_max_c4_compatible_set()
        
        if not S:
            break
            
        # Check independence constraint
        # New vertex v_new connected to S
        # Independent sets containing v_new are {v_new} U I where I is independent in G[V \ S]
        # We need alpha(G[V \ S]) <= 9
        # This is equivalent to: G[V \ S] does NOT have an independent set of size 10
        V_set = set(range(current_n)) - S
        
        if has_independent_set(V_set, 10):
            # Cannot add vertex with this S (which is the best S)
            # Since any smaller S' would have larger V\S', alpha would be larger
            break
        
        # Valid! Add the vertex
        new_v = current_n
        current_adj.append(set(S))
        for u in S:
            current_adj[u].add(new_v)
            edges.append([u, new_v])
        current_n += 1

    return {'n': current_n, 'edges': edges}
