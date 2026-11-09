import time
import random
from itertools import combinations

def _has_c5(adj, n):
    """Check if graph has a 5-cycle using a more efficient method."""
    # For each pair of non-adjacent vertices, check if they share a common neighbor
    # Actually, let's use: C5 exists iff there's a path of length 3 between adjacent vertices
    # or a path of length 4 between non-adjacent vertices... this is getting complex.
    # Use the original approach but optimized.
    for v0 in range(n):
        nbrs0 = adj[v0]
        if len(nbrs0) < 2:
            continue
        nbrs_list = list(nbrs0)
        for i in range(len(nbrs_list)):
            v1 = nbrs_list[i]
            for j in range(i + 1, len(nbrs_list)):
                v4 = nbrs_list[j]
                # Check if there's a path v1-v2-v3-v4 with v2,v3 distinct from v0,v1,v4
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v4:
                        continue
                    common = adj[v2] & adj[v4]
                    common.discard(v0)
                    common.discard(v1)
                    common.discard(v4)
                    if common:
                        return True
    return False

def _alpha_ge_9(adj, n):
    """Check if independence number >= 9 using greedy + backtracking."""
    # Greedy lower bound
    remaining = set(range(n))
    independent = []
    while remaining:
        min_deg = float('inf')
        min_v = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        independent.append(min_v)
        remaining.discard(min_v)
        remaining -= adj[min_v]
        if len(independent) >= 9:
            return True

    # Branch and bound
    def backtrack(remaining_set, current_size):
        if current_size >= 9:
            return True
        if not remaining_set:
            return False
        if current_size + len(remaining_set) < 9:
            return False
        
        # Pick a vertex with minimum degree in remaining
        v = None
        min_deg = float('inf')
        for x in remaining_set:
            deg = len(adj[x] & remaining_set)
            if deg < min_deg:
                min_deg = deg
                v = x
        
        # Branch 1: include v
        new_remaining = remaining_set - {v} - adj[v]
        if backtrack(new_remaining, current_size + 1):
            return True
        
        # Branch 2: exclude v
        new_remaining2 = remaining_set - {v}
        if backtrack(new_remaining2, current_size):
            return True
        
        return False
    
    return backtrack(set(range(n)), 0)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Known 32-vertex solution: 8 disjoint K4s
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
    
    # Build adjacency of 32-vertex graph
    adj32 = [set() for _ in range(32)]
    for u, v in best_edges:
        adj32[u].add(v)
        adj32[v].add(u)
    
    # Precompute: for each vertex a, the set of vertices d such that there's a path a-b-c-d of length 3
    # path3[a] = set of d where exists b~a, c~b, d~c, all distinct
    path3 = [set() for _ in range(32)]
    for a in range(32):
        for b in adj32[a]:
            for c in adj32[b]:
                if c == a or c == b:
                    continue
                for d in adj32[c]:
                    if d != a and d != b and d != c:
                        path3[a].add(d)
    
    random.seed(42)
    
    # Try to find a 33-vertex C5-free graph with alpha <= 8
    # New vertex 32 connects to subset S of {0..31}
    # C5 constraint: for any a, d in S, d not in path3[a]
    # alpha constraint: alpha(G32[V\S]) <= 7
    
    found_33 = False
    for attempt in range(200000):
        if time.time() > deadline - 10:
            break
        
        # |S| must be >= 25 so that |V\S| <= 7
        s_size = random.randint(25, 32)
        S = set(random.sample(range(32), s_size))
        
        # Check C5 constraint: no pair in S has a path of length 3
        c5_ok = True
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            # Check if any other vertex in S is in path3[a]
            if S & path3[a]:
                c5_ok = False
                break
        
        if not c5_ok:
            continue
        
        # C5 constraint satisfied. Check alpha constraint.
        # V\S has size 32 - s_size <= 7, so alpha(V\S) <= 7 automatically.
        # Construct and verify
        new_edges = list(best_edges)
        for v in S:
            new_edges.append([32, v])
        
        n_test = 33
        adj_test = [set() for _ in range(n_test)]
        for u, v in new_edges:
            adj_test[u].add(v)
            adj_test[v].add(u)
        
        if not _has_c5(adj_test, n_test) and not _alpha_ge_9(adj_test, n_test):
            best_n = 33
            best_edges = new_edges
            found_33 = True
            break
    
    # If we found 33, try to find 34
    if found_33 and time.time() < deadline - 15:
        # Build adjacency of 33-vertex graph
        adj33 = [set() for _ in range(33)]
        for u, v in best_edges:
            adj33[u].add(v)
            adj33[v].add(u)
        
        # Precompute path3 for 33-vertex graph
        path3_33 = [set() for _ in range(33)]
        for a in range(33):
            for b in adj33[a]:
                for c in adj33[b]:
                    if c == a or c == b:
                        continue
                    for d in adj33[c]:
                        if d != a and d != b and d != c:
                            path3_33[a].add(d)
        
        for attempt in range(100000):
            if time.time() > deadline - 5:
                break
            
            s_size = random.randint(26, 33)
            S = set(random.sample(range(33), s_size))
            
            c5_ok = True
            for a in S:
                if S & path3_33[a]:
                    c5_ok = False
                    break
            
            if not c5_ok:
                continue
            
            new_edges = list(best_edges)
            for v in S:
                new_edges.append([33, v])
            
            n_test = 34
            adj_test = [set() for _ in range(n_test)]
            for u, v in new_edges:
                adj_test[u].add(v)
                adj_test[v].add(u)
            
            if not _has_c5(adj_test, n_test) and not _alpha_ge_9(adj_test, n_test):
                best_n = 34
                best_edges = new_edges
                break
    
    return {'n': best_n, 'edges': best_edges}
