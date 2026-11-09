import time
import random

def _has_c5(adj, n):
    """Check if graph has a 5-cycle. adj is list of sets."""
    for v0 in range(n):
        nbrs0 = adj[v0]
        if len(nbrs0) < 2:
            continue
        nbrs_list = list(nbrs0)
        for i in range(len(nbrs_list)):
            v1 = nbrs_list[i]
            for j in range(i + 1, len(nbrs_list)):
                v4 = nbrs_list[j]
                found = False
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v4:
                        continue
                    common = adj[v2] & adj[v4]
                    if v0 in common:
                        common.discard(v0)
                    if v1 in common:
                        common.discard(v1)
                    if common:
                        found = True
                        break
                if found:
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
    def backtrack(remaining_set, current_independent):
        if len(current_independent) >= 9:
            return True
        if not remaining_set:
            return False
        if len(current_independent) + len(remaining_set) < 9:
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
        if backtrack(new_remaining, current_independent + [v]):
            return True
        
        # Branch 2: exclude v
        new_remaining2 = remaining_set - {v}
        if backtrack(new_remaining2, current_independent):
            return True
        
        return False
    
    return backtrack(set(range(n)), [])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Known 32-vertex solution: 8 disjoint K4s
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
    
    # Try to find a 33-vertex C5-free graph with alpha <= 8
    random.seed(42)
    
    # Build adjacency of 32-vertex graph
    adj32 = [set() for _ in range(32)]
    for u, v in best_edges:
        adj32[u].add(v)
        adj32[v].add(u)
    
    # For 33 vertices, the new vertex 32 connects to some subset S of {0..31}
    # Constraints:
    # 1. No C5: For any a, d in S that are NOT adjacent in G32, there should be no path of length 3 from a to d in G32
    # 2. No independent set of size 9: alpha(G32[V\S]) <= 7 (since new vertex is not connected to V\S)
    
    # Precompute: for each pair (a,d) in G32, whether there's a path of length 3
    # path3[a][d] = True if exists b, c such that a-b-c-d is a path (a!=b!=c!=d, all distinct except a!=d)
    # Actually we need: a-b-c-d where a~b, b~c, c~d, and a,d not necessarily distinct... wait a and d are the endpoints.
    # For C5 through vertex 32: 32-a-b-c-d-32. Need a,d in S, a~32, d~32, a~b, b~c, c~d, all distinct vertices.
    # So a-b-c-d is a path of length 3 in G32 with a and d being endpoints.
    # The constraint is: if a and d are both in S and NOT adjacent in G32, then there must be NO path of length 3 from a to d in G32.
    # Wait, even if a and d ARE adjacent, we could still have a C5: 32-a-d-... no wait.
    # 32-a-b-c-d-32: a~b, b~c, c~d, d~32, 32~a. The cycle is 32,a,b,c,d,32. For this to be a valid 5-cycle, all 5 vertices must be distinct and each consecutive pair adjacent.
    # a and d don't need to be non-adjacent. They just need to not be the same vertex.
    # So the constraint is: for any a, d in S (a != d), there is no path a-b-c-d of length 3 in G32 where b, c are distinct from a, d, 32 and from each other.
    
    # Precompute path3_exists[a][d] for all pairs
    # This is expensive to compute fully, but we can check on the fly.
    
    # Strategy: try many random subsets S of size 25-32
    # |S| must be >= 25 so that |V\S| <= 7, ensuring alpha(V\S) <= 7 trivially.
    
    for attempt in range(50000):
        if time.time() > deadline - 5:
            break
        
        # Choose |S|
        # |S| >= 25 ensures |V\S| <= 7, so alpha(V\S) <= 7 automatically
        s_size = random.randint(25, 32)
        S = set(random.sample(range(32), s_size))
        S_list = list(S)
        
        # Check C5 constraint: for any a, d in S (a < d), no path of length 3 from a to d in G32
        c5_ok = True
        for i in range(len(S_list)):
            a = S_list[i]
            # For each neighbor b of a (b != a, b in range(32))
            for b in adj32[a]:
                if b == a:
                    continue
                # For each neighbor c of b (c != b, c != a)
                for c in adj32[b]:
                    if c == a or c == b:
                        continue
                    # For each neighbor d of c (d != c, d != a, d != b)
                    # d must be in S and d != a
                    common_cd_in_S = adj32[c] & S
                    # Remove a, b, c
                    common_cd_in_S.discard(a)
                    common_cd_in_S.discard(b)
                    common_cd_in_S.discard(c)
                    if common_cd_in_S:
                        # There exists d in S adjacent to c, d != a, b, c
                        # So path a-b-c-d exists, and both a, d in S => C5 through 32
                        c5_ok = False
                        break
                if not c5_ok:
                    break
            if not c5_ok:
                break
        
        if not c5_ok:
            continue
        
        # C5 constraint satisfied. Now check alpha constraint.
        # V\S has size 32 - s_size <= 7, so alpha(V\S) <= 7 automatically.
        # The new vertex 32 is not connected to V\S.
        # Any independent set containing 32 can include at most alpha(V\S) vertices from V\S.
        # So max independent set containing 32 has size 1 + alpha(V\S) <= 1 + 7 = 8.
        # Independent sets not containing 32 are independent sets of G32, which has alpha = 8.
        # So total alpha <= 8. 
        
        # Wait, I need to double-check: the original graph G32 has alpha exactly 8.
        # An independent set in G33 not using vertex 32 is an independent set in G32, so size <= 8.
        # An independent set in G33 using vertex 32 can use at most alpha(V\S) from V\S, and since |V\S| <= 7, this is at most 7, so total at most 8.
        # Therefore alpha(G33) <= 8. 
        
        # Construct the 33-vertex graph
        new_edges = list(best_edges)
        for v in S:
            new_edges.append([32, v])
        
        # Verify
        n_test = 33
        adj_test = [set() for _ in range(n_test)]
        for u, v in new_edges:
            adj_test[u].add(v)
            adj_test[v].add(u)
        
        if not _has_c5(adj_test, n_test) and not _alpha_ge_9(adj_test, n_test):
            best_n = 33
            best_edges = new_edges
            break
    
    # If we found 33, try to find 34
    if best_n == 33 and time.time() < deadline - 10:
        # Build adjacency of 33-vertex graph
        adj33 = [set() for _ in range(33)]
        for u, v in best_edges:
            adj33[u].add(v)
            adj33[v].add(u)
        
        # Try to add vertex 33
        for attempt in range(20000):
            if time.time() > deadline - 5:
                break
            
            s_size = random.randint(26, 33)
            S = set(random.sample(range(33), s_size))
            S_list = list(S)
            
            c5_ok = True
            for i in range(len(S_list)):
                a = S_list[i]
                for b in adj33[a]:
                    if b == a:
                        continue
                    for c in adj33[b]:
                        if c == a or c == b:
                            continue
                        common_cd_in_S = adj33[c] & S
                        common_cd_in_S.discard(a)
                        common_cd_in_S.discard(b)
                        common_cd_in_S.discard(c)
                        if common_cd_in_S:
                            c5_ok = False
                            break
                    if not c5_ok:
                        break
                if not c5_ok:
                    break
            
            if not c5_ok:
                continue
            
            # |V\S| = 33 - s_size <= 7, so alpha constraint is fine
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
