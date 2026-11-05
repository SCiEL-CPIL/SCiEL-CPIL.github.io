import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    random.seed(42)
    
    best_result = {'n': 39, 'edges': []}
    best_score = 0  # min degree achieved
    
    def is_valid(n, edges):
        """Check no C4 and min degree >= n-39."""
        if n < 2:
            return True
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Check min degree
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg < n - 39:
            return False
        # Check no C4: for each pair of adjacent vertices, they must share at most 0 common neighbors
        # Actually C4: a-b-c-d-a means a and c share neighbors b and d
        # So for any two vertices u,v, |N(u) ∩ N(v)| <= 1
        for u in range(n):
            nu = adj[u]
            for v in nu:
                if v > u:
                    common = len(nu & adj[v])
                    if common > 1:
                        return False
        return True
    
    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg)
    
    def has_c4(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            nu = adj[u]
            for v in nu:
                if v > u:
                    if len(nu & adj[v]) > 1:
                        return True
        return False
    
    def add_edge_safe(n, edges, edge_set, adj, u, v):
        """Try to add edge (u,v). Returns True if added without creating C4."""
        if u == v or (u, v) in edge_set or (v, u) in edge_set:
            return False
        # Adding edge (u,v) creates C4 if there exists w such that w~u and w~v and w != u,v
        # i.e., |N(u) ∩ N(v)| >= 1 (since the new edge would make a 4-cycle with any common neighbor)
        # Wait: C4 is u-w-x-v-u. So we need a path of length 3 between u and v.
        # That means there exist w,x with u~w, w~x, x~v, and w != v, x != u, w != x
        # Simpler: adding (u,v) creates a C4 iff there is a vertex w adjacent to both u and v
        # (because then u-w-...-v-u... no wait)
        # Actually: if w is adjacent to both u and v, then u-w-v is a path of length 2.
        # For a C4 we need a path of length 3: u-a-b-v. So we need a~u, a~b, b~v.
        # The C4 check: after adding (u,v), for any pair (x,y) in the graph, |N(x)∩N(y)| <= 1.
        # Adding (u,v) means N(u) gains v and N(v) gains u.
        # New conflicts: for any w in N(u) (old), now w and v share neighbor u. If w~v already, then w and v share u and... 
        # Let me just check: after adding, is there any C4?
        # Efficient check: the new edge (u,v) is in a C4 iff there exists a path u-a-b-v of length 3 in the old graph.
        # That is, there exists a in N(u) and b in N(v) with a~b and a != v and b != u.
        # Even simpler: there exists w such that w~u and w~v (w is common neighbor) -> then u-w-v plus the new edge u-v gives a triangle, not C4.
        # For C4: we need u~a, a~b, b~v in old graph. So a in N(u)\{v}, b in N(v)\{u}, and a~b.
        # Fast check: for each a in N(u), check if any neighbor of a is in N(v)\{u}.
        nu = adj[u]
        nv = adj[v]
        for a in nu:
            if a == v:
                continue
            # check if any neighbor of a is in nv (excluding u)
            for b in adj[a]:
                if b in nv and b != u:
                    return False
        # Safe to add
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
        edges.append([u, v])
        return True
    
    def remove_edge(n, edges, edge_set, adj, u, v):
        adj[u].discard(v)
        adj[v].discard(u)
        edge_set.discard((u, v))
        edge_set.discard((v, u))
        edges.remove([u, v])
    
    # Try different n values from high to low
    for n in range(47, 38, -1):
        if time.time() > deadline - 5:
            break
        target_min_deg = n - 39
        
        # Strategy: Start with a random graph and use local search
        # Or start with a structured graph
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Build initial graph: start empty, greedily add edges
            adj = [set() for _ in range(n)]
            edges = []
            edge_set = set()
            
            # Greedy: add edges in random order, skip if creates C4
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if time.time() > deadline - 10:
                    break
                if add_edge_safe(n, edges, edge_set, adj, u, v):
                    pass
            
            # Now do local search to improve minimum degree
            deg = [len(adj[i]) for i in range(n)]
            
            # Local search: try to increase min degree
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Find vertices with low degree
                for u in range(n):
                    if deg[u] >= target_min_deg:
                        continue
                    # Try adding an edge from u to some v not already connected
                    candidates = [v for v in range(n) if v != u and v not in adj[u]]
                    random.shuffle(candidates)
                    for v in candidates:
                        # Check if adding (u,v) is safe (no C4)
                        safe = True
                        nu = adj[u]
                        nv = adj[v]
                        for a in nu:
                            if a == v:
                                continue
                            for b in adj[a]:
                                if b in nv and b != u:
                                    safe = False
                                    break
                            if not safe:
                                break
                        if safe:
                            adj[u].add(v)
                            adj[v].add(u)
                            edge_set.add((u, v))
                            edge_set.add((v, u))
                            edges.append([u, v])
                            deg[u] += 1
                            deg[v] += 1
                            improved = True
                            break
            
            # Check if we achieved target
            min_deg = min(deg)
            if min_deg >= target_min_deg:
                # Verify no C4
                if not has_c4(n, edges):
                    score = min_deg
                    if score > best_score:
                        best_score = score
                        best_result = {'n': n, 'edges': edges.copy()}
                    break  # Found valid for this n, move to higher n
            
            # Also try: start from complement perspective
            # The complement should have no K_{1,39}
            # Equivalently, in our graph, no vertex is non-adjacent to 39 others
            # i.e., every vertex has degree >= n-39
            
            # Try another approach: start with complete graph, remove edges to eliminate C4s
            if time.time() > deadline - 5:
                break
            
            adj2 = [set(range(n)) - {i} for i in range(n)]
            edges2 = [[i, j] for i in range(n) for j in range(i+1, n)]
            edge_set2 = set()
            for e in edges2:
                edge_set2.add((e[0], e[1]))
                edge_set2.add((e[1], e[0]))
            
            # Remove edges to break C4s
            # Find C4s and remove one edge from each
            for iteration in range(500):
                if time.time() > deadline - 5:
                    break
                # Find a C4
                c4_found = False
                for u in range(n):
                    if len(adj2[u]) < 2:
                        continue
                    nu = list(adj2[u])
                    for i in range(len(nu)):
                        for j in range(i+1, len(nu)):
                            a, b = nu[i], nu[j]
                            # Check if a and b have a common neighbor other than u
                            common = adj2[a] & adj2[b]
                            common.discard(u)
                            if common:
                                c = list(common)[0]
                                # C4: u-a-c-b-u
                                # Remove one edge, say (a,c)
                                adj2[a].discard(c)
                                adj2[c].discard(a)
                                edge_set2.discard((a, c))
                                edge_set2.discard((c, a))
                                edges2.remove([a, c])
                                c4_found = True
                                break
                        if c4_found:
                            break
                    if c4_found:
                        break
                if not c4_found:
                    break
            
            deg2 = [len(adj2[i]) for i in range(n)]
            min_deg2 = min(deg2)
            if min_deg2 >= target_min_deg:
                if not has_c4(n, edges2):
                    score = min_deg2
                    if score > best_score:
                        best_score = score
                        best_result = {'n': n, 'edges': edges2.copy()}
                    break
        
        if best_result['n'] >= n:
            break
    
    return best_result
