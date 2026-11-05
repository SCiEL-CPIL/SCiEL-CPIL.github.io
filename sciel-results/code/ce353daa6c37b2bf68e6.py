import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin before 150s limit
    
    def get_fallback():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append([base+i, base+j])
        return {'n': n, 'edges': edges}
    
    def has_c5(adj, n):
        """Check if graph has a C5. adj is list of sets."""
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Check for path u-a-b-c-v where all distinct
                for a in adj[u]:
                    if a == v:
                        continue
                    for b in adj[a]:
                        if b == u or b == v:
                            continue
                        for c in adj[b]:
                            if c == u or c == v or c == a:
                                continue
                            if c in adj[v]:
                                return True
        return False
    
    def has_independent_set_9(adj, n):
        """Check if graph has an independent set of size 9."""
        # Backtracking with pruning
        def backtrack(candidates, size_needed):
            if size_needed == 0:
                return True
            if len(candidates) < size_needed:
                return False
            if time.time() > DEADLINE:
                return False
            
            # Order by degree in induced subgraph (heuristic)
            # Pick vertex with highest degree to prune more
            best_v = None
            best_deg = -1
            for v in candidates:
                deg_in_sub = sum(1 for u in candidates if u != v and u in adj[v])
                if deg_in_sub > best_deg:
                    best_deg = deg_in_sub
                    best_v = v
            
            v = best_v
            # Branch 1: include v
            new_candidates = set(candidates - {v} - adj[v])
            if backtrack(new_candidates, size_needed - 1):
                return True
            
            # Branch 2: exclude v
            new_candidates = set(candidates - {v})
            if backtrack(new_candidates, size_needed):
                return True
            
            return False
        
        return backtrack(set(range(n)), 9)
    
    def is_valid(adj, n):
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True
    
    # Fallback result
    best_result = get_fallback()
    
    # Try to construct n=33
    # Strategy: Start with 8 K4s, add vertex 32, try various connections
    n = 33
    
    # Precompute the base 8 K4s
    base_adj = [set() for _ in range(n)]
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i+1, 4):
                base_adj[base+i].add(base+j)
                base_adj[base+j].add(base+i)
    
    # Try to find connections for vertex 32
    # Vertex 32 must connect to at least 1 vertex in each K4 to prevent 
    # picking one from each K4 + vertex 32 = independent set of size 9
    # But we must avoid creating C5s.
    
    # A vertex in a K4: if we connect vertex 32 to vertex v in K4_b,
    # we need to check if this creates a C5.
    # C5 involving vertex 32: 32-a-b-c-d-32 where a,b,c,d are in the K4s.
    # Since each K4 is a clique, the only way to have a path through a K4 
    # is to use 2 vertices from the same K4 (since any 2 vertices in K4 are adjacent).
    
    # Key insight: If vertex 32 connects to exactly one vertex in each K4,
    # then any cycle through vertex 32 must enter and exit each K4 through 
    # the same vertex (since the K4 is a clique, but we can't use the same 
    # vertex twice in a simple cycle). Actually, if 32 connects to v_b in K4_b,
    # a cycle 32-v_b-...-32 would need to go from v_b to some other vertex 
    # and eventually back to 32. But 32 only connects to v_b in K4_b, so 
    # to return to 32 from K4_b, we must go through v_b again, which is 
    # not allowed in a simple cycle.
    
    # Wait, that's not right. Let me reconsider.
    # If 32 connects to v_b in K4_b, a cycle could be:
    # 32 - v_b - w_b - x_c - y_c - 32
    # where w_b is another vertex in K4_b, x_c is in K4_c, y_c is in K4_c.
    # This is a 5-cycle if all vertices distinct.
    
    # So we need to be careful. Let's just try random assignments and check.
    
    found = False
    for attempt in range(5000):
        if time.time() > DEADLINE:
            break
        
        # Randomly choose which vertex in each K4 to connect to vertex 32
        adj = [set(s) for s in base_adj]  # Deep copy
        
        valid_assignment = True
        for b in range(8):
            base = b * 4
            # Choose one vertex from K4_b to connect to 32
            chosen = random.randint(0, 3)
            v = base + chosen
            adj[32].add(v)
            adj[v].add(32)
        
        # Check validity
        if not has_c5(adj, n):
            if not has_independent_set_9(adj, n):
                # Valid!
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Try n=34 as well (more difficult)
    n = 34
    base_adj = [set() for _ in range(n)]
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i+1, 4):
                base_adj[base+i].add(base+j)
                base_adj[base+j].add(base+i)
    
    for attempt in range(2000):
        if time.time() > DEADLINE:
            break
        
        adj = [set(s) for s in base_adj]
        
        # Connect vertices 32 and 33 to the K4s
        for new_v in [32, 33]:
            for b in range(8):
                base = b * 4
                chosen = random.randint(0, 3)
                v = base + chosen
                adj[new_v].add(v)
                adj[v].add(new_v)
        
        # Maybe add an edge between 32 and 33
        if random.random() < 0.5:
            adj[32].add(33)
            adj[33].add(32)
        
        if not has_c5(adj, n):
            if not has_independent_set_9(adj, n):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    return best_result
