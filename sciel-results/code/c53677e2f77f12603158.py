import time

def construct():
    start_time = time.time()
    time_limit = 115  # Leave buffer for return
    
    def is_prime_power_q(q):
        """Check if q is a prime power with q ≡ 1 (mod 4)"""
        if q % 4 != 1:
            return False
        # Check if q is a prime power
        for p in range(2, q):
            if p * p > q:
                break
            if q % p == 0:
                # Check if q is a power of p
                temp = q
                while temp % p == 0:
                    temp //= p
                if temp == 1:
                    return True
                return False
        return True  # q is prime
    
    def build_paley_graph(q):
        """Build the Paley graph on q vertices using quadratic residues"""
        if q % 4 != 1:
            return None
        
        # Find quadratic residues mod q
        residues = set()
        for x in range(1, q):
            residues.add((x * x) % q)
        
        # Build edges: (i,j) is an edge if (j-i) mod q is a quadratic residue
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                diff = (j - i) % q
                if diff in residues:
                    edges.append([i, j])
        
        return edges
    
    def has_k4(n, edges):
        """Check if graph contains a K4"""
        if n < 4:
            return False
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    common = adj[i] & adj[j]
                    for k in common:
                        if k > j:
                            common_kj = adj[k] & adj[j]
                            for l in common_kj:
                                if l > k:
                                    return True
        return False
    
    def has_independent_6(n, edges):
        """Check if graph contains an independent set of size 6"""
        if n < 6:
            return False
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build complement adjacency
        comp_adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    comp_adj[i].add(j)
                    comp_adj[j].add(i)
        
        # Backtracking search for clique of size 6 in complement
        def backtrack(candidates, size_needed):
            if size_needed == 0:
                return True
            if len(candidates) < size_needed:
                return False
            
            # Pruning: if we can't find enough common neighbors, return False
            v = candidates[0]
            new_candidates = [u for u in candidates[1:] if u in comp_adj[v]]
            
            # Include v
            if backtrack(new_candidates, size_needed - 1):
                return True
            
            # Exclude v
            if backtrack(candidates[1:], size_needed):
                return True
            
            return False
        
        return backtrack(list(range(n)), 6)
    
    # Try known prime powers q ≡ 1 (mod 4) that might work
    # Known good candidates: 13, 17, 29, 37
    candidates = [13, 17, 29, 37]
    
    best_n = 1
    best_edges = []
    
    for q in candidates:
        if time.time() - start_time > time_limit - 5:
            break
        
        if not is_prime_power_q(q):
            continue
        
        edges = build_paley_graph(q)
        if edges is None:
            continue
        
        # Verify no K4
        if has_k4(q, edges):
            continue
        
        # Verify no independent 6-set
        if has_independent_6(q, edges):
            continue
        
        # Valid!
        if q > best_n:
            best_n = q
            best_edges = edges
    
    # Fallback: if nothing works, use a small known valid graph
    # The 5-cycle C5 has no K4 and no independent 3-set, so certainly no independent 6-set
    if best_n < 5:
        best_n = 5
        best_edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
    
    return {'n': best_n, 'edges': best_edges}
