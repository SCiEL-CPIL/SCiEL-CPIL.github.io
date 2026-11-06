import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Helper: Find a clique of size k using bitmasks
    def find_clique(adj, n, k):
        if n < k:
            return None
        # Use a simple DFS with bitmasks
        # candidates is a bitmask of vertices
        # current is a list of vertices in the path
        # adj is a list of bitmasks
        
        def dfs(candidates, current):
            if len(current) == k:
                return current[:]
            
            # Pruning: if not enough candidates left, return None
            if candidates.bit_count() < k - len(current):
                return None
            
            # Heuristic: pick vertex with highest degree in candidates
            # To speed up, we can just iterate
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (candidates & adj[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return None
                
            v = best_v
            candidates ^= (1 << v)
            
            new_candidates = candidates & adj[v]
            current.append(v)
            res = dfs(new_candidates, current)
            if res:
                return res
            current.pop()
            
            # Try next vertex? The DFS above only tries the best vertex.
            # We need to try all candidates to find *any* clique.
            # The previous implementation in the prompt was better here.
            # Let's use a more standard backtracking.
            return None
        
        # The above DFS is flawed because it only tries the best vertex.
        # Let's rewrite find_clique to be robust.
        
        def dfs_robust(candidates, current):
            if len(current) == k:
                return current[:]
            
            if candidates.bit_count() < k - len(current):
                return None
            
            # Iterate through all candidates
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                new_candidates = candidates & adj[v]
                current.append(v)
                res = dfs_robust(new_candidates, current)
                if res:
                    return res
                current.pop()
            return None
        
        return dfs_robust((1 << n) - 1, [])

    def is_valid(adj, n):
        if find_clique(adj, n, 5):
            return False
        # Complement graph
        adj_comp = [((1 << n) - 1) ^ (1 << i) ^ adj[i] for i in range(n)]
        if find_clique(adj_comp, n, 5):
            return False
        return True

    # Generate Paley 37 graph
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    adj_37 = [0] * q
    for i, j in seed_edges:
        adj_37[i] |= (1 << j)
        adj_37[j] |= (1 << i)
        
    best_n = q
    best_adj = adj_37[:]

    # Target n = 42
    target_n = 42
    
    # Try to find a valid graph on 42 vertices
    # Strategy: Start with Paley 37 + 5 random vertices, then repair.
    
    def repair(adj, n, max_time=20.0):
        start = time.time()
        # We maintain the adjacency list as bitmasks
        # We try to eliminate all K5 and Independent 5-sets
        
        while time.time() - start < max_time:
            # Check for K5
            clique = find_clique(adj, n, 5)
            if clique:
                # Pick a random edge in the clique to remove
                u, v = random.choice(list(itertools.combinations(clique, 2)))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                continue
            
            # Check for Independent 5-set (K5 in complement)
            adj_comp = [((1 << n) - 1) ^ (1 << i) ^ adj[i] for i in range(n)]
            indep = find_clique(adj_comp, n, 5)
            if indep:
                # Pick a random edge in the independent set to add
                u, v = random.choice(list(itertools.combinations(indep, 2)))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue
            
            # If no violations, we are valid
            return True
        
        return False

    # Attempt 1: Extend Paley 37
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
            
        # Create initial 42-vertex graph
        adj = adj_37[:]
        for new_v in range(q, target_n):
            # Random connections to existing vertices
            pattern = 0
            for old_v in range(q):
                if random.random() < 0.5:
                    pattern |= (1 << old_v)
            adj.append(pattern)
            for old_v in range(q):
                if pattern & (1 << old_v):
                    adj[old_v] |= (1 << new_v)
            
        # Repair
        success = repair(adj, target_n, max_time=30.0)
        if success:
            best_n = target_n
            best_adj = adj
            break

    # If not found, try random 42-vertex graphs
    if best_n < target_n:
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
                
            # Random graph
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            success = repair(adj, target_n, max_time=20.0)
            if success:
                best_n = target_n
                best_adj = adj
                break

    # Convert best_adj to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
                
    return best_n, edges
