import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Fast clique finder using Tomita-style backtracking with color bounds
    def find_clique(adj, n, k):
        if n < k:
            return None
        
        # Build candidate sets as lists for the Tomita algorithm
        # Use a recursive approach with degree-based pruning
        
        def expand(clique, candidates):
            if len(clique) == k:
                return clique[:]
            
            # Prune: not enough candidates
            if len(candidates) < k - len(clique):
                return None
            
            # Order candidates by degree within candidates (greedy coloring heuristic)
            # For speed, just sort by degree in the current candidate set
            cand_list = list(candidates)
            
            # Compute degrees within candidates
            degs = []
            for v in cand_list:
                deg = 0
                for u in cand_list:
                    if u != v and (adj[v] >> u) & 1:
                        deg += 1
                degs.append((deg, v))
            degs.sort(reverse=True)
            
            # Try vertices in order of decreasing degree
            for deg, v in degs:
                if (adj[v] >> v) & 1:  # self-loop check (shouldn't happen)
                    continue
                new_clique = clique + [v]
                # New candidates: vertices in candidates that are connected to v and come after v in some ordering
                # For correctness, we need to be careful about ordering
                new_cands = [u for u in cand_list if u != v and (adj[v] >> u) & 1]
                # To avoid finding the same clique multiple times, we can restrict
                # But for finding ANY clique, we just need to search
                result = expand(new_clique, new_cands)
                if result:
                    return result
            return None
        
        return expand([], list(range(n)))

    def is_valid(adj, n):
        if find_clique(adj, n, 5):
            return False
        # Complement graph
        full_mask = (1 << n) - 1
        adj_comp = [full_mask ^ (1 << i) ^ adj[i] for i in range(n)]
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
    
    # Repair function: iteratively fix violations
    def repair(adj, n, max_time=20.0):
        start = time.time()
        full_mask = (1 << n) - 1
        
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
            adj_comp = [full_mask ^ (1 << i) ^ adj[i] for i in range(n)]
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

    # Attempt 1: Extend Paley 37 with carefully chosen new vertices
    for attempt in range(8):
        if time.time() > deadline - 30:
            break
            
        # Create initial 42-vertex graph
        adj = adj_37[:]
        for new_v in range(q, target_n):
            # Use a structured pattern: connect to vertices based on a hash-like function
            # to create a more balanced graph
            pattern = 0
            for old_v in range(q):
                # Use a pseudo-random but deterministic pattern seeded by attempt and new_v
                seed_val = (attempt * 1000 + new_v * 100 + old_v * 7) % 13
                if seed_val < 6:  # ~46% connection probability
                    pattern |= (1 << old_v)
            adj.append(pattern)
            for old_v in range(q):
                if pattern & (1 << old_v):
                    adj[old_v] |= (1 << new_v)
        
        # Also add edges between new vertices
        for i in range(q, target_n):
            for j in range(i + 1, target_n):
                seed_val = (attempt * 5000 + i * 100 + j * 3) % 13
                if seed_val < 6:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        success = repair(adj, target_n, max_time=25.0)
        if success:
            best_n = target_n
            best_adj = adj
            break

    # If not found, try random 42-vertex graphs with better density
    if best_n < target_n:
        for attempt in range(15):
            if time.time() > deadline - 30:
                break
                
            # Random graph with density tuned to avoid K5 and independent 5
            # For n=42, optimal density is around 0.5
            adj = [0] * target_n
            density = 0.48 + 0.04 * random.random()  # 0.48 to 0.52
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            success = repair(adj, target_n, max_time=15.0)
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
