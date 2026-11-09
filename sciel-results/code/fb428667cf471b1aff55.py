import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def has_k4(adj, n):
        """Check if graph has a K4. adj is n x n boolean matrix."""
        # A K4 exists if there are 4 vertices all pairwise connected
        # Check: for each pair (i,j) that are connected, check if they share 2 common neighbors that are also connected
        edges = np.where(np.triu(adj, 1), (True, True), (False, False))
        for a, b in zip(edges[0], edges[1]):
            if not adj[a, b]:
                continue
            common = np.where(adj[a] & adj[b] & (np.arange(n) != a) & (np.arange(n) != b))[0]
            common = [c for c in common if c != a and c != b]
            for i in range(len(common)):
                for j in range(i + 1, len(common)):
                    c, d = common[i], common[j]
                    if adj[c, d]:
                        return True
        return False

    def has_ind6(adj, n):
        """Check if graph has an independent set of size 6."""
        # Independent set = no edges among 6 vertices
        # Complement graph has a K6
        comp = ~adj
        np.fill_diagonal(comp, False)
        # Check for K6 in complement
        for a in range(n):
            neighbors_a = [b for b in range(n) if comp[a, b]]
            for b in neighbors_a:
                if b <= a:
                    continue
                common_ab = [c for c in neighbors_a if c > b and comp[b, c]]
                for c in common_ab:
                    common_bc = [d for d in common_ab if d > c and comp[c, d]]
                    for d in common_bc:
                        common_cd = [e for e in common_bc if e > d and comp[d, e]]
                        for e in common_cd:
                            if comp[e, [f for f in range(n) if f > e and comp[e, f]][0] if common_cd and len(common_cd) > 0 else 0]:
                                pass
                            # Need a 6th vertex connected to all of a,b,c,d,e in complement
                            common_abcde = [f for f in range(n) if f > e and comp[a,f] and comp[b,f] and comp[c,f] and comp[d,f] and comp[e,f]]
                            if common_abcde:
                                return True
        return False

    def has_ind6_fast(adj, n):
        """Faster independent set check using bitsets."""
        # Represent neighborhoods as bitmasks
        # For independent set of size 6, we need 6 vertices with no edges between them
        # i.e., in the complement graph, a K6
        comp = (~adj).astype(np.uint64)
        np.fill_diagonal(comp, 0)
        
        # For each vertex, its "available" neighbors in complement
        # Try to build independent set greedily with backtracking
        # But for exact check, we need to verify existence
        
        # Simpler: iterate over combinations but with early pruning
        verts = list(range(n))
        for combo in combinations(verts, 6):
            # Check if any edge exists among these 6
            found_edge = False
            for i in range(6):
                for j in range(i+1, 6):
                    if adj[combo[i], combo[j]]:
                        found_edge = True
                        break
                if found_edge:
                    break
            if not found_edge:
                return True
        return False

    def has_k4_fast(adj, n):
        """Faster K4 check."""
        verts = list(range(n))
        for a in range(n):
            na = [b for b in range(a+1, n) if adj[a, b]]
            for i in range(len(na)):
                for j in range(i+1, len(na)):
                    b, c = na[i], na[j]
                    if not adj[b, c]:
                        continue
                    # b and c are both neighbors of a and of each other
                    # Need a 4th vertex d that connects to a, b, c
                    common = [d for d in range(c+1, n) if adj[a,d] and adj[b,d] and adj[c,d]]
                    if common:
                        return True
        return False

    def validate(adj, n):
        """Check if graph is K4-free and has no independent 6-set."""
        if has_k4_fast(adj, n):
            return False
        if has_ind6_fast(adj, n):
            return False
        return True

    def circulant_adj(n, S):
        """Build adjacency matrix for circulant graph with connection set S."""
        adj = np.zeros((n, n), dtype=bool)
        for s in S:
            if s <= 0 or s >= n:
                continue
            for i in range(n):
                j1 = (i + s) % n
                j2 = (i - s) % n
                adj[i, j1] = True
                adj[j1, i] = True
                adj[i, j2] = True
                adj[j2, i] = True
        np.fill_diagonal(adj, False)
        return adj

    best_n = 0
    best_edges = None

    # Strategy 1: Search circulant graphs for various n
    # For circulant graphs on n vertices, the graph is determined by S subset of {1, ..., floor(n/2)}
    # For even n, n/2 is a self-loop distance (each vertex connects to exactly one other)
    
    # Try n from 35 down
    for n in range(35, 20, -1):
        if time.time() > deadline - 10:
            break
        
        max_d = n // 2
        all_distances = list(range(1, max_d + 1))
        
        # First try some structured patterns
        structured_patterns = []
        
        # Evens
        structured_patterns.append(set(d for d in all_distances if d % 2 == 0))
        # Odds
        structured_patterns.append(set(d for d in all_distances if d % 2 == 1))
        # Small distances only
        structured_patterns.append(set(all_distances[:max(1, max_d//3)]))
        # Large distances only
        structured_patterns.append(set(all_distances[max(0, max_d//3):]))
        # Every other
        structured_patterns.append(set(d for d in all_distances if d % 3 != 0))
        # Primes
        def is_prime(x):
            if x < 2: return False
            for i in range(2, int(x**0.5)+1):
                if x % i == 0: return False
            return True
        structured_patterns.append(set(d for d in all_distances if is_prime(d)))
        # Composite
        structured_patterns.append(set(d for d in all_distances if d > 1 and not is_prime(d)))
        # Powers of 2
        structured_patterns.append(set(d for d in all_distances if (d & (d-1)) == 0))
        
        found_for_n = False
        
        for S in structured_patterns:
            if time.time() > deadline - 10:
                break
            if not S:
                continue
            adj = circulant_adj(n, S)
            if validate(adj, n):
                best_n = n
                best_edges = set()
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i, j]:
                            best_edges.add((i, j))
                found_for_n = True
                break
        
        if found_for_n:
            break
        
        # If structured patterns didn't work, try random subsets with local search
        # But for circulant, the search space is 2^(n/2) which is manageable for small n
        if n <= 28:
            # Exhaustive or semi-exhaustive search over subsets
            # For n=28, max_d=14, so 2^14 = 16384 subsets
            if time.time() < deadline - 20:
                from itertools import combinations as comb
                found = False
                # Try all subsets of size around n/6 to n/4 (heuristic for good density)
                target_size = max(1, min(max_d, n // 5))
                for size in [target_size - 1, target_size, target_size + 1, target_size - 2, target_size + 2]:
                    if size < 1 or size > max_d:
                        continue
                    for S in comb(all_distances, size):
                        if time.time() > deadline - 10:
                            break
                        S = set(S)
                        adj = circulant_adj(n, S)
                        if validate(adj, n):
                            best_n = n
                            best_edges = set()
                            for i in range(n):
                                for j in range(i+1, n):
                                    if adj[i, j]:
                                        best_edges.add((i, j))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    # Strategy 2: If circulant didn't find good enough, try general local search
    if best_n < 30 and time.time() < deadline - 15:
        # Try to find a valid graph via random init + local search for n=30 to 35
        for n in range(35, max(28, best_n), -1):
            if time.time() > deadline - 10:
                break
            
            # Multiple restarts
            for restart in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Initialize with moderate edge probability
                p = 0.4 + 0.1 * restart
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < p:
                            adj[i, j] = True
                            adj[j, i] = True
                
                # Check initial
                if validate(adj, n):
                    best_n = n
                    best_edges = set()
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i, j]:
                                best_edges.add((i, j))
                    break
                
                # Local search with simulated annealing
                # Cost function: number of K4s + number of independent 6-sets
                # But computing these exactly is expensive
                # Instead, use a penalty-based approach with random moves
                
                current_adj = adj.copy()
                best_adj = adj.copy()
                best_cost = float('inf')
                
                # Simple cost: count violations in random samples
                def sample_cost(adj, n, samples=100):
                    cost = 0
                    # Sample K4s
                    for _ in range(samples):
                        a, b, c, d = random.sample(range(n), 4)
                        if adj[a,b] and adj[a,c] and adj[a,d] and adj[b,c] and adj[b,d] and adj[c,d]:
                            cost += 1
                    # Sample independent 6-sets
                    for _ in range(samples):
                        verts = random.sample(range(n), 6)
                        has_edge = False
                        for i in range(6):
                            for j in range(i+1, 6):
                                if adj[verts[i], verts[j]]:
                                    has_edge = True
                                    break
                            if has_edge:
                                break
                        if not has_edge:
                            cost += 1
                    return cost
                
                temp = 1.0
                cooling = 0.95
                steps = 0
                max_steps = 500
                
                while time.time() < deadline - 10 and steps < max_steps:
                    # Pick a random edge to flip
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    
                    old_cost = sample_cost(current_adj, n)
                    
                    # Flip edge
                    current_adj[u, v] = not current_adj[u, v]
                    current_adj[v, u] = not current_adj[v, u]
                    
                    new_cost = sample_cost(current_adj, n)
                    
                    # Accept if better or with probability exp(-delta/temp)
                    if new_cost < old_cost or random.random() < np.exp(-(new_cost - old_cost) / max(temp, 0.01)):
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_adj = current_adj.copy()
                        temp *= cooling
                    else:
                        # Revert
                        current_adj[u, v] = not current_adj[u, v]
                        current_adj[v, u] = not current_adj[v, u]
                    
                    steps += 1
                    if temp < 0.01:
                        temp = 0.5  # Restart annealing
                        steps = 0
                    
                    # Check if best is valid
                    if validate(best_adj, n):
                        best_n = n
                        best_edges = set()
                        for i in range(n):
                            for j in range(i+1, n):
                                if best_adj[i, j]:
                                    best_edges.add((i, j))
                        break
                
                if best_n >= n:
                    break

    # Fallback: known valid construction for n=17 (Paley graph)
    if best_n == 0:
        n = 17
        squares = set()
        for x in range(1, 17):
            squares.add((x * x) % 17)
        best_edges = set()
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % n in squares:
                    best_edges.add((i, j))
        best_n = n
    
    # Return result
    return {'n': best_n, 'edges': [list(e) for e in sorted(best_edges)]}
