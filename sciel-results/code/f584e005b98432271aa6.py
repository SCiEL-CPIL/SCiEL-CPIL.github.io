import time
import random
import itertools

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave margin for overhead
    
    # We want to find a graph on n vertices with no K4 and no independent 6-set.
    # Known bound: n=35 is possible. n=36 is not.
    # We will try to find a graph for n=35. If it fails, we try smaller n.
    
    def is_valid(n, edges):
        # edges is a set of (u, v) with u < v
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check for K4
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, n):
                    if not (adj[a] & (1 << c)):
                        continue
                    if not (adj[b] & (1 << c)):
                        continue
                    for d in range(c + 1, n):
                        if (adj[a] & (1 << d)) and (adj[b] & (1 << d)) and (adj[c] & (1 << d)):
                            return False
                            
        # Check for Independent 6-set
        # This is equivalent to K6 in the complement
        # Or just check if there is a set of 6 vertices with no edges between them.
        # Using bitmasks for non-neighbors might be faster for large sets, 
        # but n=35 is small.
        # We can use the same logic: check if there is a K6 in the complement.
        # Complement adj: ~adj[i] & mask
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask ^ adj[i] ^ (1 << i) # Remove self
        
        for a in range(n):
            for b in range(a + 1, n):
                if not (comp_adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, n):
                    if not (comp_adj[a] & (1 << c)):
                        continue
                    if not (comp_adj[b] & (1 << c)):
                        continue
                    for d in range(c + 1, n):
                        if not (comp_adj[a] & (1 << d)):
                            continue
                        if not (comp_adj[b] & (1 << d)):
                            continue
                        if not (comp_adj[c] & (1 << d)):
                            continue
                        for e in range(d + 1, n):
                            if not (comp_adj[a] & (1 << e)):
                                continue
                            if not (comp_adj[b] & (1 << e)):
                                continue
                            if not (comp_adj[c] & (1 << e)):
                                continue
                            if not (comp_adj[d] & (1 << e)):
                                continue
                            for f in range(e + 1, n):
                                if (comp_adj[a] & (1 << f)) and (comp_adj[b] & (1 << f)) and \
                                   (comp_adj[c] & (1 << f)) and (comp_adj[d] & (1 << f)) and \
                                   (comp_adj[e] & (1 << f)):
                                    return False
        return True

    def count_violations(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask ^ adj[i] ^ (1 << i)
            
        k4_count = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, n):
                    if not (adj[a] & (1 << c)):
                        continue
                    if not (adj[b] & (1 << c)):
                        continue
                    for d in range(c + 1, n):
                        if (adj[a] & (1 << d)) and (adj[b] & (1 << d)) and (adj[c] & (1 << d)):
                            k4_count += 1
                            
        ind6_count = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not (comp_adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, n):
                    if not (comp_adj[a] & (1 << c)):
                        continue
                    if not (comp_adj[b] & (1 << c)):
                        continue
                    for d in range(c + 1, n):
                        if not (comp_adj[a] & (1 << d)):
                            continue
                        if not (comp_adj[b] & (1 << d)):
                            continue
                        if not (comp_adj[c] & (1 << d)):
                            continue
                        for e in range(d + 1, n):
                            if not (comp_adj[a] & (1 << e)):
                                continue
                            if not (comp_adj[b] & (1 << e)):
                                continue
                            if not (comp_adj[c] & (1 << e)):
                                continue
                            if not (comp_adj[d] & (1 << e)):
                                continue
                            for f in range(e + 1, n):
                                if (comp_adj[a] & (1 << f)) and (comp_adj[b] & (1 << f)) and \
                                   (comp_adj[c] & (1 << f)) and (comp_adj[d] & (1 << f)) and \
                                   (comp_adj[e] & (1 << f)):
                                    ind6_count += 1
        return k4_count + ind6_count

    def find_graph(n, max_time):
        # Initial guess: Random graph with p=0.5
        # Or start with Paley-like structure?
        # Let's try random restarts with WalkSAT.
        
        best_edges = None
        best_score = float('inf')
        
        # Precompute all 4-subsets and 6-subsets for speed?
        # No, just iterate.
        
        for restart in range(100):
            if time.time() - start_time > max_time:
                break
                
            # Initialize
            edges = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        edges.add((i, j))
                        
            current_score = count_violations(n, edges)
            
            # WalkSAT
            for step in range(10000):
                if time.time() - start_time > max_time:
                    return best_edges, best_score
                
                if current_score == 0:
                    return edges, 0
                
                # Pick a random violation
                # This requires listing violations, which is expensive.
                # Alternative: Pick a random edge, flip it, see if score improves.
                # This is "Random Hill Climbing".
                
                # Better: Pick a random edge involved in a violation.
                # To do this efficiently, we might need to track violations.
                # For now, let's try simple random edge flip.
                
                # Generate all possible edges
                all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
                u, v = random.choice(all_edges)
                
                # Flip
                if (u, v) in edges:
                    edges.remove((u, v))
                else:
                    edges.add((u, v))
                    
                new_score = count_violations(n, edges)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(edges)
                else:
                    # Revert
                    if (u, v) in edges:
                        edges.remove((u, v))
                    else:
                        edges.add((u, v))
                        
        return best_edges, best_score

    # Try n=35
    # If we find a valid graph, return it.
    # If not, try n=34, etc.
    
    for n in range(35, 16, -1):
        if time.time() - start_time > timeout - 5:
            break
            
        edges, score = find_graph(n, timeout - (time.time() - start_time))
        
        if score == 0 and edges:
            # Convert to list of lists
            edge_list = [[u, v] for u, v in edges]
            return {'n': n, 'edges': edge_list}
            
    # Fallback: Return a small valid graph (e.g., Paley 17)
    # The seed program does this.
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
