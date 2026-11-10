import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_result = {'n': 5, 'edges': []}
    
    for n in [21, 20]:
        if time.time() > deadline - 10:
            break
            
        # Greedy repair approach
        # Start with a random graph, then iteratively fix violations
        edges = set()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    edges.add((i, j))
        
        def get_violations(edge_set):
            """Return list of (comb, mask, degs) for all regular induced subgraphs of size >= 6."""
            adj = [0] * n
            for u, v in edge_set:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            violations = []
            for k in range(6, n + 1):
                for comb in itertools.combinations(range(n), k):
                    mask = 0
                    for v in comb:
                        mask |= (1 << v)
                    
                    degs = []
                    for v in comb:
                        deg = bin(adj[v] & mask).count('1')
                        degs.append(deg)
                    
                    if len(set(degs)) == 1:
                        violations.append((comb, mask, degs))
                        
            return violations
        
        # Iterative repair
        current_edges = set(edges)
        violations = get_violations(current_edges)
        
        max_iterations = 50000
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            if not violations:
                return {'n': n, 'edges': [list(e) for e in current_edges]}
            
            # Pick a random violation
            comb, mask, degs = random.choice(violations)
            
            # Try to break this violation by flipping an edge
            # An edge (u,v) breaks the violation if:
            # - u and v are both in comb, and flipping changes their degrees differently
            # - or one is in comb and one is outside, changing degree of the one in comb
            
            # Strategy: flip an edge between two vertices in comb
            # This changes degrees of both u and v within comb
            candidates = []
            for i, u in enumerate(comb):
                for j in range(i+1, len(comb)):
                    v = comb[j]
                    candidates.append((u, v))
            
            if not candidates:
                continue
            
            random.shuffle(candidates)
            
            improved = False
            for u, v in candidates[:10]:  # Try up to 10 candidates
                key = (min(u,v), max(u,v))
                new_edges = set(current_edges)
                if key in new_edges:
                    new_edges.remove(key)
                else:
                    new_edges.add(key)
                
                new_violations = get_violations(new_edges)
                
                # Check if we reduced the number of violations
                if len(new_violations) < len(violations):
                    current_edges = new_edges
                    violations = new_violations
                    improved = True
                    break
                elif len(new_violations) == len(violations) and random.random() < 0.3:
                    # Accept with some probability to escape local minima
                    current_edges = new_edges
                    violations = new_violations
                    improved = True
                    break
            
            if not improved:
                # Try flipping an edge where one endpoint is in comb
                for u in comb:
                    for v in range(n):
                        if v in comb:
                            continue
                        key = (min(u,v), max(u,v))
                        new_edges = set(current_edges)
                        if key in new_edges:
                            new_edges.remove(key)
                        else:
                            new_edges.add(key)
                        
                        new_violations = get_violations(new_edges)
                        
                        if len(new_violations) < len(violations):
                            current_edges = new_edges
                            violations = new_violations
                            improved = True
                            break
                        elif len(new_violations) == len(violations) and random.random() < 0.2:
                            current_edges = new_edges
                            violations = new_violations
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Stuck - restart with new random graph
                current_edges = set()
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.5:
                            current_edges.add((i, j))
                violations = get_violations(current_edges)
                
                if time.time() > deadline - 5:
                    break
        
        # If we found a valid graph, it would have returned.
        # Otherwise, continue to next n.
    
    # Fallback: try smaller n
    for n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 5:
            break
            
        edges = set()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    edges.add((i, j))
        
        def get_violations_small(edge_set):
            adj = [0] * n
            for u, v in edge_set:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            violations = []
            for k in range(6, n + 1):
                for comb in itertools.combinations(range(n), k):
                    mask = 0
                    for v in comb:
                        mask |= (1 << v)
                    
                    degs = []
                    for v in comb:
                        deg = bin(adj[v] & mask).count('1')
                        degs.append(deg)
                    
                    if len(set(degs)) == 1:
                        violations.append((comb, mask, degs))
                        
            return violations
        
        current_edges = set(edges)
        violations = get_violations_small(current_edges)
        
        max_iterations = 5000
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            if not violations:
                return {'n': n, 'edges': [list(e) for e in current_edges]}
            
            comb, mask, degs = random.choice(violations)
            
            candidates = []
            for i, u in enumerate(comb):
                for j in range(i+1, len(comb)):
                    v = comb[j]
                    candidates.append((u, v))
            
            if not candidates:
                continue
            
            random.shuffle(candidates)
            
            improved = False
            for u, v in candidates[:10]:
                key = (min(u,v), max(u,v))
                new_edges = set(current_edges)
                if key in new_edges:
                    new_edges.remove(key)
                else:
                    new_edges.add(key)
                
                new_violations = get_violations_small(new_edges)
                
                if len(new_violations) < len(violations):
                    current_edges = new_edges
                    violations = new_violations
                    improved = True
                    break
                elif len(new_violations) == len(violations) and random.random() < 0.3:
                    current_edges = new_edges
                    violations = new_violations
                    improved = True
                    break
            
            if not improved:
                current_edges = set()
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.5:
                            current_edges.add((i, j))
                violations = get_violations_small(current_edges)
        
        # If we didn't find valid graph for this n, continue to smaller
    
    return best_result
