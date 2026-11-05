import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Parameters for the three instances
    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Fallback: return all vertices (valid but large)
            # We should try to return what we have or a quick greedy
            # For now, if we are this late, we just return a simple heuristic
            pass
            
        # 1. Generate Graph
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 2. Enumerate Maximal Cliques (size >= 2)
        # Bron-Kerbosch with pivoting
        cliques = []
        
        def bron_kerbosch(r, p_set, x_set):
            if not p_set and not x_set:
                if len(r) >= 2:
                    cliques.append(frozenset(r))
                return
            
            # Choose pivot u from P union X
            u = None
            max_degree = -1
            for v in p_set:
                deg = len(p_set & adj[v])
                if deg > max_degree:
                    max_degree = deg
                    u = v
            for v in p_set:
                deg = len(p_set & adj[v])
                if deg > max_degree:
                    max_degree = deg
                    u = v
            
            if u is not None:
                candidates = p_set - adj[u]
            else:
                candidates = p_set
                
            for v in list(candidates):
                new_r = r | {v}
                new_p = p_set & adj[v]
                new_x = x_set & adj[v]
                bron_kerbosch(new_r, new_p, new_x)
                p_set.remove(v)
                x_set.add(v)
        
        bron_kerbosch(frozenset(), set(range(n)), set())
        
        if not cliques:
            results.append([])
            continue
            
        # 3. Solve Hitting Set / Set Cover
        # Map vertex to list of clique indices it covers
        vertex_to_cliques = {v: [] for v in range(n)}
        for idx, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(idx)
                
        best_solution = None
        best_size = n + 1
        
        # Run multiple greedy passes with different tie-breakers
        # Tie-breaker 0: Max count, then lowest index
        # Tie-breaker 1: Max count, then highest index
        # Tie-breaker 2: Max count, then random
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            uncovered = set(range(len(cliques)))
            selected = set()
            
            while uncovered:
                best_v = None
                best_count = -1
                best_tie = None
                
                for v in range(n):
                    if v in selected:
                        continue
                    count = 0
                    for idx in vertex_to_cliques[v]:
                        if idx in uncovered:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_v = v
                        if attempt == 0:
                            best_tie = v
                        elif attempt == 1:
                            best_tie = -v
                        else:
                            best_tie = random.randint(0, 1000)
                    elif count == best_count and count > 0:
                        if attempt == 0:
                            tie_val = v
                        elif attempt == 1:
                            tie_val = -v
                        else:
                            tie_val = random.randint(0, 1000)
                            
                        if best_v is None:
                            best_v = v
                            best_tie = tie_val
                        else:
                            if attempt == 0:
                                if tie_val < best_tie:
                                    best_v = v
                                    best_tie = tie_val
                            elif attempt == 1:
                                if tie_val > best_tie:
                                    best_v = v
                                    best_tie = tie_val
                            else:
                                if tie_val < best_tie: # arbitrary random tie break
                                    best_v = v
                                    best_tie = tie_val
                
                if best_v is None:
                    break
                    
                selected.add(best_v)
                # Remove covered cliques
                for idx in vertex_to_cliques[best_v]:
                    if idx in uncovered:
                        uncovered.discard(idx)
                        
            # Local Search Polish: Remove redundant vertices
            changed = True
            while changed:
                changed = False
                for v in list(selected):
                    # Check if v is redundant
                    # A clique is uncovered if it's not covered by any other vertex in selected
                    is_redundant = True
                    for idx in vertex_to_cliques[v]:
                        # Check if any other vertex in selected covers idx
                        covered_by_other = False
                        for u in selected:
                            if u == v:
                                continue
                            if idx in vertex_to_cliques[u]:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            is_redundant = False
                            break
                    if is_redundant:
                        selected.discard(v)
                        changed = True
                        
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = selected
                
        # Convert to sorted list
        results.append(sorted(best_solution) if best_solution else [])

    return results
