import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def bron_kerbosch_maximal_cliques(n, adj):
        maximal_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection
            pivot = None
            max_count = -1
            for v in P.union(X):
                count = len(P.intersection(adj[v]))
                if count > max_count:
                    max_count = count
                    pivot = v
            
            candidates = list(P - adj[pivot])
            for v in candidates:
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        P = set(range(n))
        X = set()
        bk(set(), P, X)
        return maximal_cliques
    
    def greedy_hitting_set(cliques):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        selected = set()
        remaining = list(cliques)
        while remaining:
            best_v = None
            max_cover = -1
            # Count how many remaining cliques each vertex covers
            cover_count = defaultdict(int)
            for c in remaining:
                for v in c:
                    cover_count[v] += 1
            for v, cnt in cover_count.items():
                if cnt > max_cover:
                    max_cover = cnt
                    best_v = v
            if best_v is None:
                break
            selected.add(best_v)
            remaining = [c for c in remaining if best_v not in c]
        return list(selected)
    
    def shrink_polish(solution, cliques, time_limit):
        """Try to remove vertices from solution while maintaining validity.
        Multiple passes with different removal orders to escape local minima."""
        solution = set(solution)
        
        def is_valid(s):
            for c in cliques:
                if not c.intersection(s):
                    return False
            return True
        
        if not is_valid(solution):
            return solution
        
        # Try multiple removal strategies
        best = set(solution)
        
        # Strategy 1: greedy removal - remove vertex that is in fewest cliques
        for attempt in range(5):
            current = set(best)
            if not is_valid(current):
                continue
            
            # Get cliques covered by each vertex in current
            while True:
                if time.time() > time_limit:
                    break
                # For each vertex in current, count how many cliques it covers
                vertex_clique_count = {}
                for v in current:
                    count = 0
                    for c in cliques:
                        if v in c:
                            count += 1
                    vertex_clique_count[v] = count
                
                # Find vertex with minimum coverage
                if not current:
                    break
                min_v = min(current, key=lambda v: vertex_clique_count.get(v, 0))
                min_count = vertex_clique_count.get(min_v, 0)
                
                if min_count <= 1:
                    # Safe to remove
                    current.discard(min_v)
                else:
                    break
            
            if len(current) < len(best):
                best = current
        
        # Strategy 2: random removal order
        for attempt in range(10):
            if time.time() > time_limit:
                break
            current = set(best)
            if not is_valid(current):
                continue
            
            vertices = list(current)
            random.Random(attempt * 1000).shuffle(vertices)
            
            for v in vertices:
                if time.time() > time_limit:
                    break
                if v not in current:
                    continue
                test = current - {v}
                if is_valid(test):
                    current = test
            
            if len(current) < len(best):
                best = current
        
        return list(best)
    
    def solve_instance(cliques, time_limit):
        """Solve hitting set for one instance."""
        if not cliques:
            return []
        
        # Get initial solution from greedy
        initial = greedy_hitting_set(cliques)
        best_solution = set(initial)
        best_size = len(best_solution)
        
        # Polish the initial solution
        polished = shrink_polish(initial, cliques, time_limit)
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = set(polished)
        
        # Try local search: swap operations
        # For each pair (v_in, v_out), check if removing v_in and adding v_out improves
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current = list(best_solution)
            n = max(max(c) for c in cliques) + 1 if cliques else 0
            
            # Try removing one vertex and adding another
            for v_remove in list(best_solution):
                if time.time() > time_limit:
                    break
                for v_add in range(n):
                    if v_add in best_solution:
                        continue
                    new_set = (best_solution - {v_remove}) | {v_add}
                    if len(new_set) >= best_size:
                        continue
                    # Check validity
                    valid = True
                    for c in cliques:
                        if not c.intersection(new_set):
                            valid = False
                            break
                    if valid:
                        best_solution = new_set
                        best_size = len(best_solution)
                        improved = True
                        break
                if improved:
                    break
        
        # Final polish
        final = shrink_polish(list(best_solution), cliques, time_limit)
        if len(final) < best_size:
            best_size = len(final)
            best_solution = set(final)
        
        return list(best_solution)
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline - 5:
            # Not enough time left, use quick greedy
            edges = random_graph(n, p, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            cliques = bron_kerbosch_maximal_cliques(n, adj)
            results.append(greedy_hitting_set(cliques))
            continue
        
        edges = random_graph(n, p, seed)
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find maximal cliques
        cliques = bron_kerbosch_maximal_cliques(n, adj)
        
        # Solve hitting set with remaining time
        remaining_time = deadline - time.time()
        solution = solve_instance(cliques, start_time + remaining_time)
        results.append(solution)
        
    return results
