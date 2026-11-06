"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        n = len(adj)
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X to minimize branching
            union_px = P | X
            if union_px:
                u = max(union_px, key=lambda x: len(adj[x] & P))
                candidates = P - adj[u]
            else:
                candidates = P
                
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_clique_transversal(adj, cliques):
        """
        Find a small set of vertices that hits every maximal clique.
        Uses a greedy approach: repeatedly pick the vertex that covers the most 
        uncovered cliques, with some randomization for diversification.
        """
        if not cliques:
            return set()
        
        # Track which cliques are still uncovered
        uncovered = list(cliques)
        if not uncovered:
            return set()
        
        selected = set()
        
        # Precompute for each vertex, the list of cliques it belongs to
        vertex_to_cliques = {v: [] for v in range(len(adj))}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        while uncovered:
            # Find vertex covering the most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(len(adj)):
                count = 0
                for ci in vertex_to_cliques[v]:
                    if ci < len(uncovered):
                        count += 1
                # Add a small random tie-breaker
                count += random.random() * 0.001
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None:
                break
                
            selected.add(best_v)
            
            # Remove all cliques containing best_v
            new_uncovered = []
            for c in uncovered:
                if best_v not in c:
                    new_uncovered.append(c)
            uncovered = new_uncovered
            
            if time.time() > DEADLINE - 10:
                break
        
        return selected

    def local_search_improve(adj, cliques, initial_set):
        """
        Try to improve the transversal by removing redundant vertices
        and adding better ones.
        """
        current = set(initial_set)
        
        # Check if current is valid
        def is_valid(s):
            for c in cliques:
                if not (s & c):
                    return False
            return True
        
        if not is_valid(current):
            return current
        
        # Try removing vertices one by one
        for v in list(current):
            trial = current - {v}
            if is_valid(trial):
                current = trial
        
        # Try swapping: remove one, add one
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            for v in list(current):
                trial_base = current - {v}
                # Find best vertex to add
                for u in range(len(adj)):
                    if u in trial_base:
                        continue
                    trial = trial_base | {u}
                    if is_valid(trial) and len(trial) < len(current):
                        current = trial
                        improved = True
                        break
                    elif is_valid(trial) and len(trial) == len(current) and random.random() < 0.1:
                        # Sometimes accept same-size swaps for diversification
                        current = trial
                        improved = True
                        break
        
        return current

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE - 30:
            # Not enough time, use quick greedy
            adj = random_graph(n, p, seed)
            cliques = bron_kerbosch_maximal_cliques(adj)
            sol = greedy_clique_transversal(adj, cliques)
            results.append(sorted(sol))
            continue
        
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        # Multiple random restarts with greedy + local search
        best_sol = None
        best_size = float('inf')
        
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > DEADLINE - 15:
                break
            
            random.seed(seed + r * 1000)
            sol = greedy_clique_transversal(adj, cliques)
            
            # Local search improvement
            sol = local_search_improve(adj, cliques, sol)
            
            if len(sol) < best_size:
                best_size = len(sol)
                best_sol = set(sol)
        
        if best_sol is None:
            best_sol = greedy_clique_transversal(adj, cliques)
        
        results.append(sorted(best_sol))
    
    return results
# EVOLVE-BLOCK-END
