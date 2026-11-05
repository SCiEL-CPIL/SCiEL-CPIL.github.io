import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s buffer

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def find_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        maximal_cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            if P:
                # Choose pivot to minimize recursion
                pivot = max(P | X, key=lambda x: len(adj[x] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return maximal_cliques

    def partition_cliques(n, edges):
        if not edges:
            return []
        
        uncovered = set(edges)
        max_cliques = find_maximal_cliques(n, edges)
        
        # Sort by size descending to prioritize large cliques
        max_cliques.sort(key=len, reverse=True)
        
        partition = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_clique = None
            for clique in max_cliques:
                # Check if all edges of this clique are in uncovered
                available = True
                for u, v in combinations(clique, 2):
                    e = (u, v) if u < v else (v, u)
                    if e not in uncovered:
                        available = False
                        break
                if available:
                    best_clique = clique
                    break
            
            if best_clique is None:
                # Fallback: pick any single edge (should not happen if max cliques cover all)
                e = next(iter(uncovered))
                best_clique = frozenset(e)
            
            partition.append(list(best_clique))
            
            # Remove edges from uncovered
            for u, v in combinations(best_clique, 2):
                e = (u, v) if u < v else (v, u)
                uncovered.discard(e)
                
        return partition

    edges1 = generate_edges(32, 0.6, 601)
    edges2 = generate_edges(64, 0.55, 602)
    
    res1 = partition_cliques(32, edges1)
    res2 = partition_cliques(64, edges2)
    
    return [res1, res2]
