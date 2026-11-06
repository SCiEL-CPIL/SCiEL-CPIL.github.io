import time
import random
import numpy as np
from itertools import combinations

def construct():
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    total_time_limit = 140.0
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        remaining_time = total_time_limit - elapsed
        
        if remaining_time < 1.0:
            cliques = trivial_partition(n, p, seed)
        else:
            if idx == 0:
                instance_time_limit = min(remaining_time * 0.4, 40.0)
            else:
                instance_time_limit = min(remaining_time * 0.6, 100.0)
            
            cliques = solve_instance(n, p, seed, instance_time_limit)
            
        results.append(cliques)
        
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def generate_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def solve_instance(n, p, seed, time_limit):
    edges = generate_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    cliques = []
    start_time = time.time()
    
    # Iteratively find and remove cliques
    # Strategy: repeatedly find a maximal clique in the residual graph,
    # but use a smarter heuristic that prioritizes edges with high "clique potential"
    
    while True:
        if time.time() - start_time > time_limit:
            # Timeout: add remaining edges as pairs
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        cliques.append([u, v])
            break
        
        # Check if any edges remain
        has_edges = False
        for u in range(n):
            if adj[u]:
                has_edges = True
                break
        if not has_edges:
            break
        
        # Find a good clique to remove
        clique = find_best_clique(adj, n)
        
        if not clique:
            # Fallback: find any edge
            for u in range(n):
                if adj[u]:
                    v = min(adj[u])
                    clique = [u, v]
                    break
            if not clique:
                break
        
        # Remove all edges within the clique
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                adj[u].discard(v)
                adj[v].discard(u)
        
        cliques.append(clique)
    
    return cliques

def find_best_clique(adj, n):
    """
    Find a clique that maximizes the number of edges covered relative to its size.
    Score = (edges_in_clique) / (1 + len(clique)) to prefer dense small cliques.
    Actually, we want to minimize total cliques, so we want to cover as many edges
    as possible per clique. A k-clique covers C(k,2) edges.
    We want to maximize C(k,2) per clique, so larger cliques are better.
    But we also need to be careful not to "waste" vertices that could form
    other cliques.
    
    New strategy: Use a scoring function that considers the "opportunity cost"
    of using each vertex in this clique vs. saving it for later.
    """
    # Compute degrees
    degrees = [len(adj[u]) for u in range(n)]
    
    # Sort vertices by degree (descending)
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_clique = []
    best_score = 0
    
    # Try greedy expansion from multiple starting points
    # Focus on high-degree vertices
    K = min(30, n)
    
    for start_idx in range(K):
        start_v = sorted_verts[start_idx]
        if degrees[start_v] == 0:
            continue
        
        # Greedy expansion with scoring
        C = [start_v]
        P = set(adj[start_v])
        
        while P:
            # Pick u in P that maximizes:
            # 1. Number of new edges added (|P ∩ adj[u]|)
            # 2. But penalize if u has high degree outside P (opportunity cost)
            best_u = None
            best_u_score = -1
            
            for u in P:
                new_edges = len(P & adj[u])
                # Penalty for high external degree
                external_deg = degrees[u] - new_edges
                # Score: prefer vertices that add many edges but don't have
                # too many connections outside
                score = new_edges * 2 - external_deg * 0.1
                if score > best_u_score:
                    best_u_score = score
                    best_u = u
            
            if best_u is None:
                break
            C.append(best_u)
            P = P & adj[best_u]
        
        # Evaluate this clique
        k = len(C)
        if k >= 2:
            edges_covered = k * (k - 1) // 2
            # Score: edges covered per "clique slot"
            # We want to maximize edges per clique
            score = edges_covered
            if score > best_score:
                best_score = score
                best_clique = C[:]
    
    return best_clique
