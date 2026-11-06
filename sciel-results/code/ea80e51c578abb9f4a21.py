import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 280  # 280s safety margin
    
    # Fallback: Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    fallback_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Try to find a larger graph
    # We try n from 42 down to 38
    for n in range(42, 37, -1):
        if time.time() > deadline - 20:  # Need 20s for final processing/return
            break
        
        # Try to find a valid graph for this n
        edges = _search_graph(n, deadline)
        if edges is not None:
            return n, edges
            
    return q, fallback_edges

def _search_graph(n, deadline):
    # Try multiple random restarts
    # Limit restarts to avoid spending too much time on one n
    # If n=42 is hard, we should spend most time there.
    # If n=42 fails quickly, move to 41.
    
    # We will allocate a time budget per n
    # But simple approach: try a fixed number of steps, if fails, next n.
    
    max_restarts = 20
    max_steps = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            return None
            
        # Initialize random graph
        adj = [0] * n
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.add((i, j))
        
        # Repair loop
        for step in range(max_steps):
            if time.time() > deadline - 5:
                return None
                
            violation = _find_violation_optimized(n, adj)
            
            if violation is None:
                # Valid graph found
                return list(edges)
                
            kind, vertices = violation
            
            if kind == 'clique':
                # Remove a random edge from the clique
                v1, v2 = random.sample(vertices, 2)
                if v1 > v2: v1, v2 = v2, v1
                if (v1, v2) in edges:
                    edges.remove((v1, v2))
                    adj[v1] &= ~(1 << v2)
                    adj[v2] &= ~(1 << v1)
            elif kind == 'indep':
                # Add a random edge to the independent set
                v1, v2 = random.sample(vertices, 2)
                if v1 > v2: v1, v2 = v2, v1
                if (v1, v2) not in edges:
                    edges.add((v1, v2))
                    adj[v1] |= (1 << v2)
                    adj[v2] |= (1 << v1)
                    
        # If we exit the step loop, this restart failed
        continue
        
    return None

def _find_violation_optimized(n, adj):
    # For each vertex v, check neighbors and non-neighbors for K4
    # Randomize vertex order to avoid systematic bias
    vertices = list(range(n))
    random.shuffle(vertices)
    
    for v in vertices:
        # Get neighbors of v
        neighbors = []
        non_neighbors = []
        for u in range(n):
            if u == v: continue
            if (adj[v] >> u) & 1:
                neighbors.append(u)
            else:
                non_neighbors.append(u)
                
        # Check for K4 in neighbors
        k4 = _find_k4(neighbors, adj)
        if k4:
            return ('clique', (v,) + tuple(k4))
            
        # Check for Independent Set of size 4 in non_neighbors
        indep4 = _find_indep4(non_neighbors, adj)
        if indep4:
            return ('indep', (v,) + tuple(indep4))
            
    return None

def _find_k4(vertices, adj):
    # Find a clique of size 4 in the induced subgraph on 'vertices'
    for comb in combinations(vertices, 4):
        v0, v1, v2, v3 = comb
        if ((adj[v0] >> v1) & 1) and ((adj[v0] >> v2) & 1) and ((adj[v0] >> v3) & 1) and \
           ((adj[v1] >> v2) & 1) and ((adj[v1] >> v3) & 1) and \
           ((adj[v2] >> v3) & 1):
            return comb
    return None

def _find_indep4(vertices, adj):
    # Find an independent set of size 4 in the induced subgraph on 'vertices'
    for comb in combinations(vertices, 4):
        v0, v1, v2, v3 = comb
        if not ((adj[v0] >> v1) & 1) and not ((adj[v0] >> v2) & 1) and not ((adj[v0] >> v3) & 1) and \
           not ((adj[v1] >> v2) & 1) and not ((adj[v1] >> v3) & 1) and \
           not ((adj[v2] >> v3) & 1):
            return comb
    return None
