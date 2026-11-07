"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency for each tree
    tree_adjs = []
    for k in range(1, n + 1):
        adj = defaultdict(list)
        for u, v in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adjs.append(adj)
        
    result = [None] * n
    used_edges = set()
    
    # Process trees in reverse order to prioritize larger trees
    # T20 has 19 edges, T1 has 0.
    for k in range(n, 0, -1):
        if time.time() > deadline:
            break
            
        tree_edges = trees[k-1]
        if not tree_edges:
            result[k-1] = [0]
            continue
            
        # Try to embed tree T_k
        embedding = _embed_tree_sa(k, tree_adjs[k-1], used_edges, n, deadline)
        
        if embedding is not None:
            result[k-1] = embedding
            for u, v in tree_edges:
                hu, hv = embedding[u], embedding[v]
                edge = (min(hu, hv), max(hu, hv))
                used_edges.add(edge)
        else:
            result[k-1] = None
            
    return result

def _embed_tree_sa(k, tree_adj, used_edges, n, deadline):
    """
    Embed tree of size k into K_n avoiding used_edges using simulated annealing.
    Returns list of length k, where embedding[i] is host vertex for tree vertex i, or None.
    """
    if time.time() > deadline:
        return None
        
    if k == 1:
        return [0]
        
    # Get tree edges
    tree_edges = []
    for u in range(k):
        for v in tree_adj[u]:
            if u < v:
                tree_edges.append((u, v))
                
    # Initialize random embedding
    import random
    rng = random.Random()
    embedding = list(range(k))
    rng.shuffle(embedding)
    
    def count_conflicts(emb):
        """Count edge conflicts and vertex collisions."""
        conflicts = 0
        # Check vertex collisions
        seen = set()
        for h in emb:
            if h in seen:
                conflicts += 1
            seen.add(h)
        # Check edge conflicts with used_edges
        for u, v in tree_edges:
            hu, hv = emb[u], emb[v]
            if hu == hv:
                conflicts += 1
            else:
                edge = (min(hu, hv), max(hu, hv))
                if edge in used_edges:
                    conflicts += 1
        return conflicts
    
    # Simulated annealing
    best_emb = embedding[:]
    best_conf = count_conflicts(embedding)
    current_conf = best_conf
    
    t = 1.0
    cooling = 0.995
    min_t = 0.01
    max_steps = 5000
    
    step = 0
    while step < max_steps and t > min_t and time.time() < deadline:
        # Pick random tree vertex and new host
        tv = rng.randrange(k)
        old_h = embedding[tv]
        new_h = rng.randrange(n)
        
        if new_h == old_h:
            step += 1
            continue
            
        # Calculate delta in conflicts
        old_conf = current_conf
        
        # Remove old assignments for tv
        embedding[tv] = new_h
        new_conf = count_conflicts(embedding)
        
        delta = new_conf - old_conf
        
        if delta <= 0 or rng.random() < (1.0 / (1.0 + t)):
            # Accept
            current_conf = new_conf
            if new_conf < best_conf:
                best_conf = new_conf
                best_emb = embedding[:]
        else:
            # Reject
            embedding[tv] = old_h
            
        t *= cooling
        step += 1
        
    if best_conf == 0:
        return best_emb
    else:
        return None
# EVOLVE-BLOCK-END
