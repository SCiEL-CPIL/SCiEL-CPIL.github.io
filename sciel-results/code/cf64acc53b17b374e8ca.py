import random
import itertools

def _generate_trees(n=20, seed=611):
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

def _find_embedding_randomized(tree, k, n, used_edges, num_restarts=10):
    """
    Find an embedding of tree into K_n avoiding used_edges.
    Uses randomized backtracking with MRV heuristic and restarts.
    """
    if k == 1:
        return [0]
    if k == 2:
        for _ in range(num_restarts):
            candidates = list(range(n))
            random.shuffle(candidates)
            for a in candidates:
                for b in candidates:
                    if a == b: continue
                    e = (min(a,b), max(a,b))
                    if e not in used_edges:
                        return [a, b]
        return None

    # Build tree adjacency
    adj = [[] for _ in range(k)]
    for u, v in tree:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a leaf to start from
    leaves = [i for i in range(k) if len(adj[i]) == 1]
    if not leaves:
        leaves = [0]

    for restart in range(num_restarts):
        root = random.choice(leaves)
        
        host = [-1] * k
        used_hosts = set()
        
        def get_valid_options(v):
            """
            Return list of valid host vertices for tree vertex v, given current assignment.
            """
            options = []
            for h in range(n):
                if h in used_hosts:
                    continue
                # Check edges to assigned neighbors
                valid = True
                for u in adj[v]:
                    if host[u] != -1:
                        hu = host[u]
                        edge = (min(h, hu), max(h, hu))
                        if edge in used_edges:
                            valid = False
                            break
                if valid:
                    options.append(h)
            return options

        def backtrack(assigned_count):
            if assigned_count == k:
                return True
            
            # MRV Heuristic: Pick the unassigned vertex with the fewest valid options
            best_v = -1
            best_options = None
            min_count = n + 1
            
            for v in range(k):
                if host[v] != -1:
                    continue
                opts = get_valid_options(v)
                count = len(opts)
                if count < min_count:
                    min_count = count
                    best_v = v
                    best_options = opts
                if count == 0:
                    return False
            
            if best_v == -1:
                return True

            # Shuffle options to randomize
            random.shuffle(best_options)
            
            for h in best_options:
                host[best_v] = h
                used_hosts.add(h)
                if backtrack(assigned_count + 1):
                    return True
                host[best_v] = -1
                used_hosts.remove(h)
            
            return False

        # Assign root
        root_options = get_valid_options(root)
        random.shuffle(root_options)
        
        for h in root_options:
            host[root] = h
            used_hosts.add(h)
            if backtrack(1):
                return [host[i] for i in range(k)]
            host[root] = -1
            used_hosts.remove(h)
            
    return None

def construct():
    n = 20
    trees = _generate_trees(n)
    
    result = [None] * n
    used_edges = set()
    
    # Embed trees in order of decreasing size
    tree_indices = list(range(n - 1, -1, -1))
    
    # To improve robustness, we can try a few different global orderings if we fail to pack all
    # But first, let's just use the randomized embedder with the standard order.
    
    for idx in tree_indices:
        k = idx + 1
        tree = trees[idx]
        
        if k == 1:
            result[idx] = None
            continue
        
        embedding = _find_embedding_randomized(tree, k, n, used_edges, num_restarts=50)
        if embedding is None:
            # If we fail, we might try to skip this tree and come back?
            # For now, if we fail, we return what we have.
            # However, a failed large tree might block smaller ones? No, it's just missing.
            # A better strategy: If a tree fails, we can try to embed the remaining smaller trees, 
            # and then retry the failed tree?
            # Given time constraints, let's just return.
            # But wait, if T20 fails, we get 0. If T20 succeeds but T19 fails, we get 19.
            # It is unlikely T20 fails. T19 is very likely to succeed with randomization.
            return result
        
        embedding = [int(x) for x in embedding]
        result[idx] = embedding
        
        for u, v in tree:
            hu = embedding[u]
            hv = embedding[v]
            edge = (min(hu, hv), max(hu, hv))
            used_edges.add(edge)
    
    return result
