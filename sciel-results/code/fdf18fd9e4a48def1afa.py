import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        # Generate edges exactly as in the seed
        edges = []
        # Internal clique edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            # If we run out of time, we can't compute the next one.
            # However, these computations are very fast (n<=90), so this shouldn't happen.
            # We break to avoid timeout, but we must return valid data.
            # Since the first two are fast, we likely finish all.
            # If we fail here, we might return partial results which is invalid.
            # Better to ensure we don't hit this.
            pass
            
        edges = instance(n, k, p, seed)
        
        cliques = []
        
        # 1. The big clique on C
        # Vertices 0 to k-1 form a clique
        clique_c = list(range(k))
        # Ensure at least 2 vertices. k=10,20,30 so this is fine.
        cliques.append(clique_c)
        
        # 2. Cross edges
        # An edge (u, v) is a cross edge if one is < k and the other >= k.
        # In the generated edges, internal are (u,v) with u,v < k.
        # Cross are (u, v) with u < k, v >= k.
        # The generator adds internal first, then cross.
        # We can just iterate and check.
        
        for u, v in edges:
            # Check if it's a cross edge
            if (u < k and v >= k) or (v < k and u >= k):
                # It is a cross edge
                # Ensure standard ordering or just use as is?
                # The task says "list of distinct int vertices". Order doesn't matter for set semantics, 
                # but the evaluator might check specific structures? 
                # "Every listed set must be a clique". 
                # The raw value is the count.
                # The type check is on the list.
                # We add the edge as a clique of size 2.
                # Ensure they are Python ints (they are already from range/random)
                cliques.append([int(u), int(v)])
        
        results.append(cliques)

    return results
