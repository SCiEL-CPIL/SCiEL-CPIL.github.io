import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    # Build conflict graph: conflicts[v] = set of u in valid_nodes where u != v and u+v in squares
    # Use a dict mapping node -> set of conflicting nodes
    node_set = set(valid_nodes)
    conflicts = {v: set() for v in valid_nodes}
    for i_idx, v in enumerate(valid_nodes):
        for u in valid_nodes[i_idx + 1:]:
            if v + u in squares:
                conflicts[v].add(u)
                conflicts[u].add(v)
    
    # Greedy MIS: pick nodes in some order, add if no conflict with already-picked
    def greedy_mis(order):
        chosen = set()
        for v in order:
            # Check if v conflicts with any already chosen
            ok = True
            for u in conflicts[v]:
                if u in chosen:
                    ok = False
                    break
            if ok:
                chosen.add(v)
        return chosen
    
    # Local search: try to add nodes, or swap
    def local_search(chosen, max_iters=10000):
        chosen = set(chosen)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # Try adding each non-chosen valid node
            for v in valid_nodes:
                if v in chosen:
                    continue
                # Check if v conflicts with any chosen node
                ok = True
                for u in conflicts[v]:
                    if u in chosen:
                        ok = False
                        break
                if ok:
                    chosen.add(v)
                    improved = True
            
            # Try swapping: remove a node, add another
            # This is more complex; let's try: for each pair (remove r, add a)
            # where a not in chosen, r in chosen, and a doesn't conflict with chosen-{r}
            # and r doesn't need to be replaced by something specific
            # Actually, let's try: for each a not in chosen, check if we can add it
            # by removing one of its conflicts from chosen
            for a in valid_nodes:
                if a in chosen:
                    continue
                # Find conflicts of a that are in chosen
                conf_in_chosen = [u for u in conflicts[a] if u in chosen]
                if not conf_in_chosen:
                    # Can add directly
                    chosen.add(a)
                    improved = True
                    continue
                # Try removing each conflict and adding a
                for r in conf_in_chosen:
                    # Check if a conflicts with any other chosen node (besides r)
                    ok = True
                    for u in conflicts[a]:
                        if u in chosen and u != r:
                            ok = False
                            break
                    if ok:
                        # Also need to check: after removing r, does a conflict with remaining?
                        # We already checked above
                        chosen.discard(r)
                        chosen.add(a)
                        improved = True
                        break
        
        return chosen
    
    best = set()
    
    # Try multiple random orderings for greedy
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        order = valid_nodes[:]
        random.shuffle(order)
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=5000)
        if len(chosen) > len(best):
            best = chosen
    
    # Also try sorted orders
    for order in [valid_nodes, sorted(valid_nodes, reverse=True), sorted(valid_nodes, key=lambda x: len(conflicts[x]))]:
        if time.time() > deadline:
            break
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=5000)
        if len(chosen) > len(best):
            best = chosen
    
    if not best:
        best = {valid_nodes[0]}
    
    return list(best)
