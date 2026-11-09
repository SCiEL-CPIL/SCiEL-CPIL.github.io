import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    # Perfect squares up to 2*N
    squares = set()
    i = 1
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Filter self-conflicts: 2*v must not be a perfect square
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    node_set = set(valid_nodes)
    
    # Build adjacency: adj[v] = list of u in valid_nodes, u != v, such that u+v is a perfect square
    adj = {v: [] for v in valid_nodes}
    for idx, v in enumerate(valid_nodes):
        for u in valid_nodes[idx + 1:]:
            if v + u in squares:
                adj[v].append(u)
                adj[u].append(v)
    
    def is_independent(s):
        """Check if set s is independent (no two elements sum to a perfect square)."""
        s_list = list(s)
        for i in range(len(s_list)):
            a = s_list[i]
            # Check diagonal
            if 2 * a in squares:
                return False
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if a + b in squares:
                    return False
        return True
    
    def greedy_mis(order):
        """Greedy independent set following given order."""
        chosen = set()
        for v in order:
            if v not in node_set:
                continue
            # Check if v conflicts with any already chosen
            conflict = False
            for u in chosen:
                if v + u in squares:
                    conflict = True
                    break
            if not conflict:
                chosen.add(v)
        return chosen
    
    def local_search(s):
        """Try to improve set s by adding elements or swapping."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            # Try adding
            for v in valid_nodes:
                if v in s:
                    continue
                can_add = True
                for u in s:
                    if v + u in squares:
                        can_add = False
                        break
                if can_add:
                    s.add(v)
                    improved = True
                    break
            # Try swapping: remove one, add another
            if not improved:
                for u in list(s):
                    s.remove(u)
                    for v in valid_nodes:
                        if v in s:
                            continue
                        can_add = True
                        for w in s:
                            if v + w in squares:
                                can_add = False
                                break
                        if can_add:
                            s.add(v)
                            improved = True
                            break
                    if improved:
                        break
                    s.add(u)
        return s
    
    best_set = set()
    best_size = 0
    
    # Run multiple random orderings
    rng = random.Random(42)
    
    # First, try a deterministic ordering (by degree)
    degree_order = sorted(valid_nodes, key=lambda v: len(adj[v]))
    s = greedy_mis(degree_order)
    s = local_search(s)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
    
    # Multiple random restarts
    num_restarts = 500
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        order = valid_nodes[:]
        rng.shuffle(order)
        s = greedy_mis(order)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # Final local search on best
    if time.time() < deadline:
        best_set = local_search(best_set)
    
    return sorted(best_set)
