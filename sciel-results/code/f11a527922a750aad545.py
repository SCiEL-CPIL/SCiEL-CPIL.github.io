import random

def construct():
    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Build conflict graph
    conflicts = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            if a + b in squares:
                conflicts[a].add(b)
                if a != b:
                    conflicts[b].add(a)
    
    def greedy(order):
        chosen = set()
        for v in order:
            if not any(c in chosen for c in conflicts[v]):
                chosen.add(v)
        return chosen
    
    best_set = set()
    
    # Deterministic orders
    deg_order = sorted(range(1, n + 1), key=lambda x: len(conflicts[x]))
    s = greedy(deg_order)
    if len(s) > len(best_set):
        best_set = s
    
    deg_order_rev = sorted(range(1, n + 1), key=lambda x: -len(conflicts[x]))
    s = greedy(deg_order_rev)
    if len(s) > len(best_set):
        best_set = s
    
    s = greedy(list(range(1, n + 1)))
    if len(s) > len(best_set):
        best_set = s
    
    # Random restarts
    random.seed(42)
    for _ in range(5000):
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = greedy(order)
        if len(s) > len(best_set):
            best_set = s
    
    return sorted(best_set)
