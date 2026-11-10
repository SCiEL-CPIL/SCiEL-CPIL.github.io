import time
import random
from math import gcd

def construct():
    # Strict deadline: return by 45s to be safe against 60s/90s limits
    deadline = time.time() + 45
    N = 128

    # Precompute gcd matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict pairs: for each v, list of (a,b) such that {a,b,v} is a bad triple
    # a < b
    conflict_pairs = [[] for _ in range(N)]
    for a in range(N):
        for b in range(a+1, N):
            ga = gcd_mat[a+1][b+1]
            for c in range(N):
                if c == a or c == b:
                    continue
                if gcd_mat[a+1][c+1] == ga and gcd_mat[b+1][c+1] == ga:
                    # Triple (a,b,c) is bad
                    conflict_pairs[c].append((a,b))
                    conflict_pairs[a].append((b,c))
                    conflict_pairs[b].append((a,c))

    def can_add(v, selected_set):
        """Check if v can be added to selected_set without creating a bad triple."""
        for (a, b) in conflict_pairs[v]:
            if a in selected_set and b in selected_set:
                return False
        return True

    def greedy(order):
        """Greedy construction with given order."""
        selected = []
        selected_set = set()
        for v in order:
            if can_add(v, selected_set):
                selected.append(v)
                selected_set.add(v)
        return selected, selected_set

    def local_improve(selected, selected_set):
        """Try to add any vertex that can fit. Repeat until no more can be added."""
        improved = True
        while improved:
            improved = False
            for v in range(N):
                if v in selected_set:
                    continue
                if can_add(v, selected_set):
                    selected.append(v)
                    selected_set.add(v)
                    improved = True
        return selected, selected_set

    best = []
    best_set = set()
    best_size = 0
    random.seed(42)

    # Phase 1: Deterministic orders
    orders = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        sorted(range(N), key=lambda x: len(conflict_pairs[x])),
        sorted(range(N), key=lambda x: -len(conflict_pairs[x])),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        selected, selected_set = greedy(order)
        selected, selected_set = local_improve(selected, selected_set)
        if len(selected) > best_size:
            best_size = len(selected)
            best = list(selected)
            best_set = set(selected)

    # Phase 2: Randomized restarts
    max_restarts = 1000
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        selected, selected_set = greedy(order)
        selected, selected_set = local_improve(selected, selected_set)
        if len(selected) > best_size:
            best_size = len(selected)
            best = list(selected)
            best_set = set(selected)
            
        # Optional: Perturbation
        if random.random() < 0.2 and selected:
            for _ in range(random.randint(1, 2)):
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    v = selected.pop(idx)
                    selected_set.discard(v)
            selected, selected_set = local_improve(selected, selected_set)
            if len(selected) > best_size:
                best_size = len(selected)
                best = list(selected)
                best_set = set(selected)

    if not best:
        best = [0] # 1 is index 0
        
    # Return values 1..128
    return sorted([x+1 for x in best])
