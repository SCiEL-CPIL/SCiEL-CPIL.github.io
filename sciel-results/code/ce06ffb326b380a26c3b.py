# EVOLVE-BLOCK-START
from math import gcd
import random

def construct():
    N = 128
    # Precompute lcm for all pairs (1..128)
    lcms = [[0]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            lcms[i-1][j-1] = (i*j)//gcd(i,j)
            lcms[j-1][i-1] = lcms[i-1][j-1]
    
    # Precompute conflicts
    # conflicts[i] is a set of frozensets of pairs (j, k) such that {i, j, k} is a bad triple
    # Actually, for greedy, we just need to know if adding i creates a bad triple with two existing elements.
    # So for each i, we store the list of pairs (j, k) with j < k, j != i, k != i
    # such that (i, j, k) is bad.
    # If j and k are in S, we cannot add i.
    
    conflicts = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(a+1, N):
            L_ab = lcms[a-1][b-1]
            for c in range(1, N):
                if c == a or c == b:
                    continue
                if lcms[a-1][c-1] == L_ab and lcms[b-1][c-1] == L_ab:
                    ai, bi, ci = a-1, b-1, c-1
                    conflicts[ai].add(frozenset((bi, ci)))
                    conflicts[bi].add(frozenset((ai, ci)))
                    conflicts[ci].add(frozenset((ai, bi)))
    
    # Degree for heuristic
    degree = [len(conflicts[i]) for i in range(N)]
    
    def greedy(order):
        selected = set()
        for idx in order:
            can_add = True
            for pair in conflicts[idx]:
                a, b = tuple(pair)
                if a in selected and b in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(idx)
        
        # Completion phase: try to add any unselected element
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if i in selected:
                    continue
                can_add = True
                for pair in conflicts[i]:
                    a, b = tuple(pair)
                    if a in selected and b in selected:
                        can_add = False
                        break
                if can_add:
                    selected.add(i)
                    changed = True
        return len(selected), selected

    best_size = 0
    best_set = set()
    
    # Try different orderings
    # 1. By degree (ascending)
    order1 = sorted(range(N), key=lambda i: (degree[i], i+1))
    size1, set1 = greedy(order1)
    if size1 > best_size:
        best_size = size1
        best_set = set1
        
    # 2. By value (ascending)
    order2 = list(range(N))
    size2, set2 = greedy(order2)
    if size2 > best_size:
        best_size = size2
        best_set = set2
        
    # 3. Randomized restarts
    random.seed(42) # Fixed seed for reproducibility in evaluation if needed, or just let it run
    for _ in range(50):
        order3 = list(range(N))
        random.shuffle(order3)
        size3, set3 = greedy(order3)
        if size3 > best_size:
            best_size = size3
            best_set = set3

    result = sorted(i+1 for i in best_set)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
