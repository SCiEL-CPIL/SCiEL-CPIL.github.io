import time
import random
from math import gcd
from itertools import combinations

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Precompute conflicts
    # conflicts[x] is a set of frozensets({u, v}) such that (u, v, x) is a bad triple
    conflicts = [set() for _ in range(N + 1)]
    
    # Helper to check bad triple
    def is_bad(a, b, c):
        g_ab = gcd(a, b)
        g_ac = gcd(a, c)
        g_bc = gcd(b, c)
        return g_ab == g_ac == g_bc

    # Precompute all bad triples
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if is_bad(a, b, c):
                    conflicts[a].add(frozenset((b, c)))
                    conflicts[b].add(frozenset((a, c)))
                    conflicts[c].add(frozenset((a, b)))

    # Greedy function
    def greedy_add(candidates):
        s = set()
        s_pairs = set()  # Set of frozensets({u, v}) for u, v in s
        for x in candidates:
            can_add = True
            for pair in conflicts[x]:
                if pair in s_pairs:
                    can_add = False
                    break
            if can_add:
                s.add(x)
                for y in s:
                    if y != x:
                        s_pairs.add(frozenset((x, y)))
        return s

    best_set = set()
    best_size = 0

    # Deterministic initial orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(conflicts[x])),
        sorted(range(1, N + 1), key=lambda x: -len(conflicts[x]))
    ]

    for order in orders:
        s = greedy_add(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Random restarts
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        s = greedy_add(candidates)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    return list(best_set)
