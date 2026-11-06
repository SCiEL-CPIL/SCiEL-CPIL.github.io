import random
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_valid(S):
        """Check if set S has no bad triple."""
        for a, b, c in combinations(S, 3):
            if gcd_table[a][b] == gcd_table[b][c] == gcd_table[c][a]:
                return False
        return True

    def can_add(S_set, x):
        """Check if x can be added to S_set without forming a bad triple."""
        S_list = list(S_set)
        for i in range(len(S_list)):
            a = S_list[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if g_ax == gcd_table[b][x] == gcd_table[a][b]:
                    return False
        return True

    def count_conflicts(S_set, x):
        """Count bad triples that would form if x is added."""
        S_list = list(S_set)
        count = 0
        for i in range(len(S_list)):
            a = S_list[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if g_ax == gcd_table[b][x] == gcd_table[a][b]:
                    count += 1
        return count

    def greedy_construct(order):
        S = []
        S_set = set()
        for x in order:
            if can_add(S_set, x):
                S.append(x)
                S_set.add(x)
        return S, S_set

    def greedy_min_conflict(items):
        """Greedy: at each step pick element with min conflicts."""
        S = []
        S_set = set()
        available = list(items)
        while available and time.time() < deadline:
            best_x = None
            best_conf = float('inf')
            for x in available:
                c = count_conflicts(S_set, x)
                if c < best_conf:
                    best_conf = c
                    best_x = x
            if best_x is None or best_conf > 0:
                break
            S.append(best_x)
            S_set.add(best_x)
            available.remove(best_x)
        return S, S_set

    def local_search(S, S_set, items_base):
        """Try to add remaining elements in various orders."""
        S = list(S)
        S_set = set(S)
        remaining = [x for x in items_base if x not in S_set]
        # Try multiple random orders
        best = (S, S_set)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            S2 = list(S)
            S2_set = set(S)
            rem = [x for x in items_base if x not in S2_set]
            random.shuffle(rem)
            for x in rem:
                if can_add(S2_set, x):
                    S2.append(x)
                    S2_set.add(x)
            if len(S2) > len(best[0]):
                best = (S2, S2_set)
        return best

    def shake_search(S, S_set, items_base):
        """Perturbation: remove k random elements, re-add greedily."""
        S = list(S)
        S_set = set(S)
        best = (S, S_set)
        
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Remove 1-3 random elements
            k = random.randint(1, 3)
            removed = random.sample(S, min(k, len(S)))
            S2 = [x for x in S if x not in removed]
            S2_set = set(S2)
            
            # Try to re-add all remaining (including removed) in random order
            candidates = [x for x in items_base if x not in S2_set]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(S2_set, x):
                    S2.append(x)
                    S2_set.add(x)
            
            if len(S2) > len(best[0]):
                best = (S2, S2_set)
        
        return best

    def swap_search(S, S_set, items_base):
        """1-for-1 swap: remove one, try to add two."""
        S = list(S)
        S_set = set(S)
        best = (S, S_set)
        
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            # Pick a random element to remove
            if not S:
                break
            u = random.choice(S)
            S2 = [x for x in S if x != u]
            S2_set = set(S2)
            
            # Try to add elements back
            candidates = [x for x in items_base if x not in S2_set]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(S2_set, x):
                    S2.append(x)
                    S2_set.add(x)
            
            if len(S2) > len(best[0]):
                best = (S2, S2_set)
        
        return best

    items_base = list(range(2, N+1))
    best_size = 0
    best_set = []
    best_set_set = set()

    # Phase 1: Multiple greedy restarts
    while time.time() < deadline - 25:
        order = list(items_base)
        random.shuffle(order)
        S, S_set = greedy_construct(order)
        S, S_set = local_search(S, S_set, items_base)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            best_set_set = set(S)

    # Phase 2: Conflict-based greedy
    while time.time() < deadline - 15:
        S, S_set = greedy_min_conflict(items_base)
        S, S_set = local_search(S, S_set, items_base)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            best_set_set = set(S)

    # Phase 3: Shake search on best
    if time.time() < deadline - 10:
        S, S_set = shake_search(best_set, best_set_set, items_base)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            best_set_set = set(S)

    # Phase 4: Swap search
    if time.time() < deadline - 5:
        S, S_set = swap_search(best_set, best_set_set, items_base)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            best_set_set = set(S)

    # Final: try adding 1
    if best_set and time.time() < deadline - 1:
        can_add_one = True
        S_list = list(best_set_set)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if gcd_table[S_list[i]][S_list[j]] == 1:
                    can_add_one = False
                    break
            if not can_add_one:
                break
        if can_add_one:
            best_set.append(1)
            best_size += 1

    return best_set

