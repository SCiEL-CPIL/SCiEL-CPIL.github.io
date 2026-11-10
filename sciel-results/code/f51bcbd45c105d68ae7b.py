import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 192
    max_k = 48
    
    best_set = []
    best_size = 0

    def can_add(x, S):
        """
        Check if x can be added to set S.
        S is a list of distinct integers.
        Condition:
        1. x does not divide sum of any 3 distinct elements in S.
        2. No element a in S divides x + u + v for any distinct u, v in S\{a}.
        """
        n_s = len(S)
        
        # 1. x divides sum of 3 in S?
        if n_s >= 3:
            for c in combinations(S, 3):
                if sum(c) % x == 0:
                    return False
        
        # 2. a in S divides x + u + v?
        if n_s >= 2:
            for i, a in enumerate(S):
                for j in range(n_s):
                    if j == i:
                        continue
                    u = S[j]
                    for k in range(j + 1, n_s):
                        if k == i:
                            continue
                        v = S[k]
                        if (x + u + v) % a == 0:
                            return False
        
        return True

    def run_greedy(order_type):
        """
        Run a greedy construction with a specific ordering.
        """
        candidates = list(range(1, n + 1))
        
        if order_type == 'desc':
            candidates.sort(reverse=True)
        elif order_type == 'asc':
            candidates.sort()
        elif order_type == 'random':
            random.shuffle(candidates)

        current_set = []
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current_set):
                current_set.append(x)
                if len(current_set) >= max_k:
                    break
        
        return current_set

    # First, try deterministic Descending
    s = run_greedy('desc')
    if len(s) > best_size:
        best_size = len(s)
        best_set = s[:]

    # Random restarts
    for i in range(30):
        if time.time() > deadline:
            break
        s = run_greedy('random')
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # Local search: try to add any remaining number
    if time.time() < deadline:
        current_best = best_set[:]
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x not in current_best:
                    if can_add(x, current_best):
                        current_best.append(x)
                        added_any = True
                        break
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = current_best[:]

    # Branch and bound: try to improve by swapping elements
    if time.time() < deadline - 5:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing one element and adding two
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                temp = current[:i] + current[i+1:]
                for x in range(1, n + 1):
                    if time.time() > deadline - 5:
                        break
                    if x in temp:
                        continue
                    if can_add(x, temp):
                        temp.append(x)
                        # Try to add one more
                        for y in range(1, n + 1):
                            if time.time() > deadline - 5:
                                break
                            if y in temp:
                                continue
                            if can_add(y, temp):
                                temp.append(y)
                                if len(temp) > len(current):
                                    current = temp[:]
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]

    return best_set
