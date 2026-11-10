import math
import time
import random

def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Precompute squares and conflict sets
    squares = set(i * i for i in range(1, 23))  # up to 22^2 = 484
    # Also include 1^2 = 1, but min sum is 1+1=2, so 1 is irrelevant for pairs
    # Max sum is 256+256=512, so include squares up to 512
    squares = set(i * i for i in range(1, 23))
    # 22^2 = 484, 23^2 = 529 > 512, so 22 is enough
    
    # conflict[x] = frozenset of y != x such that x+y is a square
    # For diagonal: 2x must not be square
    diag_conflict = set()
    for x in range(1, 257):
        if 2 * x in squares:
            diag_conflict.add(x)
    
    # Build adjacency: adj[x] = set of y (y can be < x or > x) where x+y is square
    adj = [set() for _ in range(257)]
    for a in range(1, 257):
        for s in squares:
            b = s - a
            if 1 <= b <= 256 and b != a:
                adj[a].add(b)
    
    def can_add(x, S):
        """Check if x can be added to set S."""
        if x in diag_conflict:
            return False
        for y in S:
            if y in adj[x]:
                return False
        return True
    
    def greedy_fill(S, remaining):
        """Greedily add elements from remaining."""
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for x in list(remaining):
                if can_add(x, S):
                    S.add(x)
                    remaining.discard(x)
                    changed = True
    
    def local_search_1swap(S, remaining):
        """Try 1-swap: remove one from S, add one from remaining."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            for x in S_list:
                if time.time() > deadline - 2:
                    break
                S.remove(x)
                added = False
                for y in list(remaining):
                    if can_add(y, S):
                        S.add(y)
                        remaining.discard(y)
                        added = True
                        break
                if added:
                    remaining.add(x)
                    improved = True
                else:
                    S.add(x)
        return S, remaining
    
    def perturb(S, remaining, k):
        """Randomly remove k elements and re-add greedily."""
        S_copy = set(S)
        rem_copy = set(remaining)
        S_list = list(S_copy)
        random.shuffle(S_list)
        for x in S_list[:k]:
            S_copy.discard(x)
            rem_copy.add(x)
        # Greedy fill
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for x in list(rem_copy):
                if can_add(x, S_copy):
                    S_copy.add(x)
                    rem_copy.discard(x)
                    changed = True
        return S_copy, rem_copy
    
    best_S = set()
    best_remaining = set(range(1, 257))
    
    # Multiple random restarts with different strategies
    for restart in range(30):
        if time.time() > deadline - 3:
            break
        
        random.seed(restart * 7919 + 31)
        S = set()
        remaining = set(range(1, 257))
        
        # Random order greedy
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S.add(x)
                remaining.discard(x)
        
        # Greedy fill
        greedy_fill(S, remaining)
        
        # Local search
        S, remaining = local_search_1swap(S, remaining)
        
        if len(S) > len(best_S):
            best_S = set(S)
            best_remaining = set(remaining)
        
        # Iterated local search: perturb and search again
        if time.time() < deadline - 5:
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                k = random.randint(1, 5)
                S_p, rem_p = perturb(best_S, best_remaining, k)
                S_p, rem_p = local_search_1swap(S_p, rem_p)
                if len(S_p) > len(best_S):
                    best_S = set(S_p)
                    best_remaining = set(rem_p)
    
    # Final greedy fill
    greedy_fill(best_S, best_remaining)
    
    return list(best_S)
