import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    N = 192
    K = 3
    MAX_SIZE = 48
    
    all_nums = list(range(1, N + 1))
    
    # Precompute all triples of distinct values from 1..192
    # For each value v, we want to know which triples (a,b,c) with a<b<c, a!=b, b!=c, a!=c
    # have a+b+c divisible by v.
    # Actually, we need to check: for a selected set S, for each x in S,
    # no triple (a,b,c) from S\{x} with a,b,c distinct should have (a+b+c) % x == 0.
    
    # Let's precompute for each potential divisor d (1..192),
    # the list of triples (a,b,c) with 1<=a<b<c<=192, a!=d, b!=d, c!=d,
    # such that (a+b+c) % d == 0.
    # This is too large: C(192,3) is ~1.17 million. For each d, we'd store those.
    # Total memory might be high but let's see.
    
    # Alternative: Precompute for each triple (a,b,c) the set of divisors d (1..192)
    # such that d != a, d != b, d != c and (a+b+c) % d == 0.
    # Then, when building a set, we maintain for each candidate x not in S,
    # a count of how many "bad" triples are formed.
    
    # Let's think about the constraint more carefully.
    # A set S is valid iff for all x in S, there is no triple {a,b,c} subset of S\{x}
    # such that (a+b+c) % x == 0.
    
    # This is equivalent to: for all x in S, and for all triples T subset of S,
    # if x not in T, then (sum(T)) % x != 0.
    
    # Let's precompute:
    # For each x in 1..192, let bad_triples[x] be the set of triples (a,b,c) (sorted, distinct, !=x)
    # such that (a+b+c) % x == 0.
    
    # But storing all triples is expensive. C(192,3) = 192*191*190/6 = 1,168,880.
    # For each triple, we need to check which x (not in triple) divide the sum.
    
    # Let's try a different approach:
    # Precompute sum3[a][b][c] = a+b+c for all a<b<c.
    # Then for a given set S, checking validity is O(|S|^4) which is too slow for |S|~48.
    # 48^4 = 5.3 million, times some constant. Might be okay in C but slow in Python.
    
    # Better: Use the forward-checking / backtracking idea.
    # Maintain a set S. When adding x to S, we need to check:
    # 1. For each existing triple in S, does x divide its sum?
    #    i.e., for all a<b<c in S, (a+b+c) % x != 0.
    # 2. For each existing element y in S, does y divide the sum of any triple in S\{y} ∪ {x}?
    #    i.e., for each y in S, for all triples {a,b,c} subset of (S\{y}) ∪ {x} with a,b,c distinct,
    #    (a+b+c) % y != 0.
    #    This can be broken down:
    #    - Triples entirely in S\{y}: already checked when y was added.
    #    - Triples with x and two elements from S\{y}: for each pair {a,b} subset of S\{y},
    #      (x+a+b) % y != 0.
    #    - Triples with x and one element from S\{y} and... wait, triples have 3 elements.
    #      So triples containing x and two from S\{y} are the new ones to check for y.
    #    - Also, triples containing x and... no, x is one element. The triple must have 3 distinct elements.
    #      So the new triples involving x are: {x, a, b} where a,b are distinct elements from S.
    #      For each such triple, and for each y in S, y != x, we need (x+a+b) % y != 0.
    #      Also, for x itself, we need to check that no triple in S (without x) has sum divisible by x.
    
    # So when adding x to S:
    # Check A: For all triples (a,b,c) in S, (a+b+c) % x != 0.
    # Check B: For all pairs (a,b) in S, and for all y in S, y != x, (x+a+b) % y != 0.
    
    # Check A is O(|S|^3). Check B is O(|S|^3) as well (pairs * elements).
    # For |S| up to 48, 48^3 = 110,592. That's manageable.
    
    # Let's implement this.
    
    def is_valid_addition(x, S):
        """Check if x can be added to set S (list of ints) without violating constraints."""
        n = len(S)
        S_set = set(S)
        
        # Check A: x does not divide sum of any triple in S
        if n >= 3:
            for i in range(n):
                a = S[i]
                for j in range(i+1, n):
                    b = S[j]
                    for k in range(j+1, n):
                        c = S[k]
                        if (a + b + c) % x == 0:
                            return False
        
        # Check B: For each y in S, y does not divide (x + a + b) for any pair a,b in S, a!=b, a!=y, b!=y
        # Actually, a and b are from S, and they must be distinct from each other and from y.
        # Also, the triple is {x, a, b}, so x, a, b must be distinct. Since x is new, x != a and x != b.
        # We need a, b distinct from each other and from y.
        for y in S:
            # Find pairs (a, b) in S such that a != b, a != y, b != y
            # and (x + a + b) % y == 0
            for i in range(n):
                a = S[i]
                if a == y:
                    continue
                for j in range(i+1, n):
                    b = S[j]
                    if b == y:
                        continue
                    if (x + a + b) % y == 0:
                        return False
        
        return True
    
    def is_valid_set(S):
        """Verify that set S is valid. Used for final check."""
        S_list = list(S)
        n = len(S_list)
        for i, x in enumerate(S_list):
            # Check if x divides sum of any 3 distinct others
            others = [S_list[j] for j in range(n) if j != i]
            if len(others) < 3:
                continue
            for a, b, c in combinations(others, 3):
                if (a + b + c) % x == 0:
                    return False
        return True
    
    best = []
    best_len = 0
    
    def update_best(S):
        nonlocal best, best_len
        if len(S) > best_len and is_valid_set(S):
            best = list(S)
            best_len = len(S)
    
    # Phase 1: Greedy with various orderings
    def greedy(order, deadline):
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if is_valid_addition(x, S):
                S.append(x)
                S_set.add(x)
        return S
    
    # Ordering 1: Descending
    if time.time() < deadline:
        order = sorted(all_nums, reverse=True)
        res = greedy(order, deadline)
        update_best(res)
    
    # Ordering 2: Primes first
    def is_prime(x):
        if x < 2: return False
        if x < 4: return True
        if x % 2 == 0 or x % 3 == 0: return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i+2) == 0: return False
            i += 6
        return True
    
    if time.time() < deadline:
        primes = [x for x in all_nums if is_prime(x)]
        composites = [x for x in all_nums if not is_prime(x)]
        order = sorted(primes, reverse=True) + sorted(composites, reverse=True)
        res = greedy(order, deadline)
        update_best(res)
    
    # Ordering 3: Randomized greedy with local search
    rng = random.Random(42)
    
    def local_search(S, deadline, max_iters=50):
        S = list(S)
        S_set = set(S)
        best_S = list(S)
        best_len = len(S)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try to add an element
            candidates = [x for x in all_nums if x not in S_set]
            rng.shuffle(candidates)
            for x in candidates:
                if is_valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_len:
                        best_S = list(S)
                        best_len = len(S)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try a swap: remove one, add one
            if len(S) > 0 and len(candidates) > 0:
                # Pick a random element to remove
                idx = rng.randrange(len(S))
                y = S[idx]
                S_new = S[:idx] + S[idx+1:]
                S_set_new = set(S_new)
                
                # Try to add a random candidate
                for x in candidates[:10]:  # Try a few
                    if x in S_set_new:
                        continue
                    if is_valid_addition(x, S_new):
                        S = S_new + [x]
                        S_set = S_set_new | {x}
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_S
    
    # Run several randomized greedy + local search
    while time.time() < deadline - 5.0:
        # Create a random ordering with bias
        order = all_nums[:]
        rng.shuffle(order)
        # Bias towards larger numbers
        order = sorted(order, key=lambda x: rng.random() + x/200.0, reverse=True)
        
        res = greedy(order, deadline)
        if len(res) > best_len:
            update_best(res)
        
        # Local search
        if time.time() < deadline - 2.0:
            res_ls = local_search(res, deadline, max_iters=30)
            update_best(res_ls)
    
    # Final verification
    if best and not is_valid_set(best):
        # Fallback: return empty or a known valid small set
        # Let's find a valid set by greedy descending
        res = greedy(sorted(all_nums, reverse=True), deadline)
        if is_valid_set(res):
            best = res
    
    return best

# Keep the helper functions from the original for compatibility, even if not used
def _is_valid(selected):
    """Check if selected is a valid set."""
    n = len(selected)
    for idx_a, a in enumerate(selected):
        others = [v for i, v in enumerate(selected) if i != idx_a]
        if len(others) < 3:
            continue
        for b, c, d in combinations(others, 3):
            if (b + c + d) % a == 0:
                return False
    return True

def _can_add(x, S):
    """
    Check if x can be added to set S without violating the condition.
    S is a list of valid integers.
    """
    n = len(S)
    
    # 1. x does not divide sum of any 3 distinct elements in S
    if n >= 3:
        for i in range(n):
            si = S[i]
            for j in range(i+1, n):
                sj = S[j]
                for k in range(j+1, n):
                    sk = S[k]
                    if (si + sj + sk) % x == 0:
                        return False
    
    # 2. No a in S divides x + b + c for distinct b, c in S\{a}
    for a in S:
        for i in range(n):
            bi = S[i]
            if bi == a:
                continue
            for j in range(i+1, n):
                bj = S[j]
                if bj == a:
                    continue
                if (x + bi + bj) % a == 0:
                    return False
    
    return True

def _greedy(order, deadline):
    """Greedy construction following given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, pool, deadline, max_iters=100):
    """
    Improve selected by adding from pool or swapping.
    """
    selected = list(selected)
    selected_set = set(selected)
    best = selected
    best_len = len(selected)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try to add
        for x in pool:
            if x in selected_set:
                continue
            if _can_add(x, selected):
                selected.append(x)
                selected_set.add(x)
                if len(selected) > best_len:
                    best = list(selected)
                    best_len = len(selected)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap
        if len(selected) > 0 and len(pool) > 0:
            y = random.choice(selected)
            y_idx = selected.index(y)
            x = random.choice(pool)
            
            candidate = selected[:y_idx] + selected[y_idx+1:]
            if _can_add(x, candidate):
                candidate.append(x)
                selected = candidate
                selected_set = set(selected)
                improved = True
        
        if not improved:
            break
    
    return best

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _num_divisors(x):
    cnt = 0
    for d in range(1, int(x**0.5) + 1):
        if x % d == 0:
            cnt += 1
            if d != x // d:
                cnt += 1
    return cnt
