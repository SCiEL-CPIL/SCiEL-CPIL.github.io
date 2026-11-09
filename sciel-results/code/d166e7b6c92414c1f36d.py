"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256}

def _can_add(S, x):
    """Check if adding x to set S maintains validity.
    S is a set of ints. x is not in S.
    Validity: for all distinct a,b,c in S∪{x}, a ∤ b*c."""
    S_list = list(S)
    # Case 1: x is the divisor. Need: x ∤ a*b for all distinct a,b in S
    for i in range(len(S_list)):
        for j in range(len(S_list)):
            if i == j:
                continue
            if (S_list[i] * S_list[j]) % x == 0:
                return False
    # Case 2: x is one of the multiplicands. Need: a ∤ x*b for all distinct a,b in S
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(len(S_list)):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    return True

def _is_valid(S):
    """Full validation check."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy(order):
    S = set()
    for x in order:
        if _can_add(S, x):
            S.add(x)
    return S

def _local_search(S, max_iters=200, deadline=None):
    """Improve S by single-element swaps."""
    S = set(S)
    best = set(S)
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        for x in list(S):
            # Try replacing x with y
            for y in range(2, P['n'] + 1):
                if y in S:
                    continue
                S_new = set(S)
                S_new.discard(x)
                S_new.add(y)
                if _can_add(S_new - {y}, y):
                    if len(S_new) > len(best):
                        best = set(S_new)
                    S = S_new
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best

def construct():
    start = time.time()
    deadline = start + 50
    
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = False
        return [i for i in range(2, n+1) if is_p[i]]
    
    primes = sieve(P['n'])
    composites = [x for x in range(2, P['n']+1) if x not in primes]
    
    # Initial: big primes
    big_primes = set(p for p in primes if p > 128)
    best = set(big_primes)
    
    rng = random.Random(123)
    
    strategies = []
    # Descending
    strategies.append(list(range(P['n'], 1, -1)))
    # Ascending
    strategies.append(list(range(2, P['n']+1)))
    # Primes then composites
    strategies.append(primes + composites)
    # Large composites first
    strategies.append(sorted(composites, reverse=True) + primes)
    
    for i in range(10):
        if time.time() > deadline:
            break
        if i < len(strategies):
            order = strategies[i]
        else:
            order = list(range(2, P['n']+1))
            rng.shuffle(order)
        
        S = _greedy(order)
        if len(S) > len(best):
            best = S
    
    # Local search from best
    if time.time() < deadline - 3:
        best = _local_search(best, max_iters=300, deadline=deadline - 3)
    
    # More local search rounds
    if time.time() < deadline - 1:
        best = _local_search(best, max_iters=200, deadline=deadline - 1)
    
    result = [int(x) for x in best]
    return result
# EVOLVE-BLOCK-END
