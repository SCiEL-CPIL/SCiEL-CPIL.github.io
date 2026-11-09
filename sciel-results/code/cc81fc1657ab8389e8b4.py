import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    n = 192
    max_k = 48
    
    # Precompute: for each a, which triples (b,c,d) with b<c<d, a not in {b,c,d}, have a | (b+c+d)
    # But we'll check dynamically for speed
    
    def is_valid_addition(selected, x):
        """Check if adding x to selected keeps the property valid."""
        # For each a in selected + [x], check that a doesn't divide sum of any 3 distinct others
        candidates = selected + [x]
        for i, a in enumerate(candidates):
            others = [candidates[j] for j in range(len(candidates)) if j != i]
            # Check all triples from others
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    return False
        return True
    
    def full_check(selected):
        """Verify the entire set is valid."""
        for i, a in enumerate(selected):
            others = [selected[j] for j in range(len(selected)) if j != i]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    return False
        return True
    
    # Greedy: try to build a large set
    # Strategy: start with large numbers (less likely to divide small sums)
    # and numbers that are "hard to divide" (prime, etc.)
    
    def greedy_build(order):
        """Build a set by adding elements in the given order if valid."""
        selected = []
        for x in order:
            if len(selected) >= max_k:
                break
            if is_valid_addition(selected, x):
                selected.append(x)
        return selected
    
    # Try multiple orderings
    best = []
    
    # Order 1: largest first
    order1 = list(range(n, 0, -1))
    best = greedy_build(order1)
    
    if time.time() > deadline:
        return best
    
    # Order 2: prioritize primes and large numbers
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    
    # Score: prefer large, prefer prime, prefer numbers with few small divisors
    def score_number(x):
        s = x  # larger is better
        if is_prime(x):
            s += 100
        # Fewer divisors means less likely to divide sums
        div_count = sum(1 for i in range(2, x+1) if x % i == 0)
        s -= div_count * 5
        return s
    
    scored = [(score_number(x), x) for x in range(1, n+1)]
    scored.sort(key=lambda t: -t[0])
    order2 = [x for _, x in scored]
    sol2 = greedy_build(order2)
    if len(sol2) > len(best):
        best = sol2
    
    if time.time() > deadline:
        return best
    
    # Order 3: random restarts with different seeds
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a random order biased toward large numbers
        weights = [x**2 for x in range(1, n+1)]  # bias toward large
        order3 = list(range(1, n+1))
        random.shuffle(order3)
        # Mix: 70% large-first, 30% random
        large_first = list(range(n, 1, -1))
        mixed = []
        for x in order3:
            if random.random() < 0.3:
                mixed.append(x)
            else:
                mixed.append(x)
        # Actually, let's try: sort by random + size bias
        order3 = sorted(range(1, n+1), key=lambda x: random.random() * 50 + x)
        sol3 = greedy_build(order3)
        if len(sol3) > len(best):
            best = sol3
    
    if time.time() > deadline:
        return best
    
    # Local search: try swaps to improve
    def try_improve(selected):
        """Try to add more elements or swap to get a larger valid set."""
        selected = list(selected)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding any unused element
            used = set(selected)
            for x in range(n, 0, -1):
                if x not in used:
                    if is_valid_addition(selected, x):
                        selected.append(x)
                        improved = True
                        break
            if len(selected) >= max_k:
                break
            # Try swaps: remove one, add one
            for i in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                temp = selected[:i] + selected[i+1:]
                used_temp = set(temp)
                for x in range(n, 0, -1):
                    if x not in used_temp:
                        if is_valid_addition(temp, x):
                            if len(temp + [x]) > len(selected):
                                selected = temp + [x]
                                improved = True
                                break
        return selected
    
    best = try_improve(best)
    
    if time.time() > deadline:
        return best
    
    # More aggressive local search with random restarts
    random.seed(123)
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Start from current best, remove a random element, try to rebuild
        if len(best) > 0:
            temp = list(best)
            idx = random.randint(0, len(temp)-1)
            temp.pop(idx)
            # Try to add back different elements
            temp = try_improve(temp)
            if len(temp) > len(best):
                best = temp
    
    return best
