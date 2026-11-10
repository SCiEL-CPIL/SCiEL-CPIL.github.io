import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 192
    max_k = 48
    
    def is_valid_set(S):
        """Check if set S satisfies the constraint."""
        S_list = list(S)
        for i, a in enumerate(S_list):
            others = [S_list[j] for j in range(len(S_list)) if j != i]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    return False
        return True
    
    def check_addition(S, x):
        """Check if x can be added to set S."""
        # Check: for each a in S ∪ {x}, no triple of OTHERS sums to multiple of a
        new_S = list(S) + [x]
        # Check x against existing: x should not divide sum of any 3 others
        for triple in combinations(list(S), 3):
            if sum(triple) % x == 0:
                return False
        # Check existing elements: a should not divide sum of any 3 from (S\{a}) ∪ {x}
        for a in S:
            others = [v for v in S if v != a]
            # Triples from others only (already valid) or with x
            # Need to check triples that include x
            for pair in combinations(others, 2):
                s = pair[0] + pair[1] + x
                if s % a == 0:
                    return False
        return True
    
    # Greedy: try different orderings
    best_set = []
    
    # Ordering 1: descending (large numbers first)
    for order in [range(n, 0, -1), range(1, n+1)]:
        if time.time() > deadline:
            break
        S = []
        for x in order:
            if len(S) >= max_k:
                break
            if check_addition(S, x):
                S.append(x)
        if len(S) > len(best_set):
            best_set = S[:]
    
    # Try to improve with local search: remove one, try adding others
    if time.time() < deadline - 10:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each element and seeing if we can add more
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                saved = current[i]
                current_without = current[:i] + current[i+1:]
                # Try to add elements not in current
                added = False
                for x in range(n, 0, -1):
                    if x in current_without or x == saved:
                        continue
                    if check_addition(current_without, x):
                        new_set = current_without + [x]
                        if len(new_set) > len(current):
                            current = new_set
                            improved = True
                            added = True
                            break
                        elif len(new_set) == len(current):
                            # Try swap: replace saved with x
                            if x > saved:
                                current = current_without + [x]
                                improved = True
                                added = True
                                break
                if not added:
                    current = current_without + [saved]
    
    # Final attempt: greedy with random perturbations
    if time.time() < deadline - 5:
        import random
        random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            S = []
            # Random ordering biased toward large numbers
            order = list(range(1, n+1))
            random.shuffle(order)
            # Sort with bias: larger numbers more likely first
            order.sort(key=lambda x: -x + random.random()*20)
            for x in order:
                if len(S) >= max_k:
                    break
                if check_addition(S, x):
                    S.append(x)
            if len(S) > len(best_set):
                best_set = S[:]
    
    return best_set
