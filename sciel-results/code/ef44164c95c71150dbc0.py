import time
import random

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill

    # Precompute: for each value a, what (b,c) pairs would form a triple with a as smallest
    # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2, so for each divisor d of a^2: b=a+d, c=a+a^2/d
    # For each a, store list of (b,c) pairs
    triples_as_smallest = {}
    triples_as_middle = {}
    triples_as_largest = {}

    for a in range(1, 181):
        a2 = a * a
        # a is smallest: (b-a)(c-a) = a^2
        for d in range(1, a + 1):
            if a2 % d != 0:
                continue
            d2 = a2 // d
            b = a + d
            c = a + d2
            if b > a and c > b and b <= 180 and c <= 180:
                triples_as_smallest.setdefault(a, []).append((b, c))
            # also symmetric: swap roles of d and d2 if different
            if d2 != d:
                b2 = a + d2
                c2 = a + d
                if b2 > a and c2 > b2 and b2 <= 180 and c2 <= 180:
                    triples_as_smallest.setdefault(a, []).append((b2, c2))
        
        # a is middle: 1/x = 1/a + 1/c => c = a*x/(a-x), x < a
        for x in range(1, a):
            d = a - x
            num = a * x
            if num % d != 0:
                continue
            c = num // d
            if c > a and c <= 180:
                triples_as_middle.setdefault(a, []).append((x, c))
        
        # a is largest: 1/x = 1/y + 1/a => y = a*x/(a-x), x < y < a
        for x in range(1, a):
            d = a - x
            num = a * x
            if num % d != 0:
                continue
            y = num // d
            if x < y < a:
                triples_as_largest.setdefault(a, []).append((x, y))

    def can_add_fast(a, S):
        """Check if adding 'a' to valid set 'S' keeps it valid, using precomputed triples."""
        # a is smallest
        for (b, c) in triples_as_smallest.get(a, []):
            if b in S and c in S:
                return False
        # a is middle
        for (x, c) in triples_as_middle.get(a, []):
            if x in S and c in S:
                return False
        # a is largest
        for (x, y) in triples_as_largest.get(a, []):
            if x in S and y in S:
                return False
        return True

    # Precompute all valid triples for verification
    all_triples = []
    for a in range(1, 181):
        a2 = a * a
        for d in range(1, a2 + 1):
            if a2 % d != 0:
                continue
            b = a + d
            c = a + a2 // d
            if b > a and c > b and b <= 180 and c <= 180:
                all_triples.append((a, b, c))

    def is_valid(S):
        for (a, b, c) in all_triples:
            if a in S and b in S and c in S:
                return False
        return True

    odds = set(i for i in range(1, 181) if i % 2 == 1)
    evens = list(range(2, 181, 2))

    # Strategy 1: Start with odds, add evens ascending
    result1 = set(odds)
    for x in evens:
        if time.time() > deadline - 5:
            break
        if can_add_fast(x, result1):
            result1.add(x)

    best = result1
    best_size = len(best)

    # Strategy 2: Start with odds, add evens descending
    if time.time() < deadline - 10:
        result2 = set(odds)
        for x in sorted(evens, reverse=True):
            if time.time() > deadline - 5:
                break
            if can_add_fast(x, result2):
                result2.add(x)
        if len(result2) > best_size:
            best = result2
            best_size = len(best)

    # Strategy 3: Start empty, add all numbers ascending
    if time.time() < deadline - 10:
        result3 = set()
        for x in range(1, 181):
            if time.time() > deadline - 5:
                break
            if can_add_fast(x, result3):
                result3.add(x)
        if len(result3) > best_size:
            best = result3
            best_size = len(best)

    # Strategy 4: Start empty, add all numbers descending
    if time.time() < deadline - 10:
        result4 = set()
        for x in range(180, 0, -1):
            if time.time() > deadline - 5:
                break
            if can_add_fast(x, result4):
                result4.add(x)
        if len(result4) > best_size:
            best = result4
            best_size = len(best)

    # Strategy 5: Randomized greedy with different orderings
    if time.time() < deadline - 10:
        random.seed(42)
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            order = list(range(1, 181))
            random.shuffle(order)
            result5 = set()
            for x in order:
                if can_add_fast(x, result5):
                    result5.add(x)
            if len(result5) > best_size:
                best = result5
                best_size = len(best)

    # Strategy 6: Local search - remove one, re-add greedily (improved)
    current = set(best)
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for remove_elem in sorted(current):
                if time.time() > deadline - 3:
                    break
                trial = current - {remove_elem}
                # Try to add back removed element and others in sorted order
                for x in sorted(list(trial) | {remove_elem}):
                    if x not in trial:
                        if can_add_fast(x, trial):
                            trial.add(x)
                if len(trial) > len(current):
                    current = trial
                    improved = True
                    break

    if len(current) > len(best):
        best = current
        best_size = len(best)

    # Strategy 7: More aggressive local search - try removing 2 elements
    if time.time() < deadline - 3:
        current = set(best)
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            improved = False
            elems = sorted(current)
            for i in range(len(elems)):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, min(i + 5, len(elems))):
                    if time.time() > deadline - 3:
                        break
                    trial = current - {elems[i], elems[j]}
                    for x in range(1, 181):
                        if x in trial:
                            continue
                        if can_add_fast(x, trial):
                            trial.add(x)
                    if len(trial) > len(current):
                        current = trial
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    if len(current) > len(best):
        best = current
        best_size = len(best)

    # Final verification
    if not is_valid(best):
        # Fallback to a known valid set
        best = odds

    return list(best)
