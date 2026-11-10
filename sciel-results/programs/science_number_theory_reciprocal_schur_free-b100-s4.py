"""Rewritten program: stronger algorithm for large sets avoiding reciprocal Schur triples."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under the 90s kill
    N = 180
    rng = random.Random(20240517)

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # which means c = a*b/(a+b), so a+b must divide a*b.
    # Equivalently, for distinct a, b, c: 1/a = 1/b + 1/c => a*b = a*c + b*c => 
    # Let's just enumerate: for each pair (b, c) with b < c, compute a = b*c/(b+c)
    # a must be a positive integer distinct from b and c.
    # Actually, the condition is: three DISTINCT elements a, b, c in S such that 1/a = 1/b + 1/c.
    # This is equivalent to a = b*c / (b+c), so (b+c) | (b*c).
    # Note: a, b, c are distinct. a could be less than, between, or greater than b and c.
    # But since 1/a = 1/b + 1/c > 1/c, we have a < c. Similarly a < b.
    # So a is always the smallest of the three.
    
    # Let's precompute all valid triples (a, b, c) where a < b < c, a, b, c in [1, 180],
    # and 1/a = 1/b + 1/c.
    # a = b*c / (b+c). For this to be integer, (b+c) | b*c.
    # Note gcd(b, b+c) = gcd(b, c). Let g = gcd(b, c), b = g*b', c = g*c', gcd(b', c')=1.
    # Then a = g*b'*c' / (b'+c'). Since gcd(b', b'+c') = gcd(b', c') = 1, we need (b'+c') | g.
    # So a = g / (b'+c') * b'*c'.
    
    triples = []  # list of (a, b, c) with a < b < c
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if a < b and 1 <= a <= N:
                    triples.append((a, b, c))
    # Also need to consider cases where the "a" in 1/a=1/b+1/c might not be the smallest
    # Wait, I argued a < b and a < c. Let me verify: 1/a = 1/b + 1/c. Since 1/b, 1/c > 0,
    # 1/a > 1/b => a < b, and 1/a > 1/c => a < c. So yes, a is strictly the smallest.
    # So the triple is always (a, b, c) with a < b < c and a = bc/(b+c).
    
    # For fast checking, let's build a data structure: for each pair (b, c) with b < c,
    # if they form a triple, store the required a.
    # Actually, for checking if a set S has a triple, we can iterate over all triples
    # and check if all three elements are in S.
    
    # Better approach for incremental checking: maintain for each element, which triples it's in.
    # But for simplicity, let's use a set-based check that's reasonably fast.
    
    # Precompute: for each a, the set of (b, c) pairs that form a triple with a.
    # triple_map[a] = set of (b, c)
    triple_map = {}
    for a, b, c in triples:
        if a not in triple_map:
            triple_map[a] = set()
        triple_map[a].add((b, c))
    
    # For fast membership checking, we'll use a set.
    # The check: does S contain any triple?
    # For each a in S, for each (b, c) in triple_map.get(a, []), check if b in S and c in S.
    
    def has_triple(S):
        """Check if set S contains any forbidden triple."""
        for a in S:
            if a in triple_map:
                for (b, c) in triple_map[a]:
                    if b in S and c in S:
                        return True
        return False
    
    # Greedy construction: start with all numbers, remove those that cause conflicts.
    # Alternative: start empty, add numbers in some order, only add if no triple is formed.
    
    # Strategy: Start with the largest numbers (they're safest since a must be smallest).
    # Then try to add smaller numbers.
    
    # Let's try a greedy approach: process numbers in descending order.
    # For each number x, add it if it doesn't create a triple.
    
    def greedy_descending():
        S = set()
        for x in range(N, 0, -1):
            # Check if adding x creates a triple
            # x could be the "a" (smallest) in a triple, or the "b" or "c"
            # If x is "a": check if any (b, c) in triple_map[x] are both in S
            # If x is "b": check if any a such that (x, c) in triple_map[a] for some c in S
            # If x is "c": check if any a such that (b, x) in triple_map[a] for some b in S
            
            # For simplicity, just do the full check
            trial = S | {x}
            if not has_triple(trial):
                S = trial
        return S
    
    best = greedy_descending()
    best_size = len(best)
    
    # Try greedy with different orderings
    def greedy_order(order):
        S = set()
        for x in order:
            trial = S | {x}
            if not has_triple(trial):
                S = trial
        return S
    
    # Try ascending
    cand = greedy_order(list(range(1, N + 1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(best)
    
    # Try random orderings
    for _ in range(200):
        if time.time() >= deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = greedy_order(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(best)
    
    # Local search: try to add elements that are not in best
    for _ in range(2000):
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            trial = best | {x}
            if not has_triple(trial):
                best = trial
                best_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Swap search: remove one element, add another
    for _ in range(50000):
        if time.time() >= deadline:
            break
        # Pick a random element to remove
        if not best:
            break
        y = rng.choice(sorted(best))
        best_without_y = best - {y}
        # Try to add a random element
        x = rng.randint(1, N)
        if x in best_without_y:
            continue
        trial = best_without_y | {x}
        if not has_triple(trial):
            # Same size, but maybe allows future improvements
            # Only accept if it leads to a larger set later? For now, accept same-size swaps
            # to explore the space
            best = trial
    
    # Final local search after swaps
    for _ in range(2000):
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            trial = best | {x}
            if not has_triple(trial):
                best = trial
                best_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Verify
    if has_triple(best):
        # Fallback
        best = set(range(91, N + 1))
    
    return list(best)

# EVOLVE-BLOCK-END
