import time
import random
import math
from itertools import combinations

def construct():
    deadline = time.time() + 85
    N = 180
    
    # Precompute all distinct triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    # Let d = b-a. Then c-a = a^2/d.
    # b = a+d, c = a + a^2/d
    # Conditions:
    # 1. d divides a^2
    # 2. d < a (to ensure b < c, since d < a^2/d <=> d^2 < a^2)
    #    If d > a, then c < b, which is same triple.
    #    If d = a, b = c, not distinct.
    # 3. c <= N
    # 4. b, c distinct from a (guaranteed by d > 0)
    # 5. b, c distinct from each other (guaranteed by d != a)
    
    triples = []
    # Map from pair (i, j) to list of k's
    # i < j
    pair_to_k = {}
    
    for a in range(1, N + 1):
        a2 = a * a
        # d must be a divisor of a^2
        # d < a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if c <= N:
                    # We have a triple (a, b, c) with a < b < c
                    # Add to triples
                    triples.append((a, b, c))
                    # Add to pair_to_k
                    pair_to_k.setdefault((a, b), []).append(c)
                    pair_to_k.setdefault((a, c), []).append(b)
                    pair_to_k.setdefault((b, c), []).append(a)
    
    # Helper to check if a set S is valid
    # For efficiency in local search, we maintain S as a set and check conflicts incrementally.
    
    def is_valid_addition(S, x):
        """Check if x can be added to S without creating a violation."""
        # x creates a violation if there exist y, z in S such that {x, y, z} is a triple.
        # Iterate over all y in S.
        # For each y, check if there is a z in S such that {x, y, z} is a triple.
        # We can use pair_to_k.
        # For a fixed x, the possible partners y are such that (min(x,y), max(x,y)) is in pair_to_k.
        # Actually, just iterate y in S.
        for y in S:
            if x == y: continue
            key1 = (x, y) if x < y else (y, x)
            if key1 in pair_to_k:
                for z in pair_to_k[key1]:
                    if z in S and z != x and z != y:
                        return False
        return True

    def get_conflicts(S, x):
        """Return list of pairs (y, z) in S such that {x, y, z} is a violation."""
        conflicts = []
        for y in S:
            if x == y: continue
            key1 = (x, y) if x < y else (y, x)
            if key1 in pair_to_k:
                for z in pair_to_k[key1]:
                    if z in S and z != x and z != y:
                        conflicts.append((y, z))
        return conflicts

    # Local Search Strategy:
    # 1. Start with a greedy solution.
    # 2. Iteratively try to add elements. If blocked, try to resolve by removing 1 element (swap).
    # 3. If no swaps improve or maintain size with better structure, try random kicks (remove 2, add 1) or restart.
    
    best_S = set()
    best_size = 0
    
    def local_search(S, bits, iterations, deadline):
        """
        Perform local search on S.
        bits is a bitset for fast membership.
        """
        # Convert S to a set for iteration
        S = set(S)
        
        for _ in range(iterations):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try adding elements
            # Randomize order to avoid bias
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                
                conflicts = get_conflicts(S, x)
                if not conflicts:
                    S.add(x)
                    improved = True
                    # Continue trying to add more
                else:
                    # Try to resolve by removing 1 element
                    # We need to hit all conflicts (y, z).
                    # A conflict is a pair (y, z). We must remove y or z.
                    # This is a hitting set problem on the set of edges `conflicts`.
                    # Since conflicts are small, we can try removing each candidate removal.
                    # Candidate removals are all y and z in conflicts.
                    removals = set()
                    for y, z in conflicts:
                        removals.add(y)
                        removals.add(z)
                    
                    found_swap = False
                    for r in removals:
                        # Check if removing r hits all conflicts
                        if all(r == y or r == z for y, z in conflicts):
                            # Remove r, add x
                            S.remove(r)
                            S.add(x)
                            improved = True
                            found_swap = True
                            break
                    
                    if found_swap:
                        continue # Try next candidate x
                    
                    # If no single removal works, we could try 2-removal (kick)
                    # But that decreases size. We only do this if we are stuck and want to escape.
                    # For now, stick to size-increasing or size-preserving moves.
            
            if not improved:
                # Try a kick: remove 2 random elements, then try to add 1-3 elements
                # This is a heuristic to escape local optima.
                if len(S) >= 2 and random.random() < 0.1:
                    # Pick 2 elements to remove
                    to_remove = random.sample(list(S), 2)
                    for r in to_remove:
                        S.remove(r)
                    # Try to add back elements
                    candidates = [x for x in range(1, N + 1) if x not in S]
                    random.shuffle(candidates)
                    added = 0
                    for x in candidates:
                        if added >= 2: break # Net change 0
                        if is_valid_addition(S, x):
                            S.add(x)
                            added += 1
    
        return S

    # Multiple restarts with different initializations
    random.seed(42)
    
    init_strategies = [
        lambda: set(range(1, N + 1, 2)), # Odds
        lambda: set(range(1, N + 1, 3)), # Every 3rd
        lambda: set(range(N // 2 + 1, N + 1)), # Upper half
        lambda: set(range(1, N + 1)) - set(range(1, N // 3 + 1)), # Exclude lower third
        lambda: set(random.sample(range(1, N + 1), N // 2)), # Random half
    ]
    
    for strat in init_strategies:
        if time.time() > deadline - 5:
            break
        
        S = strat()
        # Ensure initial S is valid
        # Greedily remove conflicts from S
        # Sort S to have deterministic removal? Or random?
        # Let's greedily fix S:
        # While there is a violation, remove one element.
        # To maximize size, we should remove the element that is in the most violations?
        # Or just iterate and remove if conflict found.
        
        def make_valid(S):
            # Iterate through sorted list for stability
            S_list = sorted(S)
            # We need to check all triples in S
            # Simple approach: repeatedly find a violation and remove one element
            # To be efficient, we can check all pairs
            changed = True
            while changed:
                changed = False
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        a, b = S_list[i], S_list[j]
                        key = (a, b)
                        if key in pair_to_k:
                            for c in pair_to_k[key]:
                                if c in S:
                                    # Violation found: a, b, c
                                    # Remove c (arbitrary choice, or the one with most conflicts)
                                    # Let's remove c
                                    S.remove(c)
                                    changed = True
                                    break
                        if changed:
                            # Re-evaluate S_list? Or just break and restart loop
                            break
                    if changed:
                        S_list = sorted(S)
                        break
            return S

        S = make_valid(S)
        
        # Local search
        S = local_search(S, 0, 20000, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Final polish on best solution
    if best_S and time.time() < deadline - 2:
        S = local_search(best_S, 0, 50000, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    return sorted(best_S)
