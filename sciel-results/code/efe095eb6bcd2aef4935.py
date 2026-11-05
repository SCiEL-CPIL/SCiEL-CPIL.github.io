"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _unsafe_pairs_for_k(k):
    """Return set of pairs (b, c) with b < c, b,c in [1,180], such that
    a = bc/(b+c) is an integer in [1,180], a != b, a != c, and a < k.
    Actually we need a in the set. We'll precompute for each k which (b,c) are bad."""
    pairs = set()
    for b in range(1, 181):
        for c in range(b + 1, 181):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= 180 and a != b and a != c:
                    pairs.add((a, b, c))
    return pairs

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Precompute all reciprocal triples (a, b, c) with a < b < c (since a is smallest)
    # where 1/a = 1/b + 1/c, i.e., bc = a(b+c)
    # We need a, b, c all distinct and in [1, 180]
    
    # For efficiency, precompute for each pair (b, c) with b < c, the value a = bc/(b+c)
    # if it's an integer in [1, 180] and distinct from b, c
    triple_map = {}  # (b, c) -> a
    for b in range(1, 181):
        for c in range(b + 1, 181):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= 180 and a != b and a != c:
                    triple_map[(b, c)] = a
    
    # For each element x, precompute which other elements form a triple
    # If x = a (smallest): need pairs (b, c) in triple_map with triple_map[(b,c)] == x
    # If x = b or c: need to check all pairs
    
    # Strategy: Greedy + Local Search with multi-restart
    # Key insight: The constraint is that no three distinct elements form a reciprocal triple.
    
    def is_valid(subset_set):
        """Check if subset has no reciprocal triple."""
        for (b, c), a in triple_map.items():
            if a in subset_set and b in subset_set and c in subset_set:
                return False
        return True
    
    def count_violations(subset_set):
        count = 0
        for (b, c), a in triple_map.items():
            if a in subset_set and b in subset_set and c in subset_set:
                count += 1
        return count
    
    best_set = None
    best_size = 0
    
    # Start with greedy: add elements in order of "safety" (fewest conflicts)
    # Precompute conflict count for each element
    conflict_count = [0] * 181
    for (b, c), a in triple_map.items():
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1
    
    # Sort by conflict count (ascending) - add low-conflict elements first
    order = sorted(range(1, 181), key=lambda x: conflict_count[x])
    
    # Greedy construction
    greedy_set = set()
    for x in order:
        trial = greedy_set | {x}
        if is_valid(trial):
            greedy_set = trial
    best_set = set(greedy_set)
    best_size = len(greedy_set)
    
    # Local search: try to add elements that are currently excluded
    def local_search(current_set, max_iter=5000):
        current = set(current_set)
        current_size = len(current)
        excluded = set(range(1, 181)) - current
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return current, current_size
            
            # Try adding a random excluded element
            if excluded:
                candidates = list(excluded)
                random.shuffle(candidates)
                for x in candidates[:10]:
                    trial = current | {x}
                    if is_valid(trial):
                        current = trial
                        current_size += 1
                        excluded.remove(x)
                        break
            
            # Try swapping: remove one, add another (if improves size or same size)
            if not excluded:
                break
            # Try swap: remove y from current, add x from excluded
            if excluded and current:
                remove_candidates = list(current)
                random.shuffle(remove_candidates)
                add_candidates = list(excluded)
                random.shuffle(add_candidates)
                for x in add_candidates[:5]:
                    found_swap = False
                    for y in remove_candidates[:5]:
                        if x == y:
                            continue
                        trial = (current - {y}) | {x}
                        if is_valid(trial):
                            current = trial
                            excluded.remove(x)
                            excluded.add(y)
                            found_swap = True
                            break
                    if found_swap:
                        break
        
        return current, len(current)
    
    # Run local search from greedy
    improved_set, improved_size = local_search(greedy_set)
    if improved_size > best_size:
        best_set = improved_set
        best_size = improved_size
    
    # Multi-restart with different initializations
    for restart in range(10):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Already did greedy
            continue
        
        # Different starting points
        if restart == 1:
            # Start with upper half
            init_set = set(range(91, 181))
        elif restart == 2:
            # Start with even numbers
            init_set = set(range(2, 181, 2))
        elif restart == 3:
            # Start with numbers > 60
            init_set = set(range(61, 181))
        elif restart == 4:
            # Random subset of size ~90
            init_set = set(random.sample(range(1, 181), 90))
        else:
            # Random subset with varying size
            size = random.randint(60, 110)
            init_set = set(random.sample(range(1, 181), size))
        
        # First, clean up violations
        while not is_valid(init_set):
            # Find a violation and remove the element with most conflicts
            found = False
            for (b, c), a in triple_map.items():
                if a in init_set and b in init_set and c in init_set:
                    # Remove the one with highest conflict count
                    triple_members = [a, b, c]
                    max_conf = max(triple_members, key=lambda x: conflict_count[x])
                    init_set.remove(max_conf)
                    found = True
                    break
            if not found:
                break
        
        # Local search
        ls_set, ls_size = local_search(init_set, max_iter=3000)
        if ls_size > best_size:
            best_set = ls_set
            best_size = ls_size
    
    return list(best_set)

# EVOLVE-BLOCK-END
