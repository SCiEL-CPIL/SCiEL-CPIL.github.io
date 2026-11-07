"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid(S):
    """Check that no three distinct a,b,c in S have all three pair sums in S."""
    s = set(S)
    for i, a in enumerate(S):
        for j in range(i+1, len(S)):
            b = S[j]
            ab = a + b
            if ab not in s:
                continue
            for k in range(j+1, len(S)):
                c = S[k]
                if a + c in s and b + c in s:
                    return False
    return True

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    best = list(range(17, 33)) + list(range(65, 129))  # seed: 16 + 64 = 80
    best_size = len(best)

    def try_add(S_set, x):
        """Check if x can be added to S_set without creating a violation.
        Returns True if valid, False otherwise."""
        if x in S_set:
            return False
        # Adding x creates a bad triple iff there exist distinct a,b in S_set with
        # a+b in S_set, a+x in S_set, and b+x in S_set.
        # For each a in S_set where a+x in S_set, check if there exists b in S_set
        # such that b+x in S_set and a+b in S_set.
        # b+x in S_set means b in {y-x : y in S_set}
        # a+b in S_set means b in {y-a : y in S_set}
        # So b must be in S_set ∩ (S_set - x) ∩ (S_set - a)
        # Precompute S_set - x
        S_minus_x = {y - x for y in S_set if y - x >= 1}
        for a in S_set:
            if a + x in S_set:
                S_minus_a = {y - a for y in S_set if y - a >= 1}
                if S_set & S_minus_x & S_minus_a:
                    return False
        return True

    def greedy_add(S, S_set):
        """Greedily add all possible elements."""
        for x in range(1, 129):
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if try_add(S_set, x):
                S.append(x)
                S_set.add(x)
        return S, S_set

    # Phase 1: Greedy add from seed
    S, S_set = greedy_add(list(best), set(best))
    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Phase 2: Multi-pair swap local search
    # Remove 2-3 elements, then greedily add back as many as possible
    for iteration in range(5000):
        if time.time() > deadline:
            break
        S = list(best)
        S_set = set(S)
        n = len(S)
        if n < 3:
            break
        
        # Choose 1-3 elements to remove
        num_remove = random.choice([1, 2, 2, 3])
        indices = random.sample(range(n), min(num_remove, n))
        to_remove = set(S[i] for i in indices)
        
        S2 = [x for x in S if x not in to_remove]
        S2_set = set(S2)
        
        # Now greedily add back elements (including the removed ones and new ones)
        candidates = list(to_remove) + [x for x in range(1, 129) if x not in S2_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S2_set:
                continue
            if try_add(S2_set, x):
                S2.append(x)
                S2_set.add(x)
        
        if len(S2) > best_size:
            best = S2
            best_size = len(S2)

    # Phase 3: Exhaustive final polish - try adding each missing element
    S = list(best)
    S_set = set(S)
    for x in range(1, 129):
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if try_add(S_set, x):
            S.append(x)
            S_set.add(x)

    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Phase 4: Another round of multi-pair swap with smaller removals
    for iteration in range(3000):
        if time.time() > deadline:
            break
        S = list(best)
        S_set = set(S)
        n = len(S)
        if n < 2:
            break
        
        num_remove = random.choice([1, 1, 2])
        indices = random.sample(range(n), min(num_remove, n))
        to_remove = set(S[i] for i in indices)
        
        S2 = [x for x in S if x not in to_remove]
        S2_set = set(S2)
        
        candidates = list(to_remove) + [x for x in range(1, 129) if x not in S2_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S2_set:
                continue
            if try_add(S2_set, x):
                S2.append(x)
                S2_set.add(x)
        
        if len(S2) > best_size:
            best = S2
            best_size = len(S2)

    # Final verification
    if not _is_valid(best):
        best = list(range(17, 33)) + list(range(65, 129))

    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
