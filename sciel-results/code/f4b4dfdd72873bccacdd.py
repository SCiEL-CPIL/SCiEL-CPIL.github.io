"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _is_reciprocal_triple(a, b, c):
    """Check if 1/a = 1/b + 1/c with distinct a,b,c.
    Equivalent to b*c == a*(b+c)."""
    return b * c == a * (b + c)

def _find_triples_in_set(S):
    """Find all reciprocal triples in set S. Returns list of (a,b,c) where a<b<c and 1/a=1/b+1/c."""
    S_set = set(S)
    triples = []
    # a must be smallest: 1/a = 1/b + 1/c => a = bc/(b+c)
    # So for each pair (b,c), check if a = bc/(b+c) is integer and in S
    for b in S:
        for c in S:
            if b >= c:
                continue
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if a != b and a != c and a in S_set and a < b:
                    triples.append((a, b, c))
    return triples

def _count_triples(S):
    """Count reciprocal triples in set S."""
    S_set = set(S)
    count = 0
    for b in S:
        for c in S:
            if b >= c:
                continue
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if a != b and a != c and a in S_set and a < b:
                    count += 1
    return count

def _has_triple(S):
    """Check if S contains any reciprocal triple."""
    S_set = set(S)
    for b in S:
        for c in S:
            if b >= c:
                continue
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if a != b and a != c and a in S_set and a < b:
                    return True
    return False

def _is_valid(S):
    """Check if S has no reciprocal triple."""
    return not _has_triple(S)

def _try_add(S, x):
    """Try adding x to S. Return new set if valid, else None."""
    if x in S:
        return None
    S_new = set(S)
    S_new.add(x)
    if not _has_triple(S_new):
        return S_new
    return None

def _local_search(S, deadline, rng):
    """Local search with swap and kick moves to maximize |S| while keeping valid."""
    S = set(S)
    best = S
    best_size = len(S)
    all_elements = set(range(1, 181))
    
    while time.time() < deadline:
        # Try to add elements not in S
        added = False
        candidates = list(all_elements - S)
        rng.shuffle(candidates)
        for x in candidates:
            S_new = _try_add(S, x)
            if S_new is not None:
                S = S_new
                added = True
                if len(S) > best_size:
                    best_size = len(S)
                    best = set(S)
        if not added:
            # Try swap moves: remove one from S, add two not in S
            swapped = False
            S_list = list(S)
            rng.shuffle(S_list)
            for remove in S_list:
                if time.time() > deadline:
                    break
                S_minus = S - {remove}
                # Try to add two elements
                candidates = list(all_elements - S_minus)
                rng.shuffle(candidates)
                for x in candidates:
                    if time.time() > deadline:
                        break
                    S_trial = set(S_minus)
                    S_trial.add(x)
                    if not _has_triple(S_trial):
                        for y in candidates:
                            if y == x:
                                continue
                            if time.time() > deadline:
                                break
                            S_trial2 = set(S_trial)
                            S_trial2.add(y)
                            if not _has_triple(S_trial2) and len(S_trial2) > len(S):
                                S = S_trial2
                                swapped = True
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best = set(S)
                                break
            # Kick move: random perturbation
            if not swapped:
                # Remove a random element, then try to re-add greedily
                if S:
                    kick_remove = rng.choice(list(S))
                    S_kick = S - {kick_remove}
                    # Greedily re-add
                    candidates = list(all_elements - S_kick)
                    rng.shuffle(candidates)
                    for x in candidates:
                        S_new = _try_add(S_kick, x)
                        if S_new is not None:
                            S_kick = S_new
                    if len(S_kick) > len(S):
                        S = S_kick
                        if len(S) > best_size:
                            best_size = len(S)
                            best = set(S)
    
    return best

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Start with a large initial set
    # The seed uses [91, 180] which has 90 elements
    # Let's start with a greedy construction from small to large
    S = set()
    for x in range(1, 181):
        S_new = _try_add(S, x)
        if S_new is not None:
            S = S_new
    
    # Now do local search to improve
    rng = random.Random(42)
    best = _local_search(S, deadline, rng)
    
    # Try multiple restarts with different initial sets
    for trial in range(20):
        if time.time() > deadline:
            break
        # Different initial strategies
        if trial == 0:
            S_init = set(range(91, 181))
        elif trial == 1:
            S_init = set(range(1, 181))
            # Remove elements to make valid
            S_init = set()
            for x in range(1, 181):
                S_new = _try_add(S_init, x)
                if S_new is not None:
                    S_init = S_new
        elif trial == 2:
            # Random subset
            S_init = set(rng.sample(range(1, 181), 100))
            # Make valid by removing problematic elements
            S_valid = set()
            for x in sorted(S_init):
                S_new = _try_add(S_valid, x)
                if S_new is not None:
                    S_valid = S_new
            S_init = S_valid
        else:
            # Perturb best
            S_init = set(best)
            # Remove random element
            if S_init:
                S_init.discard(rng.choice(list(S_init)))
        
        result = _local_search(S_init, deadline, rng)
        if len(result) > len(best):
            best = result
    
    return list(best)

# EVOLVE-BLOCK-END
