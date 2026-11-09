import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_conflicts(n_max=180):
    """
    For each a in [1, n_max], find all pairs (b, c) with b < c, b != a, c != a
    such that 1/a = 1/b + 1/c.
    This is equivalent to: b*c = a*(b+c)
    => b*c - a*b - a*c = 0
    => (b-a)*(c-a) = a^2
    
    So for each divisor d of a^2, we get b = a + d, c = a + a^2/d.
    We need b != c, i.e., d != a^2/d, i.e., d != a.
    Also b, c must be in [1, n_max].
    """
    conflicts = {a: [] for a in range(1, n_max + 1)}
    for a in range(1, n_max + 1):
        a2 = a * a
        # Find all divisors d of a2 where d != a
        # b = a + d, c = a + a2 // d
        # Need b < c for uniqueness, so d < a2/d, i.e., d < a
        # Or we can just generate all and deduplicate
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                if d == a:
                    continue
                b1 = a + d
                c1 = a + a2 // d
                # Make sure b1 < c1
                if b1 < c1 and b1 <= n_max and c1 <= n_max and b1 >= 1 and c1 >= 1:
                    conflicts[a].append((b1, c1))
                # Check the other way: d' = a2 // d
                d2 = a2 // d
                if d2 == a:
                    continue
                b2 = a + d2
                c2 = a + d
                if b2 < c2 and b2 <= n_max and c2 <= n_max and b2 >= 1 and c2 >= 1:
                    if (b2, c2) not in conflicts[a]:
                        conflicts[a].append((b2, c2))
    return conflicts

def _is_valid(s, conflicts):
    """Check if set s has no three distinct elements a,b,c with 1/a = 1/b + 1/c."""
    s_set = set(s)
    for a in s_set:
        for (b, c) in conflicts.get(a, []):
            if b in s_set and c in s_set:
                return False
    return True

def _greedy_build(conflicts, order):
    """Greedy build: add elements in the given order if they don't create conflicts."""
    s = set()
    for a in order:
        ok = True
        for (b, c) in conflicts.get(a, []):
            if b in s and c in s:
                ok = False
                break
        if ok:
            # Also need to check: adding a might create conflict for some existing element e
            # where (a, b) or (b, a) is a conflicting pair for e
            # This is automatically handled if we check all existing elements... but that's expensive.
            # Actually, the conflict is symmetric in b,c but not in a.
            # 1/e = 1/a + 1/b means {a,b} is a conflicting pair for e.
            # So when adding a, we need to check if there exists e in s and some b in s 
            # such that {a, b} is a conflicting pair for e.
            # Equivalently: for each e in s, is there a b in s such that (min(a,b), max(a,b)) is in conflicts[e]?
            # This is O(|s| * |conflicts[e]|) per insertion.
            # Let's do a full validity check for safety, but that's O(n^2) which is fine for n=180.
            s.add(a)
            if not _is_valid(s, conflicts):
                s.remove(a)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    n_max = 180
    conflicts = _precompute_conflicts(n_max)
    
    # Start with the seed solution
    best_set = set(range(91, 181))
    best_size = len(best_set)
    
    # Try greedy builds with different orderings
    rng = random.Random(42)
    
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        # Create a random ordering
        elements = list(range(1, n_max + 1))
        rng.shuffle(elements)
        
        s = _greedy_build(conflicts, elements)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Try local search: swap elements to improve
    if time.time() < deadline - 15:
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            
            # Pick a random element to remove and try to add a different one
            if not best_set:
                break
            
            to_remove = rng.choice(list(best_set))
            candidate = set(best_set)
            candidate.remove(to_remove)
            
            # Try to add a random element not in candidate
            missing = [x for x in range(1, n_max + 1) if x not in candidate]
            if not missing:
                break
            
            a_to_add = rng.choice(missing)
            candidate.add(a_to_add)
            
            if _is_valid(candidate, conflicts):
                if len(candidate) > best_size:
                    best_size = len(candidate)
                    best_set = candidate
                else:
                    best_set = candidate  # Same size, might help future swaps
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
