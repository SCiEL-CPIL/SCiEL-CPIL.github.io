import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(n_max):
    """Precompute LCM table for all pairs in [1, n_max]."""
    n = n_max
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _build_conflict_table(lcm_table, n_max):
    """
    Build a conflict structure: for each element x, and each pair (a,b) of other elements,
    determine if {x,a,b} is a bad triple.
    
    We store for each x a set of (a,b) pairs that form a bad triple with x.
    But that's too big. Instead, for each x, we store the set of pairs (a,b) with a<b
    such that lcm(a,b) == lcm(a,x) == lcm(b,x).
    
    Actually, for greedy insertion: when we want to add x to current set S,
    we need to check if there exist a,b in S with a<b such that
    lcm(a,b) == lcm(a,x) == lcm(b,x).
    
    So for each x, precompute: bad_pairs_with_x = set of (a,b) pairs (a<b, a!=x, b!=x)
    such that lcm(a,b) == lcm(a,x) == lcm(b,x).
    
    Then when adding x to S, we check if any (a,b) in bad_pairs_with_x has both a and b in S.
    """
    n = n_max
    # For each x, list of (a,b) pairs that form bad triple with x
    bad_pairs = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            # Find all c such that lcm(a,c) == lab and lcm(b,c) == lab
            # c must satisfy: lcm(a,c) = lab and lcm(b,c) = lab
            # This means c divides lab, and c has the "right" prime power factors.
            # More precisely: lcm(a,c) = lab means c | lab and for each prime p,
            # max(v_p(a), v_p(c)) = v_p(lab).
            # Similarly for b.
            # So c must divide lab, and for each prime p:
            #   max(v_p(a), v_p(c)) = v_p(lab)
            #   max(v_p(b), v_p(c)) = v_p(lab)
            # This implies v_p(c) must be exactly v_p(lab) if v_p(a) < v_p(lab) or v_p(b) < v_p(lab).
            # If v_p(a) = v_p(lab) and v_p(b) = v_p(lab), then v_p(c) can be anything <= v_p(lab).
            
            # Simpler: just iterate over all c from 1 to n, c != a, c != b
            # and check.
            # This is O(n^3) which is 128^3 ~ 2M, fine.
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    # (a,b,c) is a bad triple
                    # For element a: pair (b,c) is bad
                    # For element b: pair (a,c) is bad
                    # For element c: pair (a,b) is bad
                    bc = (min(b,c), max(b,c))
                    ac = (min(a,c), max(a,c))
                    ab = (a, b)
                    bad_pairs[a].add(bc)
                    bad_pairs[b].add(ac)
                    bad_pairs[c].add(ab)
    
    return bad_pairs

def _greedy_construct_fast(order, bad_pairs, n_max):
    """
    Greedily add numbers in the given order if they don't form a bad triple.
    Uses precomputed bad_pairs for fast checking.
    """
    selected = set()
    selected_list = []
    
    for x in order:
        # Check if x can be added: no pair (a,b) in bad_pairs[x] has both a and b in selected
        ok = True
        for (a, b) in bad_pairs[x]:
            if a in selected and b in selected:
                ok = False
                break
        if ok:
            selected.add(x)
            selected_list.append(x)
    
    return selected_list

def _local_search_improve(subset, bad_pairs, n_max, time_limit=10.0):
    """
    Try to improve the subset by swapping elements or adding/removing.
    Simple local search: try removing one element and adding another.
    """
    start_time = time.time()
    current = set(subset)
    current_list = list(subset)
    all_numbers = set(range(1, n_max + 1))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try each element not in current
        for x in all_numbers - current:
            # Check if x can be added
            can_add = True
            for (a, b) in bad_pairs[x]:
                if a in current and b in current:
                    can_add = False
                    break
            if can_add:
                current.add(x)
                current_list = list(current)
                improved = True
                break
        
        # If we added something, also try to see if we can add more
        # Actually, let's do a more thorough pass
        if improved:
            # Try adding more elements
            for x in list(all_numbers - current):
                can_add = True
                for (a, b) in bad_pairs[x]:
                    if a in current and b in current:
                        can_add = False
                        break
                if can_add:
                    current.add(x)
                    current_list = list(current)
        
        # Try swaps: remove y from current, add x not in current
        if not improved:
            for x in all_numbers - current:
                if time.time() - start_time > time_limit:
                    break
                # Check if x can be added if we remove some y
                # For each y in current, check if removing y allows adding x
                for y in list(current):
                    # Check if x and y together cause issues
                    # After removing y, check if x can be added to current - {y}
                    new_current = current - {y}
                    can_add = True
                    for (a, b) in bad_pairs[x]:
                        if a in new_current and b in new_current:
                            can_add = False
                            break
                    if can_add:
                        # Also need to make sure that removing y and adding x doesn't create new bad triples
                        # Actually, since we're only adding x, and x was checked against new_current,
                        # the only new triples involving x are with pairs in new_current, which we checked.
                        # But we also need to ensure that the existing triples in new_current are still valid.
                        # They were valid before, and we only removed y, so they remain valid.
                        current = new_current
                        current.add(x)
                        current_list = list(current)
                        improved = True
                        break
                if improved:
                    break
    
    return current_list

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n_max = 128
    numbers = list(range(1, n_max + 1))
    
    # Precompute LCM table
    lcm_table = _precompute_lcms(n_max)
    
    # Build conflict table
    bad_pairs = _build_conflict_table(lcm_table, n_max)
    
    # Strategy 1: Primes and Prime Powers first
    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    # Order 1: Primes, Prime Powers, Others
    order1 = primes + prime_powers + others
    result1 = _greedy_construct_fast(order1, bad_pairs, n_max)
    
    # Order 2: Standard 1..128
    order2 = numbers
    result2 = _greedy_construct_fast(order2, bad_pairs, n_max)
    
    # Order 3: Sorted by number of divisors (ascending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order3 = sorted(numbers, key=num_divisors)
    result3 = _greedy_construct_fast(order3, bad_pairs, n_max)
    
    # Order 4: Sorted by number of bad pairs (ascending) - elements with fewer conflicts first
    order4 = sorted(numbers, key=lambda x: len(bad_pairs[x]))
    result4 = _greedy_construct_fast(order4, bad_pairs, n_max)
    
    best_result = max([result1, result2, result3, result4], key=len)
    
    # Try random shuffles
    random.seed(42)
    for i in range(200):
        if time.time() > deadline:
            break
        shuffled = numbers[:]
        random.shuffle(shuffled)
        res = _greedy_construct_fast(shuffled, bad_pairs, n_max)
        if len(res) > len(best_result):
            best_result = res
    
    # Local search improvement
    if time.time() < deadline - 15:
        improved = _local_search_improve(best_result, bad_pairs, n_max, time_limit=15.0)
        if len(improved) > len(best_result):
            best_result = improved
    
    # Final verification
    def _is_valid_subset(subset):
        n = len(subset)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = subset[i], subset[j], subset[k]
                    if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                        return False
        return True
    
    if not _is_valid_subset(best_result):
        # Fallback to a known valid result
        best_result = result1
    
    return best_result
