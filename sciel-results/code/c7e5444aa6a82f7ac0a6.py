"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

N = 192
MAX_SIZE = 48

# Precompute pair sums for fast conflict checking
# pair_sums[a][b][c] = (b+c) % a  -- but this is too large
# Instead, precompute for each a, the set of (b+c) values mod a that appear
# and which pairs produce them.

# Better approach: Precompute for each a, a dictionary mapping remainder r to list of (b,c) pairs
# where (b+c) % a == r. This allows O(1) lookup when checking if x+b+c is divisible by a.

def _precompute_pair_remainders():
    """
    For each a in 1..192, compute:
    pair_rem[a][r] = list of (b,c) pairs with b<c, b!=a, c!=a, (b+c)%a == r
    """
    pair_rem = {}
    for a in range(1, N+1):
        rem_dict = {}
        for b in range(1, N+1):
            if b == a:
                continue
            for c in range(b+1, N+1):
                if c == a:
                    continue
                r = (b + c) % a
                if r not in rem_dict:
                    rem_dict[r] = []
                rem_dict[r].append((b, c))
        pair_rem[a] = rem_dict
    return pair_rem

# Precompute once at module level
_PAIR_REM = _precompute_pair_remainders()

def _can_add_fast(x, current_list, current_set):
    """
    Check if x can be added to current_list (valid set) without violating constraints.
    Uses precomputed pair remainders for speed.
    """
    n = len(current_list)
    
    # Check 1: Does x divide sum of any 3 elements in current_list?
    # i.e., exists (a,b,c) in current_list with (a+b+c) % x == 0
    if n >= 3:
        # For each pair (b,c) in current_list, check if there exists a in current_list
        # such that (a + b + c) % x == 0, i.e., a % x == (-b-c) % x
        # Precompute remainders of current elements mod x
        x_rems = {}
        for a in current_list:
            r = a % x
            if r not in x_rems:
                x_rems[r] = []
            x_rems[r].append(a)
        
        # For each pair (b,c), need a such that a % x == (-b-c) % x
        # But we also need a != b and a != c
        # To keep it simple, let's just do direct triple check for small n
        # For larger n, use the remainder approach
        if n <= 20:
            for i in range(n-2):
                a = current_list[i]
                for j in range(i+1, n-1):
                    b = current_list[j]
                    for k in range(j+1, n):
                        c = current_list[k]
                        if (a + b + c) % x == 0:
                            return False
        else:
            # Use remainder counting
            # Count occurrences of each remainder mod x
            rem_count = {}
            for a in current_list:
                r = a % x
                rem_count[r] = rem_count.get(r, 0) + 1
            
            # For each pair (b,c), target_rem = (-b-c) % x
            # Need at least one a with a % x == target_rem, and a != b, a != c
            # Since b and c are specific values, we need to be careful
            # Simplification: if rem_count[target_rem] > 0, and it's not the case that
            # the only element with that remainder is b or c, then conflict exists
            # This is an approximation that may miss some cases but is fast
            # For correctness, let's do a more careful check
            for i in range(n-1):
                b = current_list[i]
                for j in range(i+1, n):
                    c = current_list[j]
                    target = (-b - c) % x
                    if target in rem_count:
                        # Check if there's an a != b, a != c with a % x == target
                        # We know rem_count[target] >= 1
                        # If rem_count[target] >= 2, or the single element is not b or c
                        # This is getting complex; let's just do direct check for correctness
                        # Actually, for n up to ~50, direct O(n^3) might be too slow
                        # Let's use a different approach: precompute triple sums mod x
                        pass
            
            # Fallback: direct check (may be slow but correct)
            for i in range(n-2):
                a = current_list[i]
                for j in range(i+1, n-1):
                    b = current_list[j]
                    for k in range(j+1, n):
                        c = current_list[k]
                        if (a + b + c) % x == 0:
                            return False
    
    # Check 2: For each a in current_list, does a divide (x + b + c) for any b,c in current_list\{a}?
    if n >= 2:
        for idx, a in enumerate(current_list):
            # Need b,c in current_list\{a} such that (x + b + c) % a == 0
            # i.e., (b + c) % a == (-x) % a
            target_rem = (-x) % a
            if target_rem in _PAIR_REM[a]:
                # Check if any pair (b,c) in _PAIR_REM[a][target_rem] is both in current_list and not equal to a
                for (b, c) in _PAIR_REM[a][target_rem]:
                    if b == a or c == a:
                        continue
                    if b in current_set and c in current_set:
                        # Both b and c are in current_list and not equal to a (since we skip if b==a or c==a)
                        # But we need b != a and c != a, which is already checked
                        # Also need b != c, which is guaranteed by b < c in precomputation
                        return False
    
    return True

def _check_validity(selected):
    """Check if no element divides the sum of any 3 distinct other selected elements."""
    n = len(selected)
    if n < 4:
        return True
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            if (t[0] + t[1] + t[2]) % a == 0:
                return False
    return True

def _greedy_construct(start_order, deadline):
    """Greedy construction: add elements in given order if valid."""
    selected = []
    selected_set = set()
    for x in start_order:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        if _can_add_fast(x, selected, selected_set):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, deadline):
    """Improve set by trying to add elements or swap elements to allow more additions."""
    best = list(selected)
    best_set = set(best)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any element
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if x in best_set:
                continue
            if _can_add_fast(x, best, best_set):
                best.append(x)
                best_set.add(x)
                improved = True
                break
        
        if time.time() > deadline:
            break
        
        # Try swaps: remove one element, try to add elements
        if len(best) > 0 and random.random() < 0.4:
            # Pick a random element to remove
            idx = random.randint(0, len(best) - 1)
            temp_best = [e for i, e in enumerate(best) if i != idx]
            temp_set = set(temp_best)
            
            # Try to add elements to temp_best
            for x in range(N, 0, -1):
                if time.time() > deadline:
                    break
                if x in temp_set:
                    continue
                if _can_add_fast(x, temp_best, temp_set):
                    temp_best.append(x)
                    temp_set.add(x)
            
            if len(temp_best) > len(best):
                best = temp_best
                best_set = temp_set
                improved = True
                break
    
    return best

def _simulated_annealing(selected, deadline):
    """
    Simulated annealing: accept worse solutions with decreasing probability
    to escape local optima.
    """
    best = list(selected)
    best_set = set(best)
    current = list(best)
    current_set = set(current)
    
    start_temp = 1.0
    end_temp = 0.01
    cooling_rate = 0.95
    
    t = start_temp
    step = 0
    while time.time() < deadline and t > end_temp:
        step += 1
        
        # Generate a neighbor: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random element
            candidates = [x for x in range(1, N+1) if x not in current_set]
            if not candidates:
                t *= cooling_rate
                continue
            x = random.choice(candidates)
            new_current = current + [x]
            new_set = current_set | {x}
            # Check validity
            if _check_validity(new_current):
                new_score = len(new_current)
                old_score = len(current)
                delta = new_score - old_score
                if delta >= 0 or random.random() < np.exp(delta / t):
                    current = new_current
                    current_set = new_set
                    if new_score > len(best):
                        best = new_current
                        best_set = new_set
            else:
                # Invalid, penalize
                delta = -1
                if random.random() < np.exp(delta / t):
                    current = new_current
                    current_set = new_set
        
        elif move_type < 0.7:
            # Try to remove a random element
            if not current:
                t *= cooling_rate
                continue
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            new_current = [e for i, e in enumerate(current) if i != idx]
            new_set = current_set - {removed}
            new_score = len(new_current)
            old_score = len(current)
            delta = new_score - old_score
            if delta >= 0 or random.random() < np.exp(delta / t):
                current = new_current
                current_set = new_set
        
        else:
            # Try a swap: remove one, add one
            if not current:
                t *= cooling_rate
                continue
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            candidates = [x for x in range(1, N+1) if x not in current_set]
            if not candidates:
                t *= cooling_rate
                continue
            added = random.choice(candidates)
            new_current = [e for i, e in enumerate(current) if i != idx] + [added]
            new_set = current_set - {removed} | {added}
            if _check_validity(new_current):
                new_score = len(new_current)
                old_score = len(current)
                delta = new_score - old_score
                if delta >= 0 or random.random() < np.exp(delta / t):
                    current = new_current
                    current_set = new_set
                    if new_score > len(best):
                        best = new_current
                        best_set = new_set
        
        t *= cooling_rate
    
    return best

def construct():
    start_time = time.time()
    # Safety margin: return before 90s wall-clock limit
    deadline = start_time + 85.0
    
    # Phase 1: Multiple greedy starts
    best = []
    
    # Start 1: Descending order
    order1 = list(range(N, 0, -1))
    cand1 = _greedy_construct(order1, deadline)
    if len(cand1) > len(best):
        best = cand1
    
    # Start 2: Primes first
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(N, 0, -1) if is_prime(x)]
    non_primes = [x for x in range(N, 0, -1) if not is_prime(x)]
    order2 = primes + non_primes
    cand2 = _greedy_construct(order2, deadline)
    if len(cand2) > len(best):
        best = cand2
    
    # Start 3: Randomized
    random.seed(42)
    for trial in range(15):
        if time.time() > deadline - 10.0:
            break
        order = list(range(N, 0, -1))
        random.shuffle(order)
        cand = _greedy_construct(order, deadline)
        if len(cand) > len(best):
            best = cand
    
    # Phase 2: Local improvement
    if time.time() < deadline - 10.0:
        improved = _local_search(best, deadline - 5.0)
        if len(improved) > len(best):
            best = improved
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5.0:
        sa_result = _simulated_annealing(best, deadline - 2.0)
        if len(sa_result) > len(best):
            best = sa_result
    
    # Final validation and cleanup
    if len(best) == 0:
        best = [N]
    
    # Ensure distinct positive integers <= N
    best = list(set(best))
    best = [x for x in best if 1 <= x <= N]
    best.sort(reverse=True)
    
    # Enforce max size
    if len(best) > MAX_SIZE:
        best = best[:MAX_SIZE]
    
    # Final validity check
    if not _check_validity(best):
        # Fallback to a known valid small set
        best = [N, N-1, N-2, N-3]
        # Verify this is valid
        if not _check_validity(best):
            best = [N]
    
    return best
# EVOLVE-BLOCK-END
