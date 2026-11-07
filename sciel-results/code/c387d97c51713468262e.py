import time
import random

N = 128

def _precompute():
    """Precompute sum_pairs and other structures."""
    sum_pairs = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s = a + b
            if s <= N:
                sum_pairs[s].append((a, b))
    return sum_pairs

def _can_add(S_mask, x, sum_pairs):
    """Check if S_mask U {x} is a valid set (no forbidden triple).
    
    A forbidden triple is three distinct members a,b,c where a+b, a+c, b+c are all in the set.
    When we add x, any new violation must involve x.
    
    Case 1: x is a member of the triple. Need distinct b,c in S with x+b, x+c, b+c in S U {x}.
    Case 2: x is a sum a+b of two members. Need distinct a,b,c in S with a+b=x, a+c in S, b+c in S.
    """
    # Case 1: x is a member (a = x)
    # Need distinct b, c in S such that (x+b in S U {x}) AND (x+c in S U {x}) AND (b+c in S U {x})
    # Since x+b > x and x+c > x, they must be in S (not x itself since x+b > x).
    # b+c in S U {x} means b+c in S OR b+c == x.
    
    # V = { y in S | x+y in S }  (y such that x+y is also in S)
    # Using bitmask: S_mask >> x gives bits where (y+x) is in S, i.e., bit y set means x+y in S
    V_mask = S_mask & (S_mask >> x)
    # V_mask has bit y set iff y in S AND x+y in S
    
    if V_mask.bit_count() >= 2:
        # Need to check if there exist distinct b, c in V such that b+c in S OR b+c == x
        v_list = []
        temp = V_mask
        while temp:
            lsb = temp & -temp
            v_list.append(lsb.bit_length() - 1)
            temp -= lsb
        n_v = len(v_list)
        for i in range(n_v):
            b = v_list[i]
            for j in range(i + 1, n_v):
                c = v_list[j]
                s = b + c
                if s == x or ((S_mask >> s) & 1):
                    return False
    
    # Case 2: x is a sum of two members (a+b = x)
    # Need distinct a, b, c in S with a+b=x, a+c in S, b+c in S.
    for a, b in sum_pairs[x]:
        if ((S_mask >> a) & 1) == 0:
            continue
        if ((S_mask >> b) & 1) == 0:
            continue
        # c in S \ {a,b} such that a+c in S and b+c in S
        # U = { c in S | a+c in S AND b+c in S }
        # a+c in S: bit c set in (S_mask >> a)
        # b+c in S: bit c set in (S_mask >> b)
        U_mask = (S_mask & (S_mask >> a)) & (S_mask & (S_mask >> b))
        U_mask &= ~(1 << a)
        U_mask &= ~(1 << b)
        if U_mask != 0:
            return False
    
    return True

def _is_valid(S_mask):
    """Check if S_mask is a valid set."""
    # We can check by trying to add each element to an empty set, but that's slow.
    # Instead, iterate over all triples.
    s = set()
    temp = S_mask
    while temp:
        lsb = temp & -temp
        s.add(lsb.bit_length() - 1)
        temp -= lsb
    
    lst = sorted(s)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i + 1, n):
            b = lst[j]
            for k in range(j + 1, n):
                c = lst[k]
                if (a + b) in s and (a + c) in s and (b + c) in s:
                    return False
    return True

def _greedy_with_order(order, sum_pairs):
    """Build a set by trying elements in given order, skipping if it creates a violation."""
    S_mask = 0
    for x in order:
        if (S_mask >> x) & 1:
            continue
        if _can_add(S_mask, x, sum_pairs):
            S_mask |= (1 << x)
    return S_mask

def _local_search(S_mask, sum_pairs, deadline, rng, max_passes=50, swap_limit=20):
    """Improve S via additions and systematic swaps using fast can_add check."""
    S_set = set()
    temp = S_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        S_set.add(idx)
        temp -= lsb
    
    not_in_S = [x for x in range(1, N + 1) if x not in S_set]
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        rng.shuffle(not_in_S)
        
        for x in not_in_S[:]:
            if time.time() > deadline:
                break
            
            # Try to add x directly
            if _can_add(S_mask, x, sum_pairs):
                S_mask |= (1 << x)
                S_set.add(x)
                not_in_S.remove(x)
                improved = True
                continue
            
            # Try swaps: remove y from S, add x
            S_list = list(S_set)
            rng.shuffle(S_list)
            
            found_swap = False
            for y in S_list[:swap_limit]:
                if time.time() > deadline:
                    break
                S_temp_mask = S_mask ^ (1 << y)
                if _can_add(S_temp_mask, x, sum_pairs):
                    S_mask = S_temp_mask | (1 << x)
                    S_set.remove(y)
                    S_set.add(x)
                    not_in_S.append(y)
                    if x in not_in_S:
                        not_in_S.remove(x)
                    found_swap = True
                    improved = True
                    break
            
            if found_swap:
                break
        
        if not improved:
            break
    
    return S_mask

def _simulated_annealing_fast(S_mask, sum_pairs, deadline, rng):
    """Simulated annealing using fast can_add for incremental checks."""
    S_set = set()
    temp = S_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        S_set.add(idx)
        temp -= lsb
    
    best_mask = S_mask
    best_size = S_mask.bit_count()
    
    temperature = 1.0
    cooling_rate = 0.997
    min_temperature = 0.005
    
    # Track best seen during SA
    best_seen = best_size
    
    while time.time() < deadline and temperature > min_temperature:
        move_type = rng.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_mask = S_mask | (1 << x)
            new_size = new_mask.bit_count()
            
            # Check if new set is valid by verifying x can be added
            valid = _can_add(S_mask, x, sum_pairs)
            # But we need to check the full set, not just incremental
            # Actually, if S_mask is valid and _can_add says x can be added, then new is valid
            if not valid:
                continue
            
            # Accept if larger or with probability
            if new_size > best_size:
                S_mask = new_mask
                S_set.add(x)
                best_mask = new_mask
                best_size = new_size
            elif rng.random() < pow(2.718, 1.0 / temperature):
                S_mask = new_mask
                S_set.add(x)
                
        elif move_type == 'remove':
            if not S_set:
                continue
            x = rng.choice(list(S_set))
            new_mask = S_mask ^ (1 << x)
            new_size = new_mask.bit_count()
            
            # Always valid (removing can't create violations)
            # Accept if we want to explore, or if it helps later
            if rng.random() < pow(2.718, -1.0 / temperature):
                S_mask = new_mask
                S_set.remove(x)
                
        else:  # swap
            if len(S_set) < 1:
                continue
            in_set = list(S_set)
            out_set = [x for x in range(1, N + 1) if x not in S_set]
            if not out_set:
                continue
            x_in = rng.choice(in_set)
            x_out = rng.choice(out_set)
            
            # Try removing x_in first, then adding x_out
            temp_mask = S_mask ^ (1 << x_in)
            if _can_add(temp_mask, x_out, sum_pairs):
                new_mask = temp_mask | (1 << x_out)
                new_size = new_mask.bit_count()
                
                if new_size > best_size:
                    S_mask = new_mask
                    S_set.discard(x_in)
                    S_set.add(x_out)
                    best_mask = new_mask
                    best_size = new_size
                elif rng.random() < pow(2.718, 0.5 / max(temperature, 0.1)):
                    S_mask = new_mask
                    S_set.discard(x_in)
                    S_set.add(x_out)
        
        temperature *= cooling_rate
    
    # Final polish: try to add any remaining elements
    for x in range(1, N + 1):
        if (S_mask >> x) & 1:
            continue
        if _can_add(S_mask, x, sum_pairs):
            S_mask |= (1 << x)
    
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    sum_pairs = _precompute()
    rng = random.Random(42)
    
    # Baseline: {63, ..., 128} has size 66
    best_mask = 0
    for x in range(63, N + 1):
        best_mask |= (1 << x)
    best_size = best_mask.bit_count()
    
    # Strategy 1: Greedy approaches with diverse orderings
    # Try different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(63, N + 1)) + list(range(1, 63)),
        list(range(1, 63)) + list(range(63, N + 1)),
    ]
    
    # Add some random orderings
    for _ in range(30):
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        orderings.append(candidates)
    
    for order in orderings:
        if time.time() > deadline - 20.0:
            break
        S_mask = _greedy_with_order(order, sum_pairs)
        if S_mask.bit_count() > best_size:
            best_size = S_mask.bit_count()
            best_mask = S_mask
    
    # Strategy 2: Local search from best set
    if time.time() < deadline - 10.0:
        best_mask = _local_search(best_mask, sum_pairs, deadline - 10.0, rng, max_passes=50, swap_limit=20)
        best_size = best_mask.bit_count()
    
    # Strategy 3: Simulated annealing from multiple starting points
    sa_deadline = deadline - 5.0
    for trial in range(5):
        if time.time() > sa_deadline:
            break
        
        if trial == 0:
            start_mask = best_mask
        else:
            # Random start: mix of upper and lower
            upper = list(range(63, N + 1))
            lower = list(range(1, 63))
            rng.shuffle(upper)
            rng.shuffle(lower)
            num_upper = rng.randint(30, 60)
            num_lower = rng.randint(0, 20)
            start_set = set(upper[:num_upper] + lower[:num_lower])
            start_mask = 0
            for x in start_set:
                start_mask |= (1 << x)
        
        S_mask = _simulated_annealing_fast(start_mask, sum_pairs, sa_deadline, rng)
        
        # Final polish on the SA result
        for x in range(1, N + 1):
            if (S_mask >> x) & 1:
                continue
            if _can_add(S_mask, x, sum_pairs):
                S_mask |= (1 << x)
        
        if S_mask.bit_count() > best_size:
            best_size = S_mask.bit_count()
            best_mask = S_mask
    
    # Strategy 4: More local search passes on the best set
    if time.time() < deadline - 2.0:
        best_mask = _local_search(best_mask, sum_pairs, deadline - 2.0, rng, max_passes=30, swap_limit=15)
        best_size = best_mask.bit_count()
    
    # Final: try to add any remaining elements
    for x in range(1, N + 1):
        if (best_mask >> x) & 1:
            continue
        if _can_add(best_mask, x, sum_pairs):
            best_mask |= (1 << x)
    
    # Final verification
    if _is_valid(best_mask):
        result = []
        temp = best_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            result.append(int(idx))
            temp -= lsb
        return result
    else:
        # Fallback to known valid set
        return [int(x) for x in range(63, N + 1)]
