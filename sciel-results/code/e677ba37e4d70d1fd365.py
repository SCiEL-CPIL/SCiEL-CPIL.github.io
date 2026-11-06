import math
import random
import time
from fractions import Fraction

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
NUM_PRIMES = len(_PRIMES)
_COSTS = [Fraction(1, p) for p in _PRIMES]
MAX_CHECK = 100000


def _seed_solution():
    """The i-th prime from 13 upward covers the integer i."""
    primes_from_13 = [p for p in _PRIMES if p >= 13]
    assign = [0] * NUM_PRIMES
    for i, p in enumerate(primes_from_13):
        idx = _PRIMES.index(p)
        r = (i + 1) % p
        if r == 0:
            r = 1
        assign[idx] = r
    return assign


def _seed_solution_random():
    """Random initial solution: pick a subset of primes with random residues."""
    assign = [0] * NUM_PRIMES
    cost = Fraction(0)
    num_to_pick = random.randint(40, 60)
    indices = random.sample(range(NUM_PRIMES), min(num_to_pick, NUM_PRIMES))
    for i in indices:
        p = _PRIMES[i]
        r = random.randint(1, p)
        assign[i] = r
        cost += _COSTS[i]
        if cost > Fraction(1, 1):
            assign[i] = 0
            cost -= _COSTS[i]
            break
    return assign


def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _COSTS[idx]
    return c <= Fraction(1, 1)


def _coverage_length(assign):
    """Full recompute of coverage length."""
    covered = bytearray(MAX_CHECK + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            covered[n] = 1
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if covered[n]:
            L = n
        else:
            break
    return L


def _build_cover_state(assign):
    """Build cover_count array from assign. Returns (cover_count, L)."""
    cover_count = [0] * (MAX_CHECK + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            cover_count[n] += 1
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if cover_count[n] > 0:
            L = n
        else:
            break
    return cover_count, L


def _add_class(cover_count, idx, r):
    """Add prime idx with residue r."""
    p = _PRIMES[idx]
    for n in range(r, MAX_CHECK + 1, p):
        cover_count[n] += 1


def _remove_class(cover_count, idx, r):
    """Remove prime idx with residue r."""
    p = _PRIMES[idx]
    for n in range(r, MAX_CHECK + 1, p):
        cover_count[n] -= 1


def _recompute_L(cover_count):
    """Recompute coverage length from cover_count."""
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if cover_count[n] > 0:
            L = n
        else:
            break
    return L


def _find_first_gap(cover_count):
    """Find the first n >= 1 where cover_count[n] == 0. Returns n or -1 if all covered."""
    for n in range(1, MAX_CHECK + 1):
        if cover_count[n] == 0:
            return n
    return -1


def _greedy_construct(time_limit):
    """
    Greedy set cover: iteratively select the prime-residue class that covers
    the most currently uncovered integers, subject to the cost constraint.
    """
    start = time.time()
    assign = [0] * NUM_PRIMES
    cover_count = [0] * (MAX_CHECK + 1)
    current_cost = Fraction(0)
    used_set = set()
    
    # Start with a seed solution
    seed = _seed_solution()
    for idx, r in enumerate(seed):
        if r != 0:
            assign[idx] = r
            current_cost += _COSTS[idx]
            used_set.add(idx)
            _add_class(cover_count, idx, r)
    
    current_L = _recompute_L(cover_count)
    best_assign = assign[:]
    best_L = current_L
    
    max_rounds = 100
    for round_num in range(max_rounds):
        if time.time() - start > time_limit:
            break
        
        gap = _find_first_gap(cover_count)
        if gap == -1:
            break
        
        best_gain = 0
        best_action = None
        
        # Option 1: Add a new prime
        for i in range(NUM_PRIMES):
            if i in used_set:
                continue
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
            
            p = _PRIMES[i]
            r_needed = gap % p
            if r_needed == 0:
                r_needed = p
            
            _add_class(cover_count, i, r_needed)
            new_L = _recompute_L(cover_count)
            _remove_class(cover_count, i, r_needed)
            
            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_action = ('add', i, r_needed)
        
        # Option 2: Change residue of an existing prime
        for i in list(used_set):
            p = _PRIMES[i]
            old_r = assign[i]
            
            r_needed = gap % p
            if r_needed == 0:
                r_needed = p
            
            if r_needed == old_r:
                continue
            
            _remove_class(cover_count, i, old_r)
            _add_class(cover_count, i, r_needed)
            new_L = _recompute_L(cover_count)
            
            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_action = ('change', i, r_needed)
            
            _remove_class(cover_count, i, r_needed)
            _add_class(cover_count, i, old_r)
        
        if best_action is not None and best_gain > 0:
            if best_action[0] == 'add':
                _, i, r = best_action
                _add_class(cover_count, i, r)
                assign[i] = r
                used_set.add(i)
                current_cost += _COSTS[i]
            else:
                _, i, r = best_action
                old_r = assign[i]
                _remove_class(cover_count, i, old_r)
                _add_class(cover_count, i, r)
                assign[i] = r
            
            current_L += best_gain
            if current_L > best_L:
                best_L = current_L
                best_assign = assign[:]
        else:
            # Random perturbation
            if used_set:
                i = random.choice(list(used_set))
                p = _PRIMES[i]
                old_r = assign[i]
                new_r = random.randint(1, p)
                if new_r != old_r:
                    _remove_class(cover_count, i, old_r)
                    _add_class(cover_count, i, new_r)
                    assign[i] = new_r
                    new_L = _recompute_L(cover_count)
                    if new_L >= best_L * 0.8:
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_assign = assign[:]
                    else:
                        _remove_class(cover_count, i, new_r)
                        _add_class(cover_count, i, old_r)
                        assign[i] = old_r
    
    return best_assign, best_L


def _local_search_incremental(assign, time_limit):
    """
    Hill-climbing local search with incremental updates.
    Tries all single-residue changes and accepts if L improves.
    """
    start = time.time()
    current = assign[:]
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    cover_count, current_L = _build_cover_state(current)
    best = current[:]
    best_L = current_L
    
    used_set = set(i for i in range(NUM_PRIMES) if current[i] != 0)
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # Try swapping residues for used primes
        for i in list(used_set):
            if time.time() - start > time_limit:
                break
            p = _PRIMES[i]
            old_r = current[i]
            
            for r in range(1, p + 1):
                if r == old_r:
                    continue
                
                _remove_class(cover_count, i, old_r)
                _add_class(cover_count, i, r)
                new_L = _recompute_L(cover_count)
                
                if new_L > current_L:
                    current[i] = r
                    current_L = new_L
                    improved = True
                    if current_L > best_L:
                        best = current[:]
                        best_L = current_L
                    break
                else:
                    _remove_class(cover_count, i, r)
                    _add_class(cover_count, i, old_r)
            
            if time.time() - start > time_limit:
                break
        
        # Try adding unused primes
        if not improved:
            unused = [i for i in range(NUM_PRIMES) if i not in used_set]
            for i in unused:
                if time.time() - start > time_limit:
                    break
                if current_cost + _COSTS[i] > Fraction(1, 1):
                    continue
                
                p = _PRIMES[i]
                best_r = -1
                best_new_L = current_L
                
                for r in range(1, p + 1):
                    _add_class(cover_count, i, r)
                    new_L = _recompute_L(cover_count)
                    _remove_class(cover_count, i, r)
                    
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_r = r
                
                if best_r != -1:
                    _add_class(cover_count, i, best_r)
                    current[i] = best_r
                    used_set.add(i)
                    current_cost += _COSTS[i]
                    current_L = best_new_L
                    improved = True
                    if current_L > best_L:
                        best = current[:]
                        best_L = current_L
        
        # Try removing primes that don't hurt
        if not improved:
            for i in list(used_set):
                if time.time() - start > time_limit:
                    break
                r = current[i]
                
                _remove_class(cover_count, i, r)
                new_L = _recompute_L(cover_count)
                
                if new_L >= current_L:
                    current[i] = 0
                    used_set.discard(i)
                    current_cost -= _COSTS[i]
                    current_L = new_L
                    improved = True
                else:
                    _add_class(cover_count, i, r)
    
    return best, best_L


def _simulated_annealing_incremental(assign, time_limit):
    """
    Simulated annealing with incremental coverage updates.
    """
    start = time.time()
    current = assign[:]
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    cover_count, current_L = _build_cover_state(current)
    
    best = current[:]
    best_L = current_L
    best_cost = current_cost
    
    T = 50.0
    cooling = 0.9999
    
    used_set = set(i for i in range(NUM_PRIMES) if current[i] != 0)
    
    while time.time() - start < time_limit:
        move_type = random.random()
        
        if move_type < 0.5:
            if not used_set:
                continue
            i = random.choice(list(used_set))
            p = _PRIMES[i]
            old_r = current[i]
            r = random.randint(1, p)
            if r == old_r:
                continue
            
            _remove_class(cover_count, i, old_r)
            _add_class(cover_count, i, r)
            current[i] = r
            
            new_L = _recompute_L(cover_count)
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best = current[:]
                    best_L = current_L
                    best_cost = current_cost
            else:
                _remove_class(cover_count, i, r)
                _add_class(cover_count, i, old_r)
                current[i] = old_r
                
        elif move_type < 0.8:
            unused = [i for i in range(NUM_PRIMES) if i not in used_set]
            if not unused:
                continue
            i = random.choice(unused)
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
            p = _PRIMES[i]
            r = random.randint(1, p)
            
            _add_class(cover_count, i, r)
            current[i] = r
            used_set.add(i)
            old_cost = current_cost
            current_cost += _COSTS[i]
            
            new_L = _recompute_L(cover_count)
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best = current[:]
                    best_L = current_L
                    best_cost = current_cost
            else:
                _remove_class(cover_count, i, r)
                current[i] = 0
                used_set.discard(i)
                current_cost = old_cost
                
        else:
            if not used_set:
                continue
            i = random.choice(list(used_set))
            r = current[i]
            
            _remove_class(cover_count, i, r)
            current[i] = 0
            used_set.discard(i)
            old_cost = current_cost
            current_cost -= _COSTS[i]
            
            new_L = _recompute_L(cover_count)
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best = current[:]
                    best_L = current_L
                    best_cost = current_cost
            else:
                _add_class(cover_count, i, r)
                current[i] = r
                used_set.add(i)
                current_cost = old_cost
        
        T *= cooling
        if T < 0.1:
            break
    
    return best, best_L


def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s kill
    
    # Phase 0: Generate multiple initial solutions
    initial_solutions = [_seed_solution()]
    
    for _ in range(3):
        initial_solutions.append(_seed_solution_random())
    
    best_assign = None
    best_L = -1
    
    for sol in initial_solutions:
        if _is_valid(sol):
            L = _coverage_length(sol)
            if L > best_L:
                best_L = L
                best_assign = sol[:]
    
    if best_assign is None:
        best_assign = _seed_solution()
        best_L = _coverage_length(best_assign)
    
    # Phase 1: Greedy construct
    if time.time() < deadline - 100:
        greedy_assign, greedy_L = _greedy_construct(min(30, deadline - time.time() - 70))
        if greedy_L > best_L and _is_valid(greedy_assign):
            best_assign = greedy_assign
            best_L = greedy_L
    
    # Phase 2: Local search refinement
    if time.time() < deadline - 60:
        ls_assign, ls_L = _local_search_incremental(
            best_assign, min(30, deadline - time.time() - 30)
        )
        if ls_L > best_L and _is_valid(ls_assign):
            best_assign = ls_assign
            best_L = ls_L
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 30:
        sa_assign, sa_L = _simulated_annealing_incremental(
            best_assign, min(40, deadline - time.time() - 15)
        )
        if sa_L > best_L and _is_valid(sa_assign):
            best_assign = sa_assign
            best_L = sa_L
    
    # Phase 4: Final local search
    if time.time() < deadline - 5:
        ls2_assign, ls2_L = _local_search_incremental(
            best_assign, min(10, deadline - time.time() - 3)
        )
        if ls2_L > best_L and _is_valid(ls2_assign):
            best_assign = ls2_assign
            best_L = ls2_L
    
    # Final validation
    if _is_valid(best_assign):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        return _seed_solution()
