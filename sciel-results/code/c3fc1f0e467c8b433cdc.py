import math
import random
import time
import sys
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

# Precompute for each prime index and each possible residue the list of numbers covered
# This is too large (168 primes * up to 1000 residues * up to 100000/2 numbers), so we compute on the fly
# but we can precompute the "step" pattern for speed.

# For faster coverage updates, we'll use a different approach:
# Instead of iterating over all numbers in range(r, MAX_CHECK+1, p), we can use
# a smarter method. But for now, let's keep the incremental approach but optimize it.

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

def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _COSTS[idx]
    return c <= Fraction(1, 1)

def _coverage_length(assign):
    """Full recompute of coverage length (for final validation)."""
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
    # Find L: largest n such that all 1..n have cover_count > 0
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

def _find_first_gap(cover_count, start_from=1):
    """Find the first n >= start_from where cover_count[n] == 0."""
    for n in range(start_from, MAX_CHECK + 1):
        if cover_count[n] == 0:
            return n
    return -1

def _greedy_construct(time_limit):
    """
    Greedy set cover: iteratively select the prime-residue class that covers
    the most currently uncovered integers, subject to the cost constraint.
    This is a different structural approach than local search/SA.
    """
    start = time.time()
    assign = [0] * NUM_PRIMES
    cover_count = [0] * (MAX_CHECK + 1)
    current_cost = Fraction(0)
    used_set = set()
    
    # Start with a seed solution to have a good initial state
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
    
    # Now try to improve by adding new primes or changing residues
    # Greedy: for each unused prime, find the best residue
    # For each used prime, try to find a better residue
    
    # First, try to add primes that cover the most new numbers
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() - start > time_limit:
            break
        
        # Find the first gap
        gap = _find_first_gap(cover_count)
        if gap == -1:
            break
        
        # Try to cover the gap by adding a new prime or changing an existing one
        best_gain = 0
        best_action = None
        
        # Option 1: Add a new prime
        for i in range(NUM_PRIMES):
            if i in used_set:
                continue
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
            
            p = _PRIMES[i]
            # The residue that covers the gap
            r_needed = gap % p
            if r_needed == 0:
                r_needed = p
            
            # How many new numbers would this cover?
            # We need to count how many n in range(r_needed, MAX_CHECK+1, p) have cover_count[n] == 0
            # But we only care about extending L, so let's just try it
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
            
            # Try the residue that covers the gap
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
            
            # Revert
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
            # No improvement possible with single changes
            # Try a random perturbation to escape local minimum
            if time.time() - start > time_limit:
                break
            
            # Randomly change a residue
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
                    # Don't revert; accept even if worse (simulated annealing style)
                    current_L = new_L
            else:
                break
    
    return best_assign, best_L

def _gap_targeted_local_search(assign, time_limit):
    """
    Local search that specifically targets gaps: find the first gap,
    then try all possible single changes that could cover it.
    """
    start = time.time()
    current = assign[:]
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    cover_count, current_L = _build_cover_state(current)
    best = current[:]
    best_L = current_L
    used_set = set(i for i in range(NUM_PRIMES) if current[i] != 0)
    
    max_rounds = 200
    for round_num in range(max_rounds):
        if time.time() - start > time_limit:
            break
        
        gap = _find_first_gap(cover_count)
        if gap == -1:
            break
        
        improved = False
        
        # Try to cover the gap by changing an existing prime's residue
        for i in list(used_set):
            if time.time() - start > time_limit:
                break
            p = _PRIMES[i]
            old_r = current[i]
            
            # The residue that covers the gap
            r_needed = gap % p
            if r_needed == 0:
                r_needed = p
            
            if r_needed == old_r:
                continue
            
            _remove_class(cover_count, i, old_r)
            _add_class(cover_count, i, r_needed)
            current[i] = r_needed
            new_L = _recompute_L(cover_count)
            
            if new_L > current_L:
                current_L = new_L
                improved = True
                if current_L > best_L:
                    best_L = current_L
                    best = current[:]
                break
            else:
                # Revert
                _remove_class(cover_count, i, r_needed)
                _add_class(cover_count, i, old_r)
                current[i] = old_r
        
        if improved:
            continue
        
        # Try to add a new prime that covers the gap
        for i in range(NUM_PRIMES):
            if time.time() - start > time_limit:
                break
            if i in used_set:
                continue
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
            
            p = _PRIMES[i]
            r_needed = gap % p
            if r_needed == 0:
                r_needed = p
            
            _add_class(cover_count, i, r_needed)
            current[i] = r_needed
            used_set.add(i)
            old_cost = current_cost
            current_cost += _COSTS[i]
            new_L = _recompute_L(cover_count)
            
            if new_L > current_L:
                current_L = new_L
                improved = True
                if current_L > best_L:
                    best_L = current_L
                    best = current[:]
                break
            else:
                # Revert
                _remove_class(cover_count, i, r_needed)
                current[i] = 0
                used_set.discard(i)
                current_cost = old_cost
        
        if improved:
            continue
        
        # Try removing a prime that doesn't hurt L
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
                break
            else:
                # Revert
                _add_class(cover_count, i, r)
        
        if not improved:
            # Stuck; try a random perturbation
            if time.time() - start > time_limit:
                break
            
            if used_set:
                i = random.choice(list(used_set))
                p = _PRIMES[i]
                old_r = current[i]
                new_r = random.randint(1, p)
                if new_r != old_r:
                    _remove_class(cover_count, i, old_r)
                    _add_class(cover_count, i, new_r)
                    current[i] = new_r
                    current_L = _recompute_L(cover_count)
    
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
            # Swap residue of a used prime
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
            # Add a prime
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
            # Remove a prime
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

def _local_search_incremental(assign, time_limit):
    """
    Hill-climbing local search with incremental updates.
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

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s kill
    
    # Start with seed solution
    best_assign = _seed_solution()
    best_L = _coverage_length(best_assign)
    
    if not _is_valid(best_assign):
        best_assign = [0] * NUM_PRIMES
        best_L = 0
    
    # Phase 1: Greedy construction (new structural approach)
    if time.time() < deadline - 60:
        greedy_time = min(30, deadline - time.time() - 10)
        greedy_assign, greedy_L = _greedy_construct(greedy_time)
        if greedy_L > best_L and _is_valid(greedy_assign):
            best_assign = greedy_assign
            best_L = greedy_L
    
    # Phase 2: Gap-targeted local search (new refinement)
    if time.time() < deadline - 40:
        gap_time = min(25, deadline - time.time() - 10)
        gap_assign, gap_L = _gap_targeted_local_search(best_assign, gap_time)
        if gap_L > best_L and _is_valid(gap_assign):
            best_assign = gap_assign
            best_L = gap_L
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 20:
        sa_time = min(25, deadline - time.time() - 10)
        sa_assign, sa_L = _simulated_annealing_incremental(best_assign, sa_time)
        if sa_L > best_L and _is_valid(sa_assign):
            best_assign = sa_assign
            best_L = sa_L
    
    # Phase 4: Final local search
    if time.time() < deadline - 5:
        ls_time = min(15, deadline - time.time() - 2)
        ls_assign, ls_L = _local_search_incremental(best_assign, ls_time)
        if ls_L > best_L and _is_valid(ls_assign):
            best_assign = ls_assign
            best_L = ls_L
    
    # Final validation
    if _is_valid(best_assign):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        # Fallback to seed
        return _seed_solution()
