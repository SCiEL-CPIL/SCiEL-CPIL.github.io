"""Rewritten program: fast bitmask verification + simulated annealing for sorting networks."""
import time
import random
import sys

N = 17
# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 85.0  # safety margin under 90s kill
_TIME_PER_CHECK = 0.00001  # rough estimate; we just count checks

def _build_initial_masks(n):
    """S[i] has bit k set iff input k has bit i = 1 (k ranges 0..2^n-1)."""
    masks = [0] * n
    limit = 1 << n
    for i in range(n):
        m = 0
        # bit k of m is 1 iff (k >> i) & 1
        # This is a periodic pattern: for each i, bits k where k has bit i set.
        # We can build it faster using pattern repetition.
        # Pattern period is 2^(i+1): first 2^i bits are 0, next 2^i bits are 1.
        # But for n=17, limit = 131072, loop is fine.
        for k in range(limit):
            if (k >> i) & 1:
                m |= (1 << k)
        masks[i] = m
    return masks

_INITIAL_MASKS = _build_initial_masks(N)
_N_MASK = (1 << N) - 1  # not directly used but handy

def _check(net):
    """Return True if net sorts all 2^N boolean inputs (zero-one principle)."""
    state = _INITIAL_MASKS[:]
    for i, j in net:
        a = state[i]
        b = state[j]
        mi = a & b
        mj = a | b
        state[i] = mi
        state[j] = mj
    # sorted iff for all k: state[k] & ~state[k+1] == 0
    for k in range(N - 1):
        if state[k] & ~state[k + 1]:
            return False
    return True

def _odd_even_transposition():
    """Standard odd-even transposition (bubble sort) network. 136 comparators."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _greedy_delete(net):
    """Repeatedly try to remove each comparator; keep if still valid. Returns (net, improved)."""
    improved = True
    while improved:
        improved = False
        n = len(net)
        for idx in range(n):
            # skip if time is tight
            if time.time() > _DEADLINE - 5.0:
                return net, improved
            removed = net[idx]
            net.pop(idx)
            if _check(net):
                improved = True
                n -= 1
            else:
                net.insert(idx, removed)
    return net, improved

def _random_comparator():
    i = random.randrange(N)
    j = random.randrange(N)
    if i == j:
        j = (j + 1) % N
    if i > j:
        i, j = j, i
    return (i, j)

def _sa_search(start_net, time_budget):
    """Simulated annealing to minimize comparator count."""
    net = start_net[:]
    best_net = net[:]
    best_len = len(net)
    
    # First, greedy delete to get a good starting point
    net, _ = _greedy_delete(net)
    if len(net) < best_len:
        best_len = len(net)
        best_net = net[:]
    
    if not _check(net):
        # Should not happen, but safety
        net = best_net[:]
        best_len = len(net)
    
    T = 5.0  # initial temperature
    T_min = 0.01
    cooling = 0.995
    
    checks = 0
    max_checks = 50000  # safety cap
    
    while time.time() < time_budget and T > T_min and checks < max_checks:
        # Choose a random move
        move_type = random.random()
        n = len(net)
        
        if move_type < 0.30 and n > 0:
            # Delete a random comparator
            idx = random.randrange(n)
            removed = net.pop(idx)
            if _check(net):
                # Accept (improvement)
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = net[:]
            else:
                # Reject
                net.insert(idx, removed)
            checks += 1
            
        elif move_type < 0.55 and n > 0:
            # Swap two random comparators
            if n >= 2:
                i = random.randrange(n)
                j = random.randrange(n)
                if i != j:
                    net[i], net[j] = net[j], net[i]
                    if _check(net):
                        if len(net) < best_len:
                            best_len = len(net)
                            best_net = net[:]
                    else:
                        net[i], net[j] = net[j], net[i]
                    checks += 1
                    
        elif move_type < 0.75 and n > 0:
            # Move a comparator to a random position
            i = random.randrange(n)
            j = random.randrange(n)
            if i != j:
                comp = net.pop(i)
                # Adjust j if needed
                if i < j:
                    j -= 1
                net.insert(j, comp)
                if _check(net):
                    if len(net) < best_len:
                        best_len = len(net)
                        best_net = net[:]
                else:
                    # Revert
                    net.pop(j)
                    net.insert(i, comp)
                checks += 1
                
        elif move_type < 0.90:
            # Insert a random comparator
            comp = _random_comparator()
            idx = random.randrange(n + 1)
            net.insert(idx, comp)
            if _check(net):
                # It's valid but longer; accept with probability
                delta = len(net) - best_len
                if delta < 0:
                    # Improvement
                    best_len = len(net)
                    best_net = net[:]
                else:
                    # Worse; accept with probability exp(-delta/T)
                    prob = pow(2.718281828, -delta / T)
                    if random.random() > prob:
                        net.pop(idx)
            else:
                net.pop(idx)
            checks += 1
            
        else:
            # Replace a random comparator with a new random one
            if n > 0:
                idx = random.randrange(n)
                old = net[idx]
                new = _random_comparator()
                net[idx] = new
                if _check(net):
                    if len(net) < best_len:
                        best_len = len(net)
                        best_net = net[:]
                else:
                    # Accept with probability based on "cost" (just invalidity)
                    prob = pow(2.718281828, -1.0 / T)
                    if random.random() > prob:
                        net[idx] = old
                checks += 1
        
        T *= cooling
    
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s margin under 90s kill
    
    # Build initial network
    net = _odd_even_transposition()
    
    # Verify it's valid (should be)
    if not _check(net):
        # Fallback: this shouldn't happen
        return net
    
    best_net = net[:]
    best_len = len(net)
    
    # Phase 1: Greedy deletion
    if time.time() < deadline - 10.0:
        net, _ = _greedy_delete(net)
        if len(net) < best_len and _check(net):
            best_len = len(net)
            best_net = net[:]
    
    # Phase 2: Simulated annealing with restarts
    # Do multiple SA runs, each starting from the current best
    while time.time() < deadline - 5.0:
        # Start SA from best_net
        sa_start = time.time()
        sa_budget = min(deadline - 5.0, sa_start + 30.0)
        result = _sa_search(best_net, sa_budget)
        
        if len(result) < best_len and _check(result):
            best_len = len(result)
            best_net = result[:]
        
        # If we've hit a good number, do more aggressive search
        if best_len <= 80 and time.time() < deadline - 5.0:
            # Continue with SA from this new best
            continue
        elif time.time() < deadline - 10.0:
            # Try a different random seed / restart
            # Perturb the best network slightly and run SA again
            perturbed = best_net[:]
            for _ in range(min(5, len(perturbed))):
                if perturbed:
                    idx = random.randrange(len(perturbed))
                    # Randomly modify
                    r = random.random()
                    if r < 0.33 and len(perturbed) > 1:
                        # Delete
                        perturbed.pop(idx)
                    elif r < 0.66:
                        # Insert
                        perturbed.insert(idx, _random_comparator())
                    else:
                        # Replace
                        perturbed[idx] = _random_comparator()
            if _check(perturbed):
                sa_budget2 = min(deadline - 5.0, time.time() + 25.0)
                result2 = _sa_search(perturbed, sa_budget2)
                if len(result2) < best_len and _check(result2):
                    best_len = len(result2)
                    best_net = result2[:]
        else:
            break
    
    # Final verification
    if _check(best_net):
        return best_net
    else:
        # Should not happen; return the original
        return _odd_even_transposition()

# EVOLVE-BLOCK-END
