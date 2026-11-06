"""Improved program: constraint-aware greedy with bitmask conflict precomputation and multi-start shake-ups."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

N = 128  # universe is [1, 128]

def _precompute():
    """Precompute gcd table and conflict masks.
    
    conflict_mask[a] is a bitmask of elements b such that there EXISTS a pair (x,y) 
    in the universe where gcd(x,y)=gcd(x,a)=gcd(y,a) AND x,y are both in some potential set.
    
    Actually, the conflict is: adding 'a' to set S is invalid iff there exist x,y in S 
    with gcd(x,y)==gcd(x,a)==gcd(y,a). This depends on S, so we can't precompute a static mask.
    
    Instead, we precompute: for each pair (x,y), what value g=gcd(x,y) do they share?
    Then for a given 'a', the conflicting pairs are those (x,y) where gcd(x,a)==gcd(y,a)==gcd(x,y).
    
    We precompute for each a and each g: the set of elements x with gcd(x,a)==g.
    Then a pair (x,y) conflicts with a iff gcd(x,y)==g where g=gcd(x,a)=gcd(y,a), i.e.,
    x,y both in {z: gcd(z,a)==g} AND gcd(x,y)==g.
    """
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            g = gcd(a, b)
            gcd_table[a][b] = g
            gcd_table[b][a] = g
    
    # For each element a, for each possible gcd value g, store list of elements x with gcd(x,a)==g
    # gcd(a,a) = a, so g ranges from 1 to a
    gcd_groups = {}  # gcd_groups[a][g] = list of x in [1,N] with gcd(x,a)==g
    for a in range(1, N + 1):
        gcd_groups[a] = {}
        for x in range(1, N + 1):
            g = gcd_table[a][x]
            if g not in gcd_groups[a]:
                gcd_groups[a][g] = []
            gcd_groups[a][g].append(x)
    
    return gcd_table, gcd_groups

def _can_add(a, selected_mask, gcd_table):
    """Check if element a can be added to the set represented by selected_mask.
    
    A conflict exists iff there are x,y in selected with gcd(x,y)==gcd(x,a)==gcd(y,a).
    For each x in selected, compute g_xa = gcd(x,a). Then we need to check if there's y in selected
    (y != x) with gcd(x,y)==g_xa AND gcd(y,a)==g_xa.
    """
    # For each x in selected, g_xa = gcd(x,a). We need y in selected with gcd(x,y)==g_xa and gcd(y,a)==g_xa.
    # Equivalently: y is in selected, y != x, gcd(x,y)==g_xa, and gcd(y,a)==g_xa.
    # Note: gcd(y,a)==g_xa means y is in gcd_groups[a][g_xa].
    # So: for each x in selected, g_xa = gcd(x,a). Check if there exists y in selected 
    #     with y != x, gcd(x,y)==g_xa, and gcd(y,a)==g_xa.
    
    sel_list = [i for i in range(1, N+1) if (selected_mask >> (i-1)) & 1]
    if len(sel_list) < 2:
        return True
    
    for x in sel_list:
        g_xa = gcd_table[x][a]
        # Need y in sel_list, y != x, gcd(x,y)==g_xa, gcd(y,a)==g_xa
        for y in sel_list:
            if y == x:
                continue
            if gcd_table[x][y] == g_xa and gcd_table[y][a] == g_xa:
                return False
    return True

def _can_add_fast(a, selected_mask, gcd_table, gcd_groups):
    """Faster check using precomputed groups.
    
    For element a to conflict with set S, there must exist x,y in S (x!=y) such that
    gcd(x,y) == gcd(x,a) == gcd(y,a) = some g.
    
    This means x and y are both in gcd_groups[a][g] for some g, AND gcd(x,y)==g.
    So: for each g that appears as gcd(x,a) for some x in S, check if the subset of S 
    that's in gcd_groups[a][g] contains a pair (x,y) with gcd(x,y)==g.
    """
    # Collect which gcd values g appear: g = gcd(x,a) for x in S
    # For each such g, the candidates are S ∩ gcd_groups[a][g]
    # We need to check if any such candidate set has a pair with gcd==g
    
    # Build the set of (g, candidate_list) pairs
    g_to_candidates = {}
    for x in range(1, N+1):
        if (selected_mask >> (x-1)) & 1:
            g = gcd_table[x][a]
            if g not in g_to_candidates:
                g_to_candidates[g] = []
            g_to_candidates[g].append(x)
    
    for g, cands in g_to_candidates.items():
        if len(cands) < 2:
            continue
        # Check if any pair in cands has gcd == g
        for i in range(len(cands)):
            for j in range(i+1, len(cands)):
                if gcd_table[cands[i]][cands[j]] == g:
                    return False
    return True

def _greedy_priority(gcd_table, gcd_groups, priority_fn, deadline):
    """Greedy construction where at each step we pick the element with highest priority
    among those that can be added."""
    selected_mask = 0
    selected_count = 0
    remaining = set(range(1, N+1))
    
    while remaining and time.time() < deadline:
        # Find all addable elements
        addable = []
        for a in remaining:
            if _can_add_fast(a, selected_mask, gcd_table, gcd_groups):
                addable.append(a)
        
        if not addable:
            break
        
        # Pick the one with highest priority
        best_a = max(addable, key=priority_fn)
        selected_mask |= (1 << (best_a - 1))
        selected_count += 1
        remaining.discard(best_a)
    
    return selected_mask, selected_count

def _shake_up(selected_mask, num_remove, gcd_table, gcd_groups, deadline):
    """Remove num_remove random elements and try to re-add."""
    sel_list = [i for i in range(1, N+1) if (selected_mask >> (i-1)) & 1]
    if len(sel_list) <= num_remove:
        return selected_mask
    
    to_remove = random.sample(sel_list, num_remove)
    new_mask = selected_mask
    for x in to_remove:
        new_mask &= ~(1 << (x-1))
    
    # Try to add back removed elements and others
    remaining = set(range(1, N+1)) - set(sel_list)
    remaining.update(to_remove)
    
    # Greedy add
    while remaining and time.time() < deadline:
        addable = [a for a in remaining if _can_add_fast(a, new_mask, gcd_table, gcd_groups)]
        if not addable:
            break
        best_a = random.choice(addable)
        new_mask |= (1 << (best_a - 1))
        remaining.discard(best_a)
    
    return new_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    gcd_table, gcd_groups = _precompute()
    
    best_mask = 0
    best_count = 0
    
    # Priority functions for greedy
    def priority_degree(a):
        """Prefer elements with fewer potential conflicts (lower degree in conflict graph)."""
        # Count how many pairs (x,y) in [1,N] would conflict with a
        # A pair (x,y) conflicts with a if gcd(x,y)==gcd(x,a)==gcd(y,a)
        count = 0
        for x in range(1, N+1):
            g_xa = gcd_table[x][a]
            for y in range(x+1, N+1):
                if gcd_table[x][y] == g_xa and gcd_table[y][a] == g_xa:
                    count += 1
        return -count  # Lower conflict count = higher priority
    
    def priority_prime(a):
        """Prefer primes and 1 (they have gcd 1 with most things, reducing equal-gcd triples)."""
        if a == 1:
            return 1000
        # Check if prime
        if a < 2:
            return 1000
        is_p = True
        for p in range(2, int(a**0.5)+1):
            if a % p == 0:
                is_p = False
                break
        if is_p:
            return 500
        return 0
    
    def priority_random(a):
        return random.random()
    
    # Strategy 1: Greedy with different priorities, multiple restarts
    strategies = [priority_degree, priority_prime, priority_random]
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        strat = strategies[trial % len(strategies)]
        
        # For random priority, re-seed each trial
        if trial % 3 == 2:
            random.seed(trial * 7919)
        
        mask, count = _greedy_priority(gcd_table, gcd_groups, strat, deadline)
        
        if count > best_count:
            best_count = count
            best_mask = mask
        
        # Shake-up improvement
        if count >= best_count - 2 and time.time() < deadline - 10:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                new_mask = _shake_up(mask, 2, gcd_table, gcd_groups, deadline)
                new_count = bin(new_mask).count('1')
                if new_count > best_count:
                    best_count = new_count
                    best_mask = new_mask
    
    # Final: try to add more to best_mask
    if best_mask and time.time() < deadline - 2:
        remaining = set(range(1, N+1)) - set(i for i in range(1,N+1) if (best_mask >> (i-1)) & 1)
        while remaining and time.time() < deadline - 1:
            addable = [a for a in remaining if _can_add_fast(a, best_mask, gcd_table, gcd_groups)]
            if not addable:
                break
            best_a = addable[0]  # deterministic
            best_mask |= (1 << (best_a - 1))
            best_count += 1
            remaining.discard(best_a)
    
    # Convert mask to list
    result = [i for i in range(1, N+1) if (best_mask >> (i-1)) & 1]
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
