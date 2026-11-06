import time
import random
import sys

N = 512

# Precompute conflicts
# conflicts[x] is a list of tuples (u, v, w) such that {x, u, v, w} is a 4-AP
# We need to check if u, v, w are all in S.
_conflicts = [[] for _ in range(N + 1)]

for a in range(1, N - 2):
    for d in range(1, (N - a) // 3 + 1):
        e = a + 3 * d
        if e > N:
            break
        b = a + d
        c = a + 2 * d
        ap = (a, b, c, e)
        
        # For each element in the AP, the other three form a conflict triple
        for i in range(4):
            x = ap[i]
            others = tuple(ap[j] for j in range(4) if j != i)
            _conflicts[x].append(others)

def _can_add(x, S):
    """Check if x can be added to set S without creating a 4-AP."""
    for u, v, w in _conflicts[x]:
        if u in S and v in S and w in S:
            return False
    return True

def _greedy(order, S=None):
    """Greedy construction given an order of elements."""
    if S is None:
        S = set()
    for x in order:
        if x not in S and _can_add(x, S):
            S.add(x)
    return S

def _local_search(S, deadline, rng):
    """
    Local search using Set representation.
    Attempts to add elements, then performs swaps to escape local optima.
    """
    current = set(S)
    current_size = len(current)
    all_elems = list(range(1, N + 1))
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any element
        # Shuffle to avoid bias, but checking all is fast enough if we break early on improvement?
        # No, we should try to add as many as possible.
        # A single pass might miss some if adding one blocks another, but greedy forward/reverse handled that.
        # Here we just do a sweep.
        for x in all_elems:
            if x not in current and _can_add(x, current):
                current.add(x)
                current_size += 1
                improved = True
        
        if improved:
            continue
        
        # 2. If no additions, try swaps
        # Pick a random element not in current
        # To be efficient, we can maintain a list of candidates, but simple random choice is okay for N=512
        # We try a limited number of swap attempts per pass to keep it fast
        attempts = 0
        max_attempts = 200
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            x = rng.choice(all_elems)
            if x in current:
                continue
            
            # Check if x is blocked
            blocking_elements = set()
            is_blocked = False
            for u, v, w in _conflicts[x]:
                if u in current and v in current and w in current:
                    is_blocked = True
                    blocking_elements.update([u, v, w])
            
            if not is_blocked:
                # Should have been added in step 1, but maybe order mattered?
                # Actually, if it's not blocked, we can add it.
                current.add(x)
                current_size += 1
                improved = True
                break # Restart loop to sweep again
            
            if not blocking_elements:
                continue
            
            # Try to remove one of the blocking elements and add x
            # We pick a random blocking element to remove
            rem = rng.choice(list(blocking_elements))
            
            # Check if removing rem allows adding x
            # We don't actually remove yet, we just check logic
            # If we remove rem, does x still have a conflict?
            # We need to ensure that for ALL conflicts of x, at least one element is removed.
            # Since we only remove ONE element, we need that element to be in EVERY conflict triple?
            # No, that's too strong.
            # Actually, if x is blocked by multiple triples, removing one element might not unblock it if another triple is still fully present.
            # So we must check: if we remove `rem`, is x addable?
            
            # Simulate removal
            # To avoid modifying set, we can just check:
            # For each conflict triple of x, is it still fully in (current - {rem})?
            # i.e., for all (u,v,w) in conflicts[x], NOT (u,v,w all in current AND rem not in {u,v,w})?
            # Simpler:
            # Create a temporary set? No, too slow.
            # Check:
            valid_swap = True
            for u, v, w in _conflicts[x]:
                # If the triple is currently fully in current, AND rem is NOT in this triple,
                # then the triple remains fully in current after removing rem.
                # So x would still be blocked.
                if u in current and v in current and w in current:
                    if rem not in (u, v, w):
                        valid_swap = False
                        break
            
            if valid_swap:
                # Perform swap
                current.discard(rem)
                current.add(x)
                # Size stays same. 
                # The task is to MAXIMIZE cardinality. 
                # A swap with same size doesn't help immediately unless it enables future additions.
                # However, standard local search for max independent set usually accepts same-size moves to escape local optima.
                # But we only have 60s. 
                # Let's only accept if it leads to an immediate addition?
                # Or just accept and hope?
                # Given the tight time, let's only accept if we can immediately add another element?
                # That's complex.
                
                # Alternative: Just accept the swap. It changes the structure.
                # But we need to ensure we don't oscillate.
                # Let's just do it.
                improved = True
                break # Restart sweep to see if new element can be added
        
        if not improved:
            # Stuck. Try a "kick": remove a few random elements and re-greedy
            if current_size > 5:
                for _ in range(3):
                    elem = rng.choice(list(current))
                    current.discard(elem)
                current_size = len(current)
                # Re-greedy fill
                for x in all_elems:
                    if x not in current and _can_add(x, current):
                        current.add(x)
                current_size = len(current)
                if current_size > len(S): # Only keep if better than original? No, local search keeps best found so far.
                    pass # We return current at the end, caller compares.
                    
    return current

def construct():
    start_time = time.time()
    # Safety margin: assume 60s limit, stop at 55s
    deadline = start_time + 55.0
    
    rng = random.Random(42)
    
    # 1. Generate multiple seeds
    seeds = []
    
    # Forward greedy
    seeds.append(_greedy(range(1, N + 1)))
    
    # Reverse greedy
    seeds.append(_greedy(range(N, 0, -1)))
    
    # Random greedy (a few times)
    for _ in range(5):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        seeds.append(_greedy(order))
        
    # Find best seed
    best = max(seeds, key=len)
    best_size = len(best)
    
    # 2. Local search on best seed
    if time.time() < deadline - 10:
        improved = _local_search(best, deadline - 5, rng)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
            
    # 3. Randomized restarts with local search
    # We have limited time. Let's do a few more restarts if time permits.
    restarts = 0
    while time.time() < deadline - 15 and restarts < 10:
        restarts += 1
        order = list(range(1, N + 1))
        rng.shuffle(order)
        seed = _greedy(order)
        
        if len(seed) < best_size:
            # Maybe perturb best instead?
            # Remove a random element from best, re-greedy
            if best_size > 1:
                elem = rng.choice(list(best))
                perturbed = best - {elem}
                seed = _greedy(order, perturbed)
                
        if len(seed) > best_size:
            best = seed
            best_size = len(best)
            
        # Local search on this new seed
        if time.time() < deadline - 5:
            improved = _local_search(seed, deadline - 2, rng)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
                
    # 4. Final local search on the global best
    if time.time() < deadline - 1:
        improved = _local_search(best, deadline, rng)
        if len(improved) > best_size:
            best = improved
            
    # 5. Verify (optional, but good for safety)
    # Given the _can_add logic is robust, verification is O(N * avg_conflicts)
    # which is fast.
    for x in best:
        for u, v, w in _conflicts[x]:
            if u in best and v in best and w in best:
                # Invalid! Should not happen.
                # Fallback to a simple valid set
                return list(_greedy(range(1, N + 1)))
                
    return sorted(list(best))

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
