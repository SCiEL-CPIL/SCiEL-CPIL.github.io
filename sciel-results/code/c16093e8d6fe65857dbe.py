import time
import random
import math

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _is_dissociated(S):
    """Check if all subset sums of {1/s for s in S} are distinct."""
    if len(S) <= 1:
        return True
    
    # Compute LCM of elements in S
    L = 1
    for s in S:
        L = _lcm(L, s)
        
    # Use exact integer arithmetic with LCM
    # LCM of [1..22] is ~2.3e8, so this is safe and fast
    weights = [L // s for s in S]
    seen = {0}
    sums = [0]
    for w in weights:
        new_sums = [x + w for x in sums]
        for ns in new_sums:
            if ns in seen:
                return False
        seen.update(new_sums)
        sums.extend(new_sums)
    return True

def _potential(S):
    """Count how many elements can be added to S while maintaining dissociation."""
    count = 0
    for e in range(1, 23):
        if e not in S:
            if _is_dissociated(S | {e}):
                count += 1
    return count

def construct():
    start = time.time()
    deadline = start + 55.0
    
    best = []
    best_size = 0
    best_potential = 0
    
    def try_greedy(order):
        current = set()
        for e in order:
            if e in current:
                continue
            new = current | {e}
            if _is_dissociated(new):
                current = new
        return current
    
    # Try various orderings
    all_elements = list(range(1, 23))
    
    # Deterministic orders
    orders = [
        list(range(22, 0, -1)),  # Descending (larger first)
        list(range(1, 23)),       # Ascending
    ]
    
    # Random orders
    rng = random.Random(42)
    for _ in range(1000): # Increased restarts
        if time.time() > deadline - 10.0:
            break
        order = all_elements[:]
        rng.shuffle(order)
        orders.append(order)
        
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        result = try_greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best = list(result)
            best_potential = _potential(set(result))
        elif len(result) == best_size:
            # Keep the one with higher potential if same size
            pot = _potential(set(result))
            if pot > best_potential:
                best_potential = pot
                best = list(result)
    
    # Local search
    if best:
        current = set(best)
        
        # Phase 1: Add any possible element
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for e in range(1, 23):
                if e not in current:
                    new = current | {e}
                    if _is_dissociated(new):
                        current = new
                        best_size = len(current)
                        best = list(current)
                        best_potential = _potential(current)
                        improved = True
                        break # Restart loop as set changed
        
        # Phase 2: 1-Swap to increase potential or size
        # We try swaps to see if we can reach a state with higher potential
        # or larger size (though size is maxed in Phase 1, swaps might allow adding more later)
        if time.time() < deadline - 5.0:
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                current_pot = best_potential
                for remove in list(current):
                    for add in range(1, 23):
                        if add in current:
                            continue
                        new = (current - {remove}) | {add}
                        if _is_dissociated(new):
                            new_pot = _potential(new)
                            # Accept if size increases (unlikely here) or potential increases
                            if len(new) > len(current) or new_pot > current_pot:
                                current = new
                                best_size = len(current)
                                best = list(current)
                                best_potential = new_pot
                                improved = True
                                break
                    if improved:
                        break
                        # After a swap, we should re-run Phase 1 (Add)
                        # So we break out of swap loop and go back to Phase 1?
                        # The structure above has Phase 1 and Phase 2 separate.
                        # Let's combine them into a main loop.

        # Refined Local Search Loop
        if time.time() < deadline - 2.0:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                
                # Try Add
                for e in range(1, 23):
                    if e not in current:
                        new = current | {e}
                        if _is_dissociated(new):
                            current = new
                            best_size = len(current)
                            best = list(current)
                            best_potential = _potential(current)
                            improved = True
                            break
                
                # Try 1-Swap
                if not improved and time.time() < deadline - 2.0:
                    current_pot = best_potential
                    for remove in list(current):
                        for add in range(1, 23):
                            if add in current:
                                continue
                            new = (current - {remove}) | {add}
                            if _is_dissociated(new):
                                new_pot = _potential(new)
                                # We accept a swap if it increases potential, 
                                # hoping it leads to a larger set later.
                                if new_pot > current_pot:
                                    current = new
                                    best_size = len(current)
                                    best = list(current)
                                    best_potential = new_pot
                                    improved = True
                                    break
                        if improved:
                            break

                # Try 2-Swap (if time permits and 1-swap didn't help)
                if not improved and time.time() < deadline - 2.0:
                    current_pot = best_potential
                    # Limit 2-swap search to avoid combinatorial explosion if too slow
                    # 22 elements, choose 2 to remove ~ 231, choose 2 to add ~ 231. Total ~ 53k checks.
                    # Each check is _is_dissociated (2^11 ops) + _potential (21 * 2^11 ops).
                    # This might be too slow. We should only do 2-swap if we are stuck and have time.
                    
                    # Optimization: Only do 2-swap if current_size < 13 (assuming 13 is max)
                    # Or just try a few random 2-swaps.
                    if len(current) < 15: # Arbitrary threshold
                        for i in range(1, 23):
                            if i in current:
                                for j in range(i+1, 23):
                                    if j in current:
                                        rem_set = {i, j}
                                        for k in range(1, 23):
                                            if k not in current:
                                                for l in range(k+1, 23):
                                                    if l not in current:
                                                        add_set = {k, l}
                                                        new = (current - rem_set) | add_set
                                                        if _is_dissociated(new):
                                                            new_pot = _potential(new)
                                                            if new_pot > current_pot:
                                                                current = new
                                                                best_size = len(current)
                                                                best = list(current)
                                                                best_potential = new_pot
                                                                improved = True
                                                                break
                                                    if improved: break
                                            if improved: break
                                    if improved: break
                            if improved: break

    if not best:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best
