"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin for 90s limit

    n = 128
    numbers = list(range(1, n + 1))

    # 1. Precompute LCM table
    def _lcm(a, b):
        return a * b // gcd(a, b)

    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples.
    # A triple (a,b,c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c).
    # Precompute for each element x, the list of (u, v) pairs such that {x, u, v} is a bad triple.
    # This allows O(1) or O(k) check for validity of adding x to a set S.
    forbidden_pairs = {i: [] for i in range(1, n + 1)}
    # Also, for efficiency in checking, we can store a bitmask of pairs?
    # No, the number of pairs is small. List of tuples is fine.
    
    # To speed up "is x valid in S", we need to check if any (u,v) in forbidden_pairs[x] is fully in S.
    # Represent S as a bitmask.
    # can_add(x, mask):
    #   for u, v in forbidden_pairs[x]:
    #     if (mask >> (u-1)) & 1 and (mask >> (v-1)) & 1: return False
    #   return True

    # Build forbidden_pairs
    # Iterate over all triples a < b < c
    # If bad, add (b,c) to forbidden[a], (a,c) to forbidden[b], (a,b) to forbidden[c]
    
    bad_triples_count = 0
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples_count += 1
                    forbidden_pairs[a].append((b, c))
                    forbidden_pairs[b].append((a, c))
                    forbidden_pairs[c].append((a, b))

    # 3. Strategy: Maximum Independent Set on Hypergraph (order 3).
    # Greedy lower bound first.
    
    def can_add(x, mask):
        for u, v in forbidden_pairs[x]:
            if ((mask >> (u - 1)) & 1) and ((mask >> (v - 1)) & 1):
                return False
        return True

    # Greedy initialization
    # Order by degree (number of forbidden pairs) ascending
    deg = {i: len(forbidden_pairs[i]) for i in numbers}
    greedy_order = sorted(numbers, key=lambda x: deg[x])
    
    current_mask = 0
    current_set = set()
    for x in greedy_order:
        if can_add(x, current_mask):
            current_mask |= (1 << (x - 1))
            current_set.add(x)
            
    best_size = len(current_set)
    best_mask = current_mask

    # Local Search: Hill Climbing with 1-swap and 1-add moves
    # We maintain a valid set S.
    # Moves:
    # 1. Add x if valid. (Improvement if size increases)
    # 2. Swap x in S with y not in S if (S - {x}) U {y} is valid. (Neutral move, might lead to improvement)
    #    To check validity of (S - {x}) U {y}, we only need to check if y is valid in (S - {x}).
    #    Because removing x can only relax constraints for other elements, it cannot create new bad triples 
    #    that didn't exist before (involving x). Wait, if we remove x, we don't create new triples.
    #    So we just check: is y valid in S \ {x}?
    #    i.e. for all (u,v) in forbidden_pairs[y], not both u and v are in (S \ {x}).
    
    # Let's try a more powerful move: "Remove 2, Add 2" or just "Perturb and Refill".
    # Given the time limit, a robust approach is:
    # 1. Start with greedy.
    # 2. Loop:
    #    a. Try all possible single additions. If any, add the one that seems best (or just one).
    #    b. If no addition possible, try all single swaps.
    #       For each x in S, for each y not in S:
    #          Check if y is valid in S \ {x}.
    #          If yes, perform swap. Then try to add more.
    #    c. If no improvement after many swaps, do a random perturbation (remove k random elements) and greedy refill.
    
    # Optimization: Precompute "can_add" check for speed?
    # The inner loop of can_add is O(deg(x)).
    
    def try_additions(mask, s_set):
        added = False
        # Sort candidates by potential? Just iterate greedy_order
        for x in greedy_order:
            if x in s_set:
                continue
            if can_add(x, mask):
                mask |= (1 << (x - 1))
                s_set.add(x)
                added = True
        return mask, s_set, added

    # Local Search Loop
    improved = True
    stall_count = 0
    
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Phase 1: Greedy Add
        current_mask, current_set, added = try_additions(current_mask, current_set)
        if added:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_mask = current_mask
            improved = True
            stall_count = 0
            continue # Restart loop to try more adds
        
        # If no adds, we are at a local maximum for single additions.
        # Phase 2: Try Swaps
        # To make this efficient, we can limit the number of swaps we try per iteration.
        # Or try all swaps and see if any leads to a better state (which it can't directly, 
        # unless it enables a future add).
        # Actually, a swap keeps size same. So we need to check if after a swap, an add becomes possible.
        # This is expensive: O(|S| * (N-|S|)) checks, each O(deg(y)).
        
        # Alternative: Randomized Local Search
        # Perform a random swap, then try to add. If size doesn't increase after trying adds, revert?
        # No, we want to explore.
        
        # Let's try "Random Swap and Fill":
        # 1. Pick random x in S.
        # 2. Pick random y not in S such that y is valid in S \ {x}.
        # 3. Perform swap.
        # 4. Try greedy adds.
        # 5. If size > previous best, keep.
        # 6. If size < previous best (shouldn't happen if we start from valid), revert.
        
        # To avoid getting stuck, we also need "Remove and Refill".
        
        # Let's combine:
        # 1. Try a few random valid swaps.
        # 2. If no improvement in size after a few swaps, remove a random element and greedy refill.
        
        # Implementation of valid swap check:
        # valid_swap(x, y, mask, s_set):
        #   # x in S, y not in S
        #   # Check if y is valid in S \ {x}
        #   # S \ {x} mask = mask & ~(1 << (x-1))
        #   new_mask = mask & ~(1 << (x - 1))
        #   return can_add(y, new_mask)
        
        # We will try up to 50 random swaps.
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Pick random x from current_set
            x = random.choice(list(current_set))
            
            # Find valid y not in current_set
            # We can iterate through candidates not in set
            valid_y_found = False
            # To speed up, we might just check all not-in-set elements?
            # N is 128, so at most 128 checks.
            for y in greedy_order:
                if y in current_set:
                    continue
                # Check if swap x->y is valid
                new_mask = current_mask & ~(1 << (x - 1))
                if can_add(y, new_mask):
                    # Perform swap
                    current_mask = new_mask | (1 << (y - 1))
                    current_set.remove(x)
                    current_set.add(y)
                    valid_y_found = True
                    break
            
            if not valid_y_found:
                break # No valid swap for this x? Try next x? Or give up on swaps?
            
            # After swap, try to add
            current_mask, current_set, added = try_additions(current_mask, current_set)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_mask = current_mask
                improved = True
                stall_count = 0
                break # Break out of swap loop, restart main loop
            
            # If no improvement, we might want to revert?
            # No, let's keep the swap and try more. 
            # But if we keep swapping without improvement, we might wander.
            # Let's limit consecutive non-improving swaps.
            if not added:
                stall_count += 1
                if stall_count > 20:
                    break # Too many stalls, try perturbation
            else:
                stall_count = 0
        
        # If still no improvement, do Perturbation
        if not improved:
            # Remove 2-3 random elements and greedy refill
            removals = min(3, len(current_set) // 4 + 1)
            to_remove = random.sample(list(current_set), removals)
            for x in to_remove:
                current_mask &= ~(1 << (x - 1))
                current_set.remove(x)
            
            current_mask, current_set, added = try_additions(current_mask, current_set)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_mask = current_mask
                improved = True
            
            if time.time() > deadline - 5:
                break

    # Final Greedy Add
    current_mask, current_set, _ = try_additions(current_mask, current_set)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_mask = current_mask

    # Extract result
    result = [i + 1 for i in range(n) if (best_mask >> i) & 1]
    
    # Sanity Check
    # Verify no bad triples
    is_set = set(result)
    valid = True
    for a in result:
        for b in result:
            if a >= b: continue
            for c in result:
                if b >= c: continue
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to a simple greedy if something went wrong
        # This should not happen, but just in case
        current_mask = 0
        current_set = set()
        for x in greedy_order:
            if can_add(x, current_mask):
                current_mask |= (1 << (x - 1))
                current_set.add(x)
        result = sorted(current_set)

    return result

# EVOLVE-BLOCK-END
