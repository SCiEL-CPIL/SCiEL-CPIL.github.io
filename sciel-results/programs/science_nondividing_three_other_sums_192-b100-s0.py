import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe
    
    N = 192
    all_numbers = set(range(1, N + 1))
    
    # Fast validity check for adding x to set S
    # S is a set of ints
    # Returns True if x can be added to S
    def can_add(x, S):
        # Check 1: For each a in S, does x + y + z == 0 mod a for some y, z in S\{a}?
        # y, z must be distinct and not a.
        # If len(S) < 2, no pairs y, z exist, so no violation for existing a.
        if len(S) >= 2:
            S_list = list(S)
            for a in S_list:
                # We need y, z in S, y != z, y != a, z != a
                # such that (x + y + z) % a == 0
                # => y + z == (-x) % a
                # Let target = (-x) % a
                # We need to find distinct y, z in S\{a} summing to target?
                # Wait, y+z can be target + k*a.
                # Max y+z is (N-1) + (N-2) ~ 380.
                # a is at most 192.
                # So k can be 0 or 1 (maybe 2 if a is small).
                
                # More efficient:
                # Iterate y in S\{a}.
                # For each y, check if (target - y) is in S\{a, y}?
                # No, y+z = target + k*a.
                # So z = target + k*a - y.
                # We can just iterate y in S\{a} and check if any z in S\{a, y} satisfies (x+y+z) % a == 0.
                # This is O(|S|^2) per a.
                
                # Optimization:
                # Create a set of S without a for fast lookup.
                # S_minus_a = S - {a}
                # If len(S_minus_a) < 2, continue.
                
                # Let's just do the double loop, it's fast enough.
                found = False
                # To avoid creating sets repeatedly, we can use a list and check membership
                # But S is small.
                for y in S_list:
                    if y == a:
                        continue
                    for z in S_list:
                        if z == a or z == y:
                            continue
                        if (x + y + z) % a == 0:
                            return False
        else:
            # If S has < 2 elements, we can't form a pair y,z for existing a.
            # But we still need to check if x itself is safe?
            # Check 2 handles x.
            pass
            
        # Check 2: Does x divide sum of any 3 distinct elements in S?
        # Need y, z, w in S distinct such that (y+z+w) % x == 0.
        if len(S) >= 3:
            S_list = list(S)
            # Triple loop O(|S|^3)
            # 48^3 is ~110k. 
            # We can optimize with pair sums.
            # Precompute pair sums? No, S changes.
            # Just do triple loop with early break?
            # Or: for each y, z, check if w exists.
            # w = (k*x - y - z).
            # Iterate y, z in S.
            # Check if w in S and w != y and w != z.
            S_set = S # already a set
            found = False
            # To speed up, iterate y, z
            # We can break early if we find one.
            for i in range(len(S_list)):
                y = S_list[i]
                for j in range(len(S_list)):
                    if j == i:
                        continue
                    z = S_list[j]
                    # We need w such that y+z+w is multiple of x.
                    # y+z+w = k*x => w = k*x - y - z
                    # w must be in S, w != y, w != z.
                    # Max sum is 3*192 = 576.
                    # Min sum is 1+2+3 = 6.
                    # k*x is in [6, 576].
                    # So k is in [ceil(6/x), floor(576/x)].
                    min_k = (6 + x - 1) // x
                    max_k = 576 // x
                    # Actually, we can just check specific w values.
                    # But iterating k and checking w in S is O(range(k)).
                    # range(k) is small (576/x).
                    # So inner loop is O(576/x).
                    # Total O(|S|^2 * 576/x).
                    # For x=1, 576. For x=192, 3.
                    # This is much faster than O(|S|^3) if x is large.
                    # If x is small, 576/x is large, but |S|^2 is small.
                    
                    for k in range(min_k, max_k + 1):
                        w = k * x - y - z
                        if w in S_set and w != y and w != z:
                            return False
                            
        return True

    # Initial Greedy Construction
    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # Generate different orders
    # 1. Descending
    order1 = list(range(N, 0, -1))
    # 2. Ascending
    order2 = list(range(1, N + 1))
    # 3. Random
    rng = random.Random(42)
    order3 = list(range(1, N + 1))
    rng.shuffle(order3)
    
    best_S = set()
    best_size = 0
    
    for order in [order1, order2, order3]:
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    # Local Search (Hill Climbing)
    # Try to swap elements to increase size or maintain size with better properties?
    # Objective: Maximize |S|.
    # Move: Remove one, add one. If size increases, accept.
    # If size stays same, maybe accept with some probability (Simulated Annealing)?
    # Or just try to add elements to a slightly smaller set?
    
    # Let's try a "repair" strategy.
    # Start with a random valid set of size K.
    # Try to add elements.
    # If stuck, remove an element, try to add two?
    
    # Simpler: 
    # While time remains:
    #   1. Try to add any valid element to best_S.
    #   2. If no element can be added, try a swap: remove a, add b, if valid and |S| same, keep it?
    #      Actually, if we remove a and add b, size is same. We want to increase size.
    #      So we should look for a state where we can add.
    #      Maybe remove 2, add 2?
    
    # Let's use a random restart local search.
    
    def local_search(start_S, rng):
        S = start_S.copy()
        current_size = len(S)
        stall = 0
        
        while time.time() < deadline:
            improved = False
            
            # Try to add an element
            # Prioritize larger numbers? Or random?
            # Larger numbers are less likely to be divisors of sums, but more likely to be sums?
            # Actually, if x is large, it's hard for x to divide sum of 3 others (sum < 3x usually).
            # So large x is "safe" as a divisor.
            # But small x is "dangerous" as a divisor.
            # So we should add large numbers first.
            
            candidates = sorted(all_numbers - S, reverse=True)
            rng.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, S):
                    S.add(x)
                    current_size += 1
                    improved = True
                    stall = 0
                    break
            
            if improved:
                continue
            
            # If no single add works, try a swap: remove one, add one.
            # This keeps size constant, but might allow future adds.
            # Or try remove 1, add 2 (size +1).
            
            # Try Remove 1, Add 2
            elems = list(S)
            rng.shuffle(elems)
            
            found_move = False
            for rem in elems:
                # Try to add 2 elements from candidates
                # This is O(|S| * |candidates|^2 * check_cost).
                # Too slow?
                # Let's try Remove 1, Add 1 first to escape local minima.
                # If we remove a "bad" element, we might be able to add 2 "good" ones.
                
                # Let's just do a random swap to explore.
                add_candidates = [x for x in all_numbers if x not in S]
                if not add_candidates:
                    break
                rng.shuffle(add_candidates)
                
                for add in add_candidates[:10]: # Limit tries
                    new_S = (S - {rem}) | {add}
                    # Check if new_S is valid
                    # We can check incrementally?
                    # Removing rem: 
                    # 1. rem was not dividing sums of others (was valid).
                    # 2. Others were not dividing sums involving rem.
                    #    Now rem is gone. Those constraints are gone.
                    #    So removing rem cannot create a violation.
                    # Adding add:
                    # 1. add might divide sum of 3 others in new_S\{add}.
                    # 2. Others might divide sum involving add.
                    # So we only need to check if `add` can be added to `S - {rem}`.
                    if can_add(add, S - {rem}):
                        S = new_S
                        improved = True
                        stall = 0
                        found_move = True
                        break
                if found_move:
                    break
            
            if not improved:
                stall += 1
                if stall > 50:
                    break
                    
        return S

    # Run multiple local searches
    for i in range(50):
        if time.time() > deadline:
            break
        
        # Generate diverse starts
        if i == 0:
            start_S = best_S
        else:
            # Random subset
            k = rng.randint(20, 40)
            start_S = set(rng.sample(list(all_numbers), k))
            # Filter to valid
            valid_start = set()
            for x in sorted(start_S, reverse=True):
                if can_add(x, valid_start):
                    valid_start.add(x)
            start_S = valid_start
            
        S = local_search(start_S, rng)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    # Final polish
    # Try to add any remaining
    for x in sorted(all_numbers - best_S, reverse=True):
        if can_add(x, best_S):
            best_S.add(x)
            
    return list(best_S)

if __name__ == "__main__":
    # Test
    res = construct()
    print(len(res))
