"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def _num_omega(n):
    """Count distinct prime factors."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 128
    universe = list(range(1, n + 1))
    universe_set = set(universe)
    
    # Bitset representation: bit i corresponds to element i (1-indexed, so bit 1 is element 1)
    # We use Python integers as arbitrary precision bitsets.
    
    # Precompute LCM table
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = _lcm(i, j)

    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    # Precompute bad_pairs[x]: list of (a, b) such that if a and b are selected, x cannot be added
    # Note: (a,b,x) is bad if lcm(a,b)==lcm(a,c)==lcm(b,c). Here c is x.
    bad_pairs = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = get_lcm(a, b)
            for c in range(b + 1, n + 1):
                if get_lcm(a, c) == lab and get_lcm(b, c) == lab:
                    # c cannot be added if a and b are present
                    bad_pairs[c].append((a, b))
                    # a cannot be added if b and c are present
                    bad_pairs[a].append((b, c))
                    # b cannot be added if a and c are present
                    bad_pairs[b].append((a, c))

    # Optimize bad_pairs for fast lookup: convert to list of tuples (a_bit, b_bit)
    # where a_bit = 1 << a, b_bit = 1 << b
    optimized_bad_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a, b in bad_pairs[x]:
            optimized_bad_pairs[x].append((1 << a, 1 << b))

    def can_add(x, selected_mask):
        """Check if x can be added to the set represented by selected_mask."""
        for a_bit, b_bit in optimized_bad_pairs[x]:
            if (selected_mask & a_bit) and (selected_mask & b_bit):
                return False
        return True

    # Greedy with a given order
    def greedy(order):
        selected_mask = 0
        for x in order:
            if can_add(x, selected_mask):
                selected_mask |= (1 << x)
        return selected_mask

    # Convert mask to list for return
    def mask_to_list(mask):
        return [i for i in range(1, n + 1) if (mask & (1 << i))]

    # Local search: Kick and Refill
    def local_search_kick_refill(initial_mask, max_iterations=20000):
        current_mask = initial_mask
        best_mask = initial_mask
        best_size = bin(initial_mask).count('1')
        
        # Precompute list of elements for random selection
        all_elements = list(universe)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            # Determine kick size
            # Smaller kicks are safer, larger kicks explore more
            # Start small, increase if stuck? Or just random.
            # Let's use a mix: mostly small, some large.
            kick_size = random.randint(1, 5)
            
            if kick_size > best_size:
                kick_size = max(1, best_size - 1)
                
            if kick_size == 0:
                continue

            # Select elements to remove from current_mask
            # We need to pick 'kick_size' bits that are set in current_mask
            set_bits = [i for i in range(1, n + 1) if (current_mask & (1 << i))]
            
            if len(set_bits) < kick_size:
                continue
                
            removed_elements = random.sample(set_bits, kick_size)
            
            # Remove them
            new_mask = current_mask
            for x in removed_elements:
                new_mask &= ~(1 << x)
                
            # Refill: greedily add any element that can be added
            # To make this faster and more effective, we can shuffle the candidates
            # or just iterate through all and add if possible.
            # Iterating through all 128 elements is fast enough (128 * 128 checks max)
            
            # Optimization: Only check elements not in new_mask
            # We can just iterate 1..128
            for x in range(1, n + 1):
                if new_mask & (1 << x):
                    continue
                if can_add(x, new_mask):
                    new_mask |= (1 << x)
                    
            new_size = bin(new_mask).count('1')
            
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask
                current_mask = new_mask # Move to new best to continue exploring around it
            else:
                # With some probability, accept a non-improving move to escape local minima?
                # Or just revert. Standard kick-refill usually reverts if not improved.
                # Let's stick to strict improvement for now to ensure we don't degrade.
                # current_mask remains unchanged.
                pass

        return best_mask

    # Generate multiple orderings
    def make_order(seed, strategy):
        if strategy == 0:
            # Natural order
            return list(universe)
        elif strategy == 1:
            # By number of prime factors
            return sorted(universe, key=lambda x: (_num_omega(x), x))
        elif strategy == 2:
            # Random
            random.seed(seed)
            order = list(universe)
            random.shuffle(order)
            return order
        elif strategy == 3:
            # Reverse natural
            return list(reversed(universe))
        elif strategy == 4:
            # By value descending, then omega
            return sorted(universe, key=lambda x: (-_num_omega(x), x))
        elif strategy == 5:
            # Random with bias toward low omega
            random.seed(seed)
            weights = [1.0 / (1 + _num_omega(x)) for x in universe]
            order = []
            remaining = list(universe)
            remaining_weights = weights[:]
            while remaining:
                total = sum(remaining_weights)
                r = random.random() * total
                cum = 0
                for idx, w in enumerate(remaining_weights):
                    cum += w
                    if cum >= r:
                        order.append(remaining[idx])
                        remaining.pop(idx)
                        remaining_weights.pop(idx)
                        break
            return order
        return list(universe)

    # Run multiple greedy passes with different strategies
    best_mask = 0
    best_size = 0
    candidates = [] # Store masks for local search

    # Deterministic strategies
    strategies = [0, 1, 3, 4]
    for s in strategies:
        if time.time() - start_time > DEADLINE:
            break
        order = make_order(0, s)
        mask = greedy(order)
        size = bin(mask).count('1')
        candidates.append((size, mask))
        if size > best_size:
            best_size = size
            best_mask = mask

    # Randomized greedy passes
    num_random_greedy = 50
    for i in range(num_random_greedy):
        if time.time() - start_time > DEADLINE - 10: # Leave time for local search
            break
        order = make_order(i * 12345 + 67890, 2)
        mask = greedy(order)
        size = bin(mask).count('1')
        candidates.append((size, mask))
        if size > best_size:
            best_size = size
            best_mask = mask

    # Biased random greedy passes (low omega)
    for i in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = make_order(i * 99999 + 1, 5)
        mask = greedy(order)
        size = bin(mask).count('1')
        candidates.append((size, mask))
        if size > best_size:
            best_size = size
            best_mask = mask

    # Sort candidates by size descending, take top few for local search
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Local search on the best candidates
    # We will run local search on the top 5-10 candidates to diversify search
    num_ls_attempts = 10
    for i in range(min(num_ls_attempts, len(candidates))):
        if time.time() - start_time > DEADLINE - 2:
            break
            
        size, mask = candidates[i]
        # Only run local search if it's close to the best or if it's the best
        if size >= best_size - 2:
            improved_mask = local_search_kick_refill(mask, max_iterations=5000)
            improved_size = bin(improved_mask).count('1')
            if improved_size > best_size:
                best_size = improved_size
                best_mask = improved_mask

    # Final polish on the absolute best
    if time.time() - start_time < DEADLINE - 1:
        improved_mask = local_search_kick_refill(best_mask, max_iterations=5000)
        improved_size = bin(improved_mask).count('1')
        if improved_size > best_size:
            best_size = improved_size
            best_mask = improved_mask

    # Convert best mask to list
    result = mask_to_list(best_mask)
    
    # Ensure nonempty
    if not result:
        result = [1]

    return result
# EVOLVE-BLOCK-END
