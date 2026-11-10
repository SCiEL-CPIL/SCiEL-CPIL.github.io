import time
import random
import sys

def construct():
    # Set a strict deadline to ensure we return within the limit (90s hard limit, 60s soft limit mentioned in contract)
    # We use 50s to be safe against overhead and evaluation environment variations.
    deadline = time.time() + 50
    n = 256
    
    # Precompute conflicts using bitmasks
    # conflicts_mask[i] is a bitmask of j such that (i*j) % (i+j) == 0
    # We use 0-indexed internally for bitmask operations (0..255 representing 1..256)
    # So index i corresponds to value i+1
    conflicts_mask = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                # Set bit b in mask for a, and bit a in mask for b
                # Using 1-indexed values directly as bit positions might be wasteful (bit 256)
                # But Python integers handle arbitrary size. 
                # Alternatively, map value v to bit v-1. Let's use bit v-1 for efficiency in counting.
                # Actually, using 1<<v is fine and easier to read, Python ints are arbitrary precision.
                conflicts_mask[a] |= (1 << b)
                conflicts_mask[b] |= (1 << a)
                
    def is_valid_set(s):
        """Check if set s is a valid independent set."""
        mask = 0
        for a in s:
            if conflicts_mask[a] & mask:
                return False
            mask |= (1 << a)
        return True

    def greedy_build(order, start_mask=0):
        """Build a set by greedily adding elements from order that don't conflict."""
        s = set()
        mask = start_mask
        for e in order:
            if not (conflicts_mask[e] & mask):
                s.add(e)
                mask |= (1 << e)
        return s, mask

    def local_search(s, max_iters=500, deadline_check=None):
        """
        Local search with add, 1-swap (1-for-1), 2-swap (2-for-1), and 2-for-1 growth moves.
        """
        s = set(s)
        mask = 0
        for a in s:
            mask |= (1 << a)
        best_size = len(s)
        
        # Precompute list of all elements for iteration
        all_elements = list(range(1, n + 1))
        
        for _ in range(max_iters):
            if deadline_check and time.time() > deadline_check:
                break
            improved = False
            
            # 1. Try to add any non-member (Growth)
            # Optimization: Only check elements not in mask
            for b in all_elements:
                if not (mask & (1 << b)):
                    if not (conflicts_mask[b] & mask):
                        s.add(b)
                        mask |= (1 << b)
                        best_size = len(s)
                        improved = True
                        break
            if improved:
                continue
            
            # 2. Try 1-swap: remove a, add b
            # b must conflict with exactly one element a in S
            # To speed up, iterate over b not in S
            for b in all_elements:
                if mask & (1 << b):
                    continue
                conflicts_with_b = conflicts_mask[b] & mask
                if conflicts_with_b == 0:
                    continue  # Should have been caught in step 1
                if conflicts_with_b & (conflicts_with_b - 1) == 0:
                    # Exactly one conflict
                    a_bit = conflicts_with_b
                    a = a_bit.bit_length() - 1
                    # Swap: remove a, add b
                    new_mask = (mask ^ a_bit) | (1 << b)
                    s.discard(a)
                    s.add(b)
                    mask = new_mask
                    # Size stays same, but structure changes
                    improved = True
                    break
            if improved:
                continue

            # 3. Try 2-swap (Lateral): remove a, c, add b
            # b must conflict with exactly two elements a, c in S
            for b in all_elements:
                if mask & (1 << b):
                    continue
                conflicts_with_b = conflicts_mask[b] & mask
                if conflicts_with_b == 0:
                    continue
                # Check if exactly 2 bits set
                # popcount check
                c1 = conflicts_with_b
                c2 = c1 & (c1 - 1)
                if c2 == 0:
                    continue # 1 bit
                c3 = c2 & (c2 - 1)
                if c3 != 0:
                    continue # 3+ bits
                
                # Exactly 2 conflicts
                # Extract the two bits
                a_bit = conflicts_with_b & (-conflicts_with_b)
                c_bit = conflicts_with_b ^ a_bit
                a = a_bit.bit_length() - 1
                c = c_bit.bit_length() - 1
                
                # Remove a, c, add b
                new_mask = (mask & ~a_bit & ~c_bit) | (1 << b)
                s.discard(a)
                s.discard(c)
                s.add(b)
                mask = new_mask
                # Size decreases by 1. This is a "downhill" move, usually only done in SA.
                # In standard local search for max independent set, we might skip this unless we want to escape.
                # Let's skip pure size-decreasing moves in this deterministic pass to maintain progress,
                # unless we are stuck. But "stuck" is defined by no add or swap.
                # Actually, let's treat this as a valid move if it enables future growth.
                # For simplicity and speed, let's prioritize moves that don't decrease size.
                # If we are here, we found no add and no 1-swap.
                # A 2-for-1 move decreases size. It might be worth it to escape a trap.
                # Let's do it with some probability or just do it.
                improved = True
                break
            if improved:
                continue
            
            # 4. Try 2-for-1 Growth? No, that's remove 1 add 2.
            # Remove a, add b, c.
            # b and c must not conflict with each other.
            # b and c must not conflict with S \ {a}.
            # b and c can conflict with a.
            
            # This is more expensive. Let's try a random kick if stuck.
            
            # Random Kick: Remove K random elements, then greedily re-add.
            K = random.randint(3, 8)
            to_remove = random.sample(list(s), min(K, len(s)))
            for r in to_remove:
                s.discard(r)
                mask &= ~(1 << r)
            
            # Re-add greedily
            # Shuffle order for randomization
            candidates = [e for e in all_elements if e not in s]
            random.shuffle(candidates)
            for e in candidates:
                if not (conflicts_mask[e] & mask):
                    s.add(e)
                    mask |= (1 << e)
            
            improved = True
            # If we didn't improve size, we might have made it worse. 
            # Local search will try to fix it.
            
        return s, best_size

    best = []
    best_size = 0
    
    # Phase 1: Initial Construction
    # Heuristic: Start with elements that have low degree (fewer conflicts)
    degrees = [bin(conflicts_mask[i]).count('1') for i in range(1, n + 1)]
    # Sort by degree ascending
    initial_order = sorted(range(1, n + 1), key=lambda x: degrees[x-1])
    
    s, mask = greedy_build(initial_order, 0)
    s, sz = local_search(s, max_iters=1000, deadline_check=deadline - 10)
    if sz > best_size:
        best = list(s)
        best_size = sz
        
    # Phase 2: Multiple Restarts with Randomized Greedy + Local Search
    # Use remaining time for restarts
    # Each restart: 
    # 1. Generate a random ordering (or perturb best solution)
    # 2. Greedy build
    # 3. Local search
    
    all_elements = list(range(1, n + 1))
    
    while time.time() < deadline - 5:
        # Randomized restart
        # Mix of random order and order based on degree with noise
        # Or start from best and kick
        
        # Strategy A: Start from best, kick, rebuild
        if best and random.random() < 0.7:
            s = set(best)
            # Kick: remove K random elements
            K = random.randint(5, 15)
            to_remove = random.sample(list(s), min(K, len(s)))
            for r in to_remove:
                s.discard(r)
            
            # Rebuild mask
            mask = 0
            for a in s:
                mask |= (1 << a)
            
            # Greedily re-add from all elements not in s, in random order
            remaining = [e for e in all_elements if e not in s]
            random.shuffle(remaining)
            
            for e in remaining:
                if not (conflicts_mask[e] & mask):
                    s.add(e)
                    mask |= (1 << e)
        else:
            # Strategy B: Fresh random greedy
            # Weighted random order? Just random shuffle
            random.shuffle(all_elements)
            s, mask = greedy_build(all_elements, 0)
        
        # Local search
        s, sz = local_search(s, max_iters=500, deadline_check=deadline - 5)
        
        if sz > best_size:
            best = list(s)
            best_size = sz
            # If we found a new best, we can increase the "kick" aggressiveness or just continue
            
    return best
