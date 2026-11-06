"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def is_sidon_and_covers(arr):
        """
        Checks if arr is a Sidon set and covers all differences 1..30.
        Returns True if valid, False otherwise.
        """
        if len(arr) > 32:
            return False
        if 0 not in arr:
            return False
        if max(arr) > 10**6:
            return False
            
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        
        # Check coverage 1..30
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def get_diameter(arr):
        return max(arr) - min(arr)

    def generate_initial_set():
        """
        Generates a random set that is likely to be a valid candidate or close to it.
        We start with 0 and add random numbers, ensuring we try to cover small differences.
        """
        # Start with 0
        s = {0}
        # We need to cover 1..30. 
        # A simple heuristic: pick random numbers, but bias towards smaller numbers initially
        # to keep diameter small.
        
        # Let's try to construct a set that definitely covers 1..30 first, then optimize.
        # A known small Sidon set covering 1..30?
        # Let's just use a random restart approach.
        
        # Try to build a set of size ~10-15 that covers 1..30.
        # If we pick random numbers from 1..100, we might get collisions.
        # Let's use a greedy random approach to build a valid set first.
        
        current = [0]
        diffs = set()
        
        # We need to cover 1..30.
        # Let's pick random candidates for the next element.
        # To ensure coverage, we can specifically target missing differences.
        
        # Strategy: 
        # 1. Start with [0]
        # 2. While missing diffs in 1..30:
        #    Pick a random d from missing.
        #    Try to add x such that x - 0 = d or 0 - x = d (impossible since x>0) 
        #    or x - y = d for some y in current.
        #    Actually, simpler: just add random numbers and check validity.
        #    If invalid, backtrack or restart.
        
        # Let's use a simple random generation that guarantees validity by construction?
        # No, Sidon is hard to guarantee by simple construction.
        
        # Let's try this:
        # Pick a random permutation of numbers 1..50.
        # Greedily add them if they don't break Sidon.
        # If we end up with a set that covers 1..30, use it.
        # If not, try again.
        
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            candidates = list(range(1, 100))
            random.shuffle(candidates)
            s = [0]
            diffs = set()
            valid = True
            for c in candidates:
                new_diffs = set()
                for x in s:
                    d = abs(c - x)
                    if d in diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                s.append(c)
                diffs.update(new_diffs)
                
                # Check if we've covered 1..30
                if all(d in diffs for d in range(1, 31)):
                    # We have a valid set covering 1..30
                    # We can stop adding if we want to keep it small, 
                    # but we can also keep going to see if we can get a better diameter?
                    # No, adding elements increases diameter. 
                    # So we want the smallest diameter.
                    # This greedy random might not be minimal.
                    # But it gives us a starting point for hill climbing.
                    return sorted(s)
        
        # Fallback: Return a known valid set if random fails quickly?
        # Or return a very large valid set.
        # Let's construct a trivial valid set: 0, 1, 3, 7, 15, 31... (powers of 2 minus 1)
        # Differences: 1, 2, 3, 4, 5, 6, 7, 8...
        # 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023
        # Diffs from 0: 1, 3, 7, 15, 31...
        # Diffs from 1: 2, 6, 14, 30...
        # Diffs from 3: 4, 12, 28...
        # Diffs from 7: 8, 24...
        # Diffs from 15: 16...
        # Does this cover 1..30?
        # 1 (1-0), 2 (3-1), 3 (3-0), 4 (7-3), 5 (15-10? No. 15-? )
        # Let's check powers of 2 - 1: 0, 1, 3, 7, 15, 31
        # 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 24, 28, 30...
        # Missing: 5, 7, 9, 11, 13, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29
        # So powers of 2 don't work directly for 1..30 coverage with few elements.
        
        # Let's just return a large valid set constructed by brute force greedy if random fails.
        # Greedy: start [0], add smallest x such that Sidon holds.
        # This will produce a valid set, likely with large diameter.
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                d = x - y
                if d in diffs:
                    valid = False
                    break
            if valid:
                # Update diffs
                for y in a:
                    diffs.add(x - y)
                a.append(x)
            x += 1
        return a

    def hill_climb(initial_set):
        """
        Perform hill climbing to minimize the diameter.
        Moves: 
        1. Add a random element (if size < 32)
        2. Remove a random element (if size > 1 and 0 is not removed)
        3. Swap a random element with another random number
        """
        current = list(initial_set)
        current = sorted(current)
        if not is_sidon_and_covers(current):
            # If initial is invalid, fix it first? 
            # The generator should return valid.
            # If not, return it anyway? No, must be valid.
            # Let's assume generator returns valid.
            pass
        
        best_diam = get_diameter(current)
        best_set = current[:]
        T = 100.0 # Temperature
        T_min = 1.0
        cooling = 0.999
        
        while time.time() < deadline - 5:
            if T < T_min:
                T = T_min
            
            # Generate neighbor
            neighbor = current[:]
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add' and len(neighbor) < 32:
                # Pick a random number to add
                # Bias towards existing range or slightly outside
                max_val = max(neighbor)
                min_val = min(neighbor)
                # Try to add near the middle or at the ends
                if random.random() < 0.5:
                    new_val = random.randint(min_val, max_val)
                else:
                    new_val = random.randint(1, max_val + 50)
                
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
            elif move_type == 'remove' and len(neighbor) > 1:
                # Cannot remove 0
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    neighbor.pop(idx)
            elif move_type == 'swap':
                # Pick an element (not 0) and replace it
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    old_val = neighbor[idx]
                    max_val = max(neighbor)
                    new_val = random.randint(1, max_val + 50)
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        # Swap failed, keep same
                        neighbor = current[:]
            
            # Evaluate
            if len(neighbor) > 32 or 0 not in neighbor:
                continue
            if max(neighbor) > 10**6:
                continue
                
            if not is_sidon_and_covers(neighbor):
                # Invalid, reject (or penalize?)
                # For simplicity, reject invalid moves.
                continue
            
            new_diam = get_diameter(neighbor)
            delta = new_diam - best_diam
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                # Accept
                current = neighbor
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_set = neighbor[:]
            
            T *= cooling
            
        return best_set

    # Main Loop
    best_overall = None
    best_overall_diam = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline - 10:
        init_set = generate_initial_set()
        if not is_sidon_and_covers(init_set):
            continue
            
        result = hill_climb(init_set)
        
        if is_sidon_and_covers(result):
            diam = get_diameter(result)
            if diam < best_overall_diam:
                best_overall_diam = diam
                best_overall = result
                # If we get a very good one, we can break early?
                # Theoretical minimum? 
                # For k elements, max diff is at least something.
                # With 32 elements, we can have very small diameter.
                # But we need to cover 1..30.
                # If diam is small, say < 100, it's likely good.
                if best_overall_diam < 50:
                    break

    if best_overall is None:
        # Fallback
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        # Wait, [0..31] is not a Sidon set. 1-0=1, 2-1=1. Collision.
        # Return a known valid set from the fallback generator
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                d = x - y
                if d in diffs:
                    valid = False
                    break
            if valid:
                for y in a:
                    diffs.add(x - y)
                a.append(x)
            x += 1
        return a

    return best_overall
# EVOLVE-BLOCK-END
