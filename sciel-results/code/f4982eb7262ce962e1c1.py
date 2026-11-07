"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    def compute_sums_by_card(elements):
        """Compute subset sums indexed by cardinality."""
        n = len(elements)
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        for i, val in enumerate(elements):
            for k in range(i + 1, 0, -1):
                for s in list(sums_by_card[k - 1]):
                    sums_by_card[k].add(s + val)
        return sums_by_card
    
    def check_full(sums_by_card, n):
        """Check cross-cardinality collisions for full sums_by_card."""
        for k in range(n + 1):
            for l in range(k + 1, n + 1):
                if sums_by_card[k] & sums_by_card[l]:
                    return False
        return True
    
    def try_add_element(elements, sums_by_card, candidate):
        """Try adding candidate to elements. Returns (success, new_sums_by_card)."""
        n = len(elements)
        new_sums = [set(s) for s in sums_by_card]
        new_sums.append(set())
        
        for k in range(n + 1, 0, -1):
            for s in new_sums[k - 1]:
                new_sums[k].add(s + candidate)
        
        # Check if new cardinality level (n+1) collides with any existing level
        new_level = new_sums[n + 1]
        for k in range(n + 1):
            if new_level & new_sums[k]:
                return False, None
        
        # Also check: adding candidate may have added new sums to existing levels
        # that now collide with other levels. We need to check all pairs involving
        # levels that were modified.
        # Actually, only level n+1 is entirely new. Levels 1..n may have gained elements.
        # We need to check if any level k (1<=k<=n) now has a sum that's in level l (l!=k).
        # But we already verified no collisions before. The only new sums are in levels 1..n+1.
        # A sum s in new_sums[k] (k<=n) that wasn't in old sums_by_card[k] could collide
        # with some sum in new_sums[l] for l != k.
        # For efficiency, let's just do a full check if n is small, or check incrementally.
        
        # Incremental check: for each level k in 1..n, find newly added sums and check
        # against all other levels.
        for k in range(1, n + 1):
            old_k = sums_by_card[k]
            new_k = new_sums[k]
            added = new_k - old_k
            if added:
                for l in range(n + 2):
                    if l == k:
                        continue
                    if added & new_sums[l]:
                        return False, None
        
        return True, new_sums
    
    def greedy_with_state(candidate_order, max_size=60):
        """Greedy construction maintaining incremental subset sum state."""
        elements = []
        sums_by_card = [set()]
        sums_by_card[0].add(0)
        
        for candidate in candidate_order:
            if len(elements) >= max_size:
                break
            if time.time() > deadline:
                break
            success, new_sums = try_add_element(elements, sums_by_card, candidate)
            if success:
                elements.append(candidate)
                sums_by_card = new_sums
        
        return elements
    
    def densify_with_state(base_elements, candidate_order, max_size=60):
        """Densify a base set using incremental state."""
        elements = list(base_elements)
        sums_by_card = compute_sums_by_card(elements)
        
        if not check_full(sums_by_card, len(elements)):
            return elements
        
        changed = True
        while changed and len(elements) < max_size:
            changed = False
            for candidate in candidate_order:
                if time.time() > deadline:
                    break
                if candidate in elements:
                    continue
                success, new_sums = try_add_element(elements, sums_by_card, candidate)
                if success:
                    elements.append(candidate)
                    sums_by_card = new_sums
                    changed = True
                    break
        
        return elements
    
    # Strategy 1: descending order
    best = greedy_with_state(range(256, 0, -1))
    
    if time.time() > deadline:
        return [int(x) for x in sorted(best)]
    
    # Strategy 2: ascending order
    s2 = greedy_with_state(range(1, 257))
    if len(s2) > len(best):
        best = s2
    
    if time.time() > deadline:
        return [int(x) for x in sorted(best)]
    
    # Strategy 3: Densify from power-of-2 base
    base1 = [1, 2, 4, 8, 16, 32, 64, 128]
    s3 = densify_with_state(base1, range(1, 257))
    if len(s3) > len(best):
        best = s3
    
    if time.time() > deadline:
        return [int(x) for x in sorted(best)]
    
    # Strategy 4: Densify from another base
    base2 = [1, 3, 7, 15, 31, 63, 127]
    s4 = densify_with_state(base2, range(1, 257))
    if len(s4) > len(best):
        best = s4
    
    if time.time() > deadline:
        return [int(x) for x in sorted(best)]
    
    # Strategy 5: Densify with descending candidate order
    s5 = densify_with_state(base1, range(256, 0, -1))
    if len(s5) > len(best):
        best = s5
    
    if time.time() > deadline:
        return [int(x) for x in sorted(best)]
    
    # Random restarts with incremental state
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random initial set
        init_size = random.randint(2, 12)
        base = random.sample(range(1, 257), init_size)
        base.sort()
        
        # Try both ascending and descending densification
        order = range(1, 257) if trial % 2 == 0 else range(256, 0, -1)
        s = densify_with_state(base, order)
        if len(s) > len(best):
            best = s
    
    # Final polish: try to add any remaining elements to best set
    if time.time() < deadline - 5:
        best_elements = sorted(best)
        sums_by_card = compute_sums_by_card(best_elements)
        if check_full(sums_by_card, len(best_elements)):
            changed = True
            while changed and len(best_elements) < 60:
                changed = False
                for candidate in range(1, 257):
                    if time.time() > deadline - 2:
                        break
                    if candidate in best_elements:
                        continue
                    success, new_sums = try_add_element(best_elements, sums_by_card, candidate)
                    if success:
                        best_elements.append(candidate)
                        sums_by_card = new_sums
                        changed = True
                        break
            best = best_elements
    
    # Ensure we return Python ints in a list
    return [int(x) for x in sorted(best)]

# EVOLVE-BLOCK-END
