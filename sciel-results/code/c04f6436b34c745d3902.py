import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute subset relations: sub[m1][m2] = True if m1 is strict subset of m2
    # We'll use a list of lists or a set of pairs for fast lookup
    # Actually, let's precompute for each mask, the list of masks it is a strict subset of
    # And the list of masks that are strict subsets of it
    
    # For efficiency, let's precompute:
    # strict_subsets[mask] = set of masks that are strict subsets of mask
    # strict_supersets[mask] = set of masks that are strict supersets of mask
    
    # Actually, for counting comparable pairs in a family, we can just check pairs.
    # But for local search, we need fast evaluation.
    
    # Let's precompute a boolean matrix or use bit tricks.
    # is_sub[a][b] = 1 if a is strict subset of b
    
    # Better: precompute for each mask a, the bitmask of all b such that a is strict subset of b
    # That's 4096 * 4096 bits = 2M bits = 256KB, manageable.
    
    # Actually, let's just precompute a list of comparable pairs for all 4096 masks.
    # There are C(4096,2) = ~8.4M pairs. That's a lot but doable.
    # But we only need it for local search moves.
    
    # Alternative: for a given family (list of 40 masks), count comparable pairs in O(m^2) = 1600 checks.
    # Each check is O(1) with bit ops. So 1600 ops per evaluation.
    # For local search with many moves, this is fine.
    
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    def count_comparable(family):
        """Count comparable pairs in family (list of masks)."""
        count = 0
        k = len(family)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = family[i], family[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (a & b) == b and a != b:
                    count += 1
        return count
    
    def evaluate_and_update(family, current_score, removed_mask, added_mask):
        """
        Efficiently update score when replacing removed_mask with added_mask.
        Returns new_score, and whether the move is valid (no duplicates).
        """
        # Check for duplicates
        if added_mask in family and added_mask != removed_mask:
            return None, False
        
        # Compute change in comparable pairs
        # Remove contributions involving removed_mask
        delta = 0
        for other in family:
            if other == removed_mask:
                continue
            # Was removed_mask comparable with other?
            if (removed_mask & other) == removed_mask and removed_mask != other:
                delta -= 1
            elif (removed_mask & other) == other and removed_mask != other:
                delta -= 1
            # Will added_mask be comparable with other?
            if (added_mask & other) == added_mask and added_mask != other:
                delta += 1
            elif (added_mask & other) == other and added_mask != other:
                delta += 1
        
        new_score = current_score + delta
        return new_score, True
    
    # Greedy construction: build a chain-like structure
    # Start with empty set, add sets that maximize comparable pairs
    
    # Strategy: use a greedy approach with local search
    # 1. Start with a random or structured initial family
    # 2. Perform local search (1-move, 2-move) to improve
    
    best_family = None
    best_score = -1
    
    # Multiple restarts
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize family
        if restart == 0:
            # Structured start: nested sets
            family = []
            # Create a chain of sets of increasing size
            for size in range(0, 13):  # sizes 0 to 12
                # Pick a set of this size
                if size == 0:
                    family.append(0)
                else:
                    # Use first 'size' bits
                    mask = (1 << size) - 1
                    if mask < total_masks:
                        family.append(mask)
            # Pad to 40 with more sets
            random.seed(42 + restart)
            while len(family) < m:
                mask = random.randint(0, total_masks - 1)
                if mask not in family:
                    family.append(mask)
        else:
            # Random start with bias toward comparable sets
            random.seed(1000 + restart)
            family = []
            # Pick a random chain first
            base_mask = 0
            for _ in range(min(15, m)):
                # Add a random bit
                bits = [b for b in range(n) if not (base_mask >> b) & 1]
                if bits:
                    b = random.choice(bits)
                    base_mask |= (1 << b)
                if base_mask not in family:
                    family.append(base_mask)
            
            # Fill rest randomly
            while len(family) < m:
                mask = random.randint(0, total_masks - 1)
                if mask not in family:
                    family.append(mask)
        
        current_score = count_comparable(family)
        
        # Local search: 1-move optimization
        improved = True
        max_iterations = 200
        iteration = 0
        
        while improved and iteration < max_iterations:
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            iteration += 1
            
            # Try all 1-moves: replace one set with another
            best_move = None
            best_delta = 0
            
            for i in range(m):
                old_mask = family[i]
                # Try adding a random set of masks
                for _ in range(30):  # Try 30 random candidates per position
                    new_mask = random.randint(0, total_masks - 1)
                    if new_mask == old_mask:
                        continue
                    if new_mask in family:
                        continue
                    
                    # Quick check: compute delta
                    delta = 0
                    for j in range(m):
                        if j == i:
                            continue
                        other = family[j]
                        # Was old_mask comparable with other?
                        if (old_mask & other) == old_mask and old_mask != other:
                            delta -= 1
                        elif (old_mask & other) == other and old_mask != other:
                            delta -= 1
                        # Will new_mask be comparable with other?
                        if (new_mask & other) == new_mask and new_mask != other:
                            delta += 1
                        elif (new_mask & other) == other and new_mask != other:
                            delta += 1
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (i, new_mask)
            
            if best_move and best_delta > 0:
                i, new_mask = best_move
                family[i] = new_mask
                current_score += best_delta
                improved = True
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_family = list(family)
    
    # Convert to list of lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
