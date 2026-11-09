"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 256

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    
    Strategy:
    1. Use a randomized greedy approach with a "blocking" heuristic.
       At each step, among all remaining candidates that can be added to the current
       Sidon set, pick one that blocks (conflicts with) the most other candidates.
       This is analogous to finding small maximal independent sets in conflict graphs.
    2. Run multiple restarts with different random seeds to explore the space.
    3. Keep the smallest maximal Sidon set found.
    4. Use a deadline guard to ensure we finish before the 60s limit.
    """
    deadline = time.time() + 55.0  # 5s safety margin
    
    def is_sidon_valid(selected, sums_set, candidate):
        """Check if adding 'candidate' to 'selected' keeps the set Sidon."""
        new_sums = [candidate + a for a in selected] + [2 * candidate]
        # Check for duplicates within new_sums
        if len(set(new_sums)) != len(new_sums):
            return False
        # Check against existing sums
        if sums_set.intersection(new_sums):
            return False
        return True
    
    def greedy_blocking(seed):
        """
        Greedy algorithm with blocking heuristic.
        Returns a maximal Sidon set.
        """
        rng = random.Random(seed)
        selected = []
        sums_set = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            # Find all candidates that can be added
            valid_candidates = []
            for x in remaining:
                new_sums = [x + a for a in selected] + [2 * x]
                if len(set(new_sums)) == len(new_sums) and not sums_set.intersection(new_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # For each valid candidate, compute how many other candidates it blocks
            # A candidate x blocks y if adding x makes y no longer addable
            best_candidate = None
            best_block_count = -1
            best_random_tie = -1
            
            for x in valid_candidates:
                # Simulate adding x
                temp_sums = sums_set | set([x + a for a in selected] + [2 * x])
                temp_selected = selected + [x]
                
                block_count = 0
                for y in remaining:
                    if y == x:
                        continue
                    # Check if y is still valid after adding x
                    new_sums_y = [y + a for a in temp_selected] + [2 * y]
                    if len(set(new_sums_y)) != len(new_sums_y) or temp_sums.intersection(new_sums_y):
                        block_count += 1
                
                # Add a small random tiebreaker to diversify restarts
                random_tie = rng.random()
                
                if block_count > best_block_count or (block_count == best_block_count and random_tie > best_random_tie):
                    best_block_count = block_count
                    best_random_tie = random_tie
                    best_candidate = x
            
            if best_candidate is None:
                break
            
            # Add the best candidate
            x = best_candidate
            selected.append(x)
            sums_set.update([x + a for a in selected[:-1]] + [2 * x])
            remaining.remove(x)
            
            # Check deadline
            if time.time() > deadline:
                break
        
        # The set is now maximal (no more elements can be added)
        return selected
    
    def verify_maximal_sidon(selected):
        """Verify that selected is a valid Sidon set and is maximal."""
        if not selected:
            return False, False
        if len(set(selected)) != len(selected):
            return False, False
        if any(x < 1 or x > N for x in selected):
            return False, False
        
        # Check Sidon property
        sums = {}
        for i, a in enumerate(selected):
            for j in range(i, len(selected)):
                b = selected[j]
                s = a + b
                if s in sums:
                    return False, False
                sums[s] = (a, b)
        
        # Check maximality: no missing element can be added
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Try adding x
            new_sums = [x + a for a in selected] + [2 * x]
            if len(set(new_sums)) != len(new_sums):
                continue
            if sums.intersection(new_sums):
                continue
            # x can be added, so not maximal
            return True, False
        
        return True, True
    
    best_set = None
    best_size = N + 1
    
    # First, try a few deterministic seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        result = greedy_blocking(seed)
        if result:
            valid, maximal = verify_maximal_sidon(result)
            if valid and maximal:
                if len(result) < best_size:
                    best_size = len(result)
                    best_set = result[:]
    
    # If we have time, try more random seeds
    if time.time() < deadline - 2.0:
        rng = random.Random()
        attempts = 0
        while time.time() < deadline - 1.0:
            seed = rng.randint(0, 2**31 - 1)
            result = greedy_blocking(seed)
            if result:
                valid, maximal = verify_maximal_sidon(result)
                if valid and maximal:
                    if len(result) < best_size:
                        best_size = len(result)
                        best_set = result[:]
            attempts += 1
    
    if best_set is None:
        # Fallback: simple ascending greedy
        selected = []
        sums_set = set()
        for x in range(1, N + 1):
            new_sums = [x + a for a in selected] + [2 * x]
            if len(set(new_sums)) == len(new_sums) and not sums_set.intersection(new_sums):
                selected.append(x)
                sums_set.update(new_sums)
        return selected
    
    return best_set
# EVOLVE-BLOCK-END
