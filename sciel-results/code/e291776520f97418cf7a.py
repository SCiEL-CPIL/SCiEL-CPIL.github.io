"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin (evaluator kills at 90s)
    
    def check_validity(sums_set):
        """Check if a set of subset sums is valid (no two divide each other)."""
        if len(sums_set) <= 1:
            return True
        sorted_sums = sorted(sums_set)
        max_sum = sorted_sums[-1]
        for i in range(len(sorted_sums)):
            a = sorted_sums[i]
            for k in range(2, max_sum // a + 1):
                if (a * k) in sums_set:
                    return False
        return True
    
    def get_all_sums(chosen):
        """Get all non-empty subset sums of chosen elements."""
        if not chosen:
            return set()
        sums = set()
        for i in range(len(chosen)):
            sums.add(chosen[i])
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                sums.add(chosen[i] + chosen[j])
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                for k in range(j+1, len(chosen)):
                    sums.add(chosen[i] + chosen[j] + chosen[k])
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                for k in range(j+1, len(chosen)):
                    for l in range(k+1, len(chosen)):
                        sums.add(chosen[i] + chosen[j] + chosen[k] + chosen[l])
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                for k in range(j+1, len(chosen)):
                    for l in range(k+1, len(chosen)):
                        for m in range(l+1, len(chosen)):
                            sums.add(chosen[i] + chosen[j] + chosen[k] + chosen[l] + chosen[m])
        return sums
    
    def can_add_fast(x, current_sums):
        """Fast check if x can be added. Returns (valid, new_sums) or (False, None)."""
        # New sums = current_sums ∪ {x} ∪ {s+x for s in current_sums}
        new_sums = set(current_sums)
        new_sums.add(x)
        
        # Check for duplicates: s+x == s' for some s,s' in current_sums
        for s in current_sums:
            val = s + x
            if val in new_sums and val != x:
                return False, None
            new_sums.add(val)
        
        # Check divisibility
        if not check_validity(new_sums):
            return False, None
        return True, new_sums
    
    def greedy_from_candidates(candidates, initial_chosen=None, initial_sums=None):
        """Greedy construction from a list of candidates."""
        chosen = list(initial_chosen) if initial_chosen else []
        current_sums = set(initial_sums) if initial_sums else set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            ok, new_sums = can_add_fast(x, current_sums)
            if ok:
                chosen.append(x)
                current_sums = new_sums
        return chosen
    
    def randomized_search(num_restarts=200):
        """Try multiple random orderings and return the best result."""
        best = []
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Create a random permutation of candidates 2..256
            candidates = list(range(2, 257))
            random.shuffle(candidates)
            
            result = greedy_from_candidates(candidates)
            if len(result) > len(best):
                best = result
        
        return best
    
    # First, try the known good seed
    seed_result = [31, 32, 34]
    
    # Try greedy from high to low
    high_to_low = list(range(256, 1, -1))
    result1 = greedy_from_candidates(high_to_low)
    
    # Try randomized search
    result2 = randomized_search(100)
    
    # Pick the best
    candidates = [seed_result, result1, result2]
    best = max(candidates, key=len)
    
    # Try to improve by adding more elements with different strategies
    if len(best) < 10 and time.time() < deadline - 5:
        # Try adding elements one by one from a larger pool
        current_chosen = list(best)
        current_sums = get_all_sums(current_chosen)
        
        # Try all remaining numbers in random order
        remaining = [x for x in range(2, 257) if x not in current_chosen]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 2:
                break
            ok, new_sums = can_add_fast(x, current_sums)
            if ok:
                current_chosen.append(x)
                current_sums = new_sums
        
        if len(current_chosen) > len(best):
            best = current_chosen
    
    # Ensure we return a list of Python ints
    return list(best)

# EVOLVE-BLOCK-END
