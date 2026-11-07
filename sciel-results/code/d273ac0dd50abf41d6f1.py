import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    # Max possible sum for 10 elements of max 256 is 2560.
    # Let's allow up to 12 elements just in case, max sum 3072.
    MAX_SUM_LIMIT = 4096

    def get_sums(elements):
        """
        Efficiently generate all non-empty subset sums.
        Returns a set of sums.
        """
        sums = {0}
        for x in elements:
            new_sums = {s + x for s in sums}
            sums.update(new_sums)
        sums.discard(0)
        return sums

    def check_divisibility(sums):
        """
        Check if any two distinct sums in the set divide each other.
        """
        sorted_sums = sorted(sums)
        # Optimization: If s divides t, then t = k*s.
        # We can iterate through each sum s, and check if any multiple k*s is in sums.
        # Since sums are up to 4000, we can use a lookup set for O(1) checks.
        sum_set = set(sums)
        for s in sorted_sums:
            # Check multiples of s
            multiple = s * 2
            while multiple <= max(sums):
                if multiple in sum_set:
                    return False
                multiple += s
        return True

    def is_valid(elements):
        """
        Full validation: distinct subset sums and no divisibility conflicts.
        """
        if not elements:
            return True
        sums = get_sums(elements)
        # Distinctness is guaranteed by using a set in get_sums? 
        # No, get_sums returns a set, so duplicates are removed.
        # But the task says: "Two different subsets with the same sum still make the set invalid".
        # So we must check if the number of unique sums is less than 2^n - 1.
        n = len(elements)
        if len(sums) < (1 << n) - 1:
            return False
        return check_divisibility(sums)

    def try_add(elements, val):
        """
        Try to add val to elements.
        Returns (success, new_sums) or (False, None).
        """
        if val in elements:
            return False, None
        
        # Get current sums
        # We need to check if adding val creates duplicate sums or divisibility conflicts.
        # New sums will be existing sums + val, plus val itself.
        # Actually, it's easier to just compute the new set of sums and check.
        # But we can prune.
        
        old_sums = get_sums(elements)
        # New sums are old_sums U {s + val for s in old_sums} U {val}
        # Wait, subset sums of A U {val} are S(A) U (S(A) + val).
        # 0 is in S(A) usually for DP, but we discarded it.
        # Let's stick to the full set including 0 for calculation, then discard 0.
        
        sums_with_0 = old_sums | {0}
        new_sums_with_0 = sums_with_0 | {s + val for s in sums_with_0}
        new_sums = new_sums_with_0 - {0}
        
        # Check for duplicates
        # If len(new_sums) < 2^(len(elements)+1) - 1, there's a collision.
        n_new = len(elements) + 1
        if len(new_sums) < (1 << n_new) - 1:
            return False, None
            
        # Check divisibility
        if not check_divisibility(new_sums):
            return False, None
            
        return True, new_sums

    def try_swap(elements, idx_remove, val_add):
        """
        Try to swap element at idx_remove with val_add.
        """
        if val_add in elements:
            return False, None
        
        new_elements = elements[:idx_remove] + elements[idx_remove+1:] + [val_add]
        # We can just validate the new elements directly
        if is_valid(new_elements):
            sums = get_sums(new_elements)
            return True, sums
        return False, None

    # Initial Greedy Construction
    # Sort candidates by some heuristic? Smaller numbers tend to create more conflicts?
    # Or larger numbers?
    # Let's try greedy with randomization.
    
    best_result = []
    
    # Run multiple random greedy attempts
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        random.seed(attempt)
        elements = []
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for val in candidates:
            if time.time() > deadline:
                break
            success, _ = try_add(elements, val)
            if success:
                elements.append(val)
                if len(elements) > len(best_result):
                    best_result = list(elements)
                    # If we reach a high size, we can try to optimize further
                    # but for now, just keep going if time permits.
                    # Actually, greedy might get stuck.
        
        # If we found a good set, try to improve it with local search
        if len(elements) > len(best_result):
            best_result = list(elements)
        
        # If we have a decent set, do a local search phase
        if len(elements) >= 5 and time.time() < deadline - 10:
            current_elements = list(elements)
            current_sums = get_sums(current_elements)
            
            # Local search: try to add or swap
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                
                # Try adding
                for val in range(1, 257):
                    if val in current_elements:
                        continue
                    success, new_sums = try_add(current_elements, val)
                    if success:
                        current_elements.append(val)
                        current_sums = new_sums
                        improved = True
                        if len(current_elements) > len(best_result):
                            best_result = list(current_elements)
                        break # Restart adding loop after improvement
                
                if not improved:
                    # Try swapping
                    for i in range(len(current_elements)):
                        for val in range(1, 257):
                            if val in current_elements:
                                continue
                            success, new_sums = try_swap(current_elements, i, val)
                            if success:
                                # Accept swap if it doesn't reduce size? 
                                # Swap keeps size same. 
                                # We only want to swap if it helps us add later?
                                # For now, accept random valid swaps to escape local minima.
                                if random.random() < 0.1: # Low prob to accept neutral swap
                                    current_elements[i] = val
                                    current_sums = new_sums
                                    improved = True
                                break
                        if improved:
                            break

    # Final validation of best_result
    if is_valid(best_result):
        return [int(x) for x in best_result]
    
    # Fallback: return empty or minimal valid set if something went wrong
    # A single element is always valid
    return [1]
