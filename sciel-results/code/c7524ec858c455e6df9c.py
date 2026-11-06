import itertools
import time

def construct():
    # Based on analysis, the set {7, 9, 10, ..., 24} is likely valid and has size 17.
    # Let's verify this specific set and return it.
    # If we want to be safe, we can run a quick check.
    
    candidate = [7] + list(range(9, 25))
    
    # Quick verification to ensure no bugs in manual derivation
    recips = {i: 1.0/i for i in candidate}
    valid = True
    for x in candidate:
        target = recips[x]
        others = [y for y in candidate if y != x]
        for r in range(1, len(others)+1):
            for subset in itertools.combinations(others, r):
                if abs(sum(recips[y] for y in subset) - target) < 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to smaller known good set if verification fails
        return list(range(13, 25))

    # Try to add 3 to the candidate set
    # Check if {3, 7, 9, ..., 24} is valid
    new_candidate = [3] + candidate
    recips_new = {i: 1.0/i for i in new_candidate}
    valid_new = True
    # We only need to check if 3 is a subsum of others, 
    # and if adding 3 creates a subsum for any existing element.
    # Actually, adding 3 might make an existing element x have a new subset sum 
    # that includes 3.
    # So we must re-verify all elements.
    for x in new_candidate:
        target = recips_new[x]
        others = [y for y in new_candidate if y != x]
        # Optimization: only check subsets that include 3? 
        # No, existing subsets are already valid. 
        # New subsets must include 3.
        # So for each x, check if 1/x = 1/3 + sum(subset of others\{3})
        # Or if x != 3, 1/x = sum(subset of others) where subset might include 3.
        # This is still exponential.
        # Given time, we can just run the full check for the new candidate.
        # But it's slow.
        # Let's just check if 3 is safe (not a subsum) 
        # and if any existing x becomes unsafe.
        # A fast heuristic:
        # If 3 is not a subsum of others, and 1/3 is not a subsum of others, 
        # it doesn't mean existing x are safe.
        # Example: 1/6 = 1/12 + 1/12. If we add 3, 1/6 = 1/12 + 1/12 still holds.
        # But 1/6 = 1/3 + 1/6? No.
        # 1/6 = 1/3 + 1/12 + 1/12? No.
        # The risk is 1/x = 1/3 + S.
        # So 1/3 = 1/x - S.
        # We can check for each x != 3, if 1/x - 1/3 is a subset sum of others\{3,x}.
        # This is still hard.
        
        # Let's just assume 3 is likely invalid or valid and try to verify with a timeout?
        # No, let's just return the current valid set to be safe, 
        # unless we can quickly verify 3.
        
        pass

    # For now, return the verified candidate
    return candidate
