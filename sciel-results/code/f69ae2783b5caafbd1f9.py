import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Precompute reciprocals
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    
    def is_valid(S):
        """
        Returns True if S is reciprocal-subsum-free.
        """
        S_list = list(S)
        n = len(S_list)
        if n < 2:
            return True
            
        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i+1:]
            if not others:
                continue
                
            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Generate subset sums for left
            left_sums = [Fraction(0, 1)]
            for y in left:
                ry = recip[y]
                # Create new sums and extend
                new_sums = [s + ry for s in left_sums]
                left_sums.extend(new_sums)
                
            # Generate subset sums for right
            right_sums = [Fraction(0, 1)]
            for y in right:
                ry = recip[y]
                new_sums = [s + ry for s in right_sums]
                right_sums.extend(new_sums)
                
            right_set = set(right_sums)
            
            for l in left_sums:
                if (target - l) in right_set:
                    return False
                    
        return True

    def find_violation(S):
        """
        Returns (x, T) if violation found, else None.
        T is the subset of others that sums to 1/x.
        """
        S_list = list(S)
        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i+1:]
            if not others:
                continue
                
            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Store sums and the mask/indices to reconstruct T?
            # For simplicity, we just need to know if a violation exists to trigger repair.
            # But to remove the right element, we need to know which ones are involved.
            # We can reconstruct T by checking which subset sums to target.
            
            left_sums = {Fraction(0, 1): 0} # sum -> mask
            for j, y in enumerate(left):
                ry = recip[y]
                # Iterate over existing items to avoid modifying dict size during iteration
                items = list(left_sums.items())
                for s, mask in items:
                    new_s = s + ry
                    new_mask = mask | (1 << j)
                    if new_s not in left_sums:
                        left_sums[new_s] = new_mask
            
            right_sums = {Fraction(0, 1): 0}
            for j, y in enumerate(right):
                ry = recip[y]
                items = list(right_sums.items())
                for s, mask in items:
                    new_s = s + ry
                    new_mask = mask | (1 << j)
                    if new_s not in right_sums:
                        right_sums[new_s] = new_mask
            
            right_set = set(right_sums.keys())
            
            for l_sum, l_mask in left_sums.items():
                rem = target - l_sum
                if rem in right_set:
                    r_mask = right_sums[rem]
                    # Reconstruct T
                    T = []
                    for j in range(len(left)):
                        if l_mask & (1 << j):
                            T.append(left[j])
                    for j in range(len(right)):
                        if r_mask & (1 << j):
                            T.append(right[j])
                    return x, T
                    
        return None

    # Strategy: Start with full set, repair by removing elements.
    # Heuristic: Remove the element with the LARGEST value (smallest reciprocal) from the violation set {x} U T.
    # Why? Large numbers are "cheap" reciprocals. They are often used in many sums.
    # Removing a "cheap" element might break many potential violations?
    # Actually, let's try removing the element with the SMALLEST value (largest reciprocal) first?
    # Let's test both in a small loop or just pick one.
    # The "Remove Largest" heuristic worked for N=6,7.
    
    current_set = set(range(1, 25))
    
    max_repair_steps = 100
    for _ in range(max_repair_steps):
        if time.time() > deadline:
            break
            
        viol = find_violation(current_set)
        if viol is None:
            break
            
        x, T = viol
        candidates = [x] + T
        
        # Heuristic: Remove the element with the largest value
        # (i.e., smallest 1/value)
        # This tends to keep the "heavy" elements (1, 2, 3) which are harder to form.
        # Wait, if we keep 1, 2, 3, they are more likely to form sums?
        # 1/1 is large. 1/2 is large.
        # It is EASIER to form 1/1 (needs large terms) than 1/24 (needs small terms)?
        # No, 1/1 = 1. Needs sum of reciprocals = 1.
        # 1/24 = 0.04. Needs sum of reciprocals = 0.04.
        # With distinct integers, it's hard to get small sums.
        # So large numbers (24) are "safe" as targets?
        # But they are "dangerous" as components?
        # If we remove 24, we lose a component that might be part of a sum for 1?
        # 1/24 is small. It contributes little.
        # So removing 24 is "cheap" in terms of breaking sums?
        # Removing 1 is "expensive" because 1/1 is a big chunk.
        
        # Let's try removing the element with the LARGEST value.
        victim = max(candidates)
        current_set.remove(victim)
        
    # After repair, try to add back elements
    # This is a "growth" phase
    candidates = list(set(range(1, 25)) - current_set)
    # Sort by value?
    # Try adding in random order or sorted order
    for c in candidates:
        if time.time() > deadline:
            break
        if c in current_set:
            continue
        new_set = current_set | {c}
        if is_valid(new_set):
            current_set = new_set

    # Final polish: Try swaps
    # If we can swap a removed element for a kept element and gain size?
    # Swap doesn't change size.
    # But it might allow adding a third element.
    
    # Given time constraints, the repair + add-back strategy is robust.
    
    if not current_set:
        return [24]
        
    return list(current_set)

if __name__ == "__main__":
    result = construct()
    print(result)
