import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    all_elements = list(range(1, 25))
    recip = {i: Fraction(1, i) for i in all_elements}
    
    def is_valid(subset):
        """Check if no element's reciprocal equals sum of reciprocals of a subset of others."""
        s = list(subset)
        n = len(s)
        if n <= 1:
            return True
        
        for i in range(n):
            target = recip[s[i]]
            others = [s[j] for j in range(n) if j != i]
            # Check all non-empty subsets of others
            m = len(others)
            # Use bit manipulation for subset enumeration
            for mask in range(1, 1 << m):
                total = Fraction(0, 1)
                for j in range(m):
                    if mask & (1 << j):
                        total += recip[others[j]]
                if total == target:
                    return False
        return True
    
    def count_violations(subset):
        """Count how many violation pairs exist."""
        s = list(subset)
        n = len(s)
        count = 0
        for i in range(n):
            target = recip[s[i]]
            others = [s[j] for j in range(n) if j != i]
            m = len(others)
            for mask in range(1, 1 << m):
                total = Fraction(0, 1)
                for j in range(m):
                    if mask & (1 << j):
                        total += recip[others[j]]
                if total == target:
                    count += 1
        return count
    
    # Strategy: Start with all 24, iteratively remove elements causing most violations
    current = set(all_elements)
    
    # First, check if all 24 is valid (unlikely)
    if is_valid(current):
        return sorted(current)
    
    # Greedy removal: repeatedly remove the element whose removal fixes most violations
    while not is_valid(current) and time.time() < deadline:
        # Find which element to remove
        best_remove = None
        best_violations_after = float('inf')
        
        for elem in sorted(current, key=lambda x: x):  # Try removing smaller elements first
            candidate = current - {elem}
            if not candidate:
                break
            # Count violations in candidate
            v = count_violations(candidate)
            if v < best_violations_after:
                best_violations_after = v
                best_remove = elem
            if v == 0:
                break
        
        if best_remove is None:
            break
        
        current.remove(best_remove)
    
    # Now try to add back elements we removed
    removed = set(all_elements) - current
    for elem in sorted(removed, reverse=True):  # Try adding larger elements first
        if time.time() > deadline:
            break
        candidate = current | {elem}
        if is_valid(candidate):
            current = candidate
    
    # Try another pass of adding
    removed = set(all_elements) - current
    for elem in sorted(removed, reverse=True):
        if time.time() > deadline:
            break
        candidate = current | {elem}
        if is_valid(candidate):
            current = candidate
    
    result = sorted(current)
    if not result:
        result = [24]
    
    return result

