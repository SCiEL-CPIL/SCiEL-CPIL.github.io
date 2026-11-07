"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    def check_property(elements):
        """Check if the set of elements satisfies the cardinality-separation property."""
        n = len(elements)
        if n == 0:
            return True
        max_sum = sum(elements)
        # sums_by_card[k] = set of achievable sums using exactly k elements
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        
        for i, val in enumerate(elements):
            # iterate backwards to avoid using same element twice
            for k in range(i + 1, 0, -1):
                for s in list(sums_by_card[k - 1]):
                    sums_by_card[k].add(s + val)
        
        # Check cross-cardinality collisions
        for k in range(n + 1):
            for l in range(k + 1, n + 1):
                if sums_by_card[k] & sums_by_card[l]:
                    return False
        return True
    
    def greedy_construct(max_val=256, max_size=60):
        """Greedy: try to build a set by adding elements one by one."""
        elements = []
        # Try adding elements in descending order
        for candidate in range(max_val, 0, -1):
            if len(elements) >= max_size:
                break
            new_elements = elements + [candidate]
            if check_property(new_elements):
                elements.append(candidate)
        return elements
    
    # Try greedy with different strategies
    best = []
    
    # Strategy 1: descending order
    s1 = greedy_construct(256, 60)
    if len(s1) > len(best):
        best = s1
    
    if time.time() > deadline:
        return best
    
    # Strategy 2: Try to add elements in ascending order
    def greedy_ascending(max_val=256, max_size=60):
        elements = []
        for candidate in range(1, max_val + 1):
            if len(elements) >= max_size:
                break
            new_elements = elements + [candidate]
            if check_property(new_elements):
                elements.append(candidate)
        return elements
    
    s2 = greedy_ascending(256, 60)
    if len(s2) > len(best):
        best = s2
    
    if time.time() > deadline:
        return best
    
    # Strategy 3: Try to fill gaps - start with a base set and try inserting
    # Use a more sophisticated approach: start with sparse set, try to densify
    def densify(base_elements, max_val=256, max_size=60):
        """Try to add more elements to a base set."""
        elements = list(base_elements)
        changed = True
        while changed and len(elements) < max_size:
            changed = False
            for candidate in range(1, max_val + 1):
                if candidate in elements:
                    continue
                new_elements = elements + [candidate]
                if check_property(new_elements):
                    elements.append(candidate)
                    changed = True
                    break
        return elements
    
    # Start with a sparse base and densify
    # Base: powers of 2 scaled
    base1 = [1, 2, 4, 8, 16, 32, 64, 128]
    s3 = densify(base1, 256, 60)
    if len(s3) > len(best):
        best = s3
    
    if time.time() > deadline:
        return best
    
    # Try different bases
    base2 = [1, 3, 7, 15, 31, 63, 127]
    s4 = densify(base2, 256, 60)
    if len(s4) > len(best):
        best = s4
    
    if time.time() > deadline:
        return best
    
    # Random restarts
    import random
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        # Random initial set
        init_size = random.randint(3, 10)
        base = random.sample(range(1, 257), init_size)
        base.sort()
        s = densify(base, 256, 60)
        if len(s) > len(best):
            best = s
    
    # Ensure we return Python ints in a list
    return [int(x) for x in sorted(best)]

# EVOLVE-BLOCK-END
