import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Helper to check if a is strict subset of b
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    # Count comparable pairs in a family (list of masks)
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_strict_subset(family[i], family[j]) or is_strict_subset(family[j], family[i]):
                    count += 1
        return count
    
    # Generate initial family: use a mix of small and large sets to form chains
    # Strategy: select sets that are likely to be comparable
    # Use a greedy approach: start with empty set, then add sets that maximize comparable pairs
    
    # Initial solution: select masks that form a structured pattern
    # Use sets of different sizes to create potential chains
    random.seed(42)
    
    # Start with a diverse initial family
    # Include empty set, singletons, and larger sets
    initial_family = []
    
    # Add empty set
    initial_family.append(0)
    
    # Add some singletons
    for i in range(1, min(12, m - len(initial_family))):
        initial_family.append(1 << i)
    
    # Add some pairs
    for i in range(2):
        for j in range(i + 1, 12):
            if len(initial_family) >= m:
                break
            initial_family.append((1 << i) | (1 << j))
        if len(initial_family) >= m:
            break
    
    # Fill remaining with random masks
    used = set(initial_family)
    while len(initial_family) < m:
        mask = random.randint(0, total_masks - 1)
        if mask not in used:
            initial_family.append(mask)
            used.add(mask)
    
    # Ensure exactly m distinct masks
    family = initial_family[:m]
    
    best_family = family[:]
    best_score = count_comparable(best_family)
    
    # Local search: try swapping each element with a random other mask
    while time.time() < deadline:
        improved = False
        for i in range(m):
            if time.time() > deadline:
                break
            # Try replacing family[i] with a random mask
            for _ in range(5):  # Try a few random replacements per position
                if time.time() > deadline:
                    break
                new_mask = random.randint(0, total_masks - 1)
                if new_mask in family:
                    continue
                
                old_mask = family[i]
                family[i] = new_mask
                new_score = count_comparable(family)
                
                if new_score > best_score:
                    best_score = new_score
                    best_family = family[:]
                    improved = True
                else:
                    family[i] = old_mask
        
        if not improved:
            # Try a different random restart with small probability
            if random.random() < 0.1:
                family = best_family[:]
                # Perturb: swap a few elements
                for _ in range(3):
                    i = random.randint(0, m - 1)
                    j = random.randint(0, m - 1)
                    if i != j:
                        family[i], family[j] = family[j], family[i]
    
    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
    
    return result
