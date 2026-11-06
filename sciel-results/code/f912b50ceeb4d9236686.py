import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s limit
    
    MASK = (1 << 201) - 1  # Bits 0..200
    TARGET = 200
    TARGET_BIT = 1 << 200
    
    def compute_bits(elements):
        """Compute reachable bitset for a set of elements."""
        bits = 1
        for x in elements:
            bits |= (bits << x)
            bits &= MASK
        return bits
    
    def can_add(bits, x):
        """Check if adding x to a set with given bits keeps target unreachable."""
        new_bits = bits | (bits << x)
        new_bits &= MASK
        return not (new_bits & TARGET_BIT), new_bits
    
    # Phase 1: Greedy initialization - try multiple orders
    best_set = set()
    best_bits = 1
    best_size = 0
    
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        sorted(range(1, 101), key=lambda x: random.random()),
        sorted(range(1, 101), key=lambda x: random.random()),
        sorted(range(1, 101), key=lambda x: -x % 7),
    ]
    
    for order in orders:
        current_set = set()
        current_bits = 1
        for x in order:
            ok, new_bits = can_add(current_bits, x)
            if ok:
                current_set.add(x)
                current_bits = new_bits
        if len(current_set) > best_size:
            best_set = current_set
            best_bits = current_bits
            best_size = len(current_set)
    
    # Phase 2: Greedy addition - try to add any excluded element
    current_set = set(best_set)
    current_bits = best_bits
    current_size = best_size
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for x in range(1, 101):
            if x not in current_set:
                ok, new_bits = can_add(current_bits, x)
                if ok:
                    current_set.add(x)
                    current_bits = new_bits
                    current_size += 1
                    improved = True
    
    # Phase 3: Systematic local search - try (remove 1, add 2) moves
    # This can increase size by 1
    def try_remove_add2():
        """Try to find a move that removes one element and adds two, increasing size by 1."""
        nonlocal current_set, current_bits, current_size
        nonlocal start_time, time_limit
        
        included = list(current_set)
        excluded = [x for x in range(1, 101) if x not in current_set]
        
        if not excluded:
            return False
        
        # For each element to remove, try each pair of elements to add
        # But this is O(n^3) = up to 100^3 = 1M iterations, each with bitset ops
        # We need to be efficient
        
        # Precompute: for each excluded element, what would the bits look like
        # if we added it to the current set? (only valid if it doesn't create target)
        
        # Actually, let's just iterate carefully with time checks
        for y in included:
            if (time.time() - start_time) > time_limit:
                return False
            
            # Compute bits without y
            # We can't easily "remove" from bitset, so recompute
            # But we can use the fact that we're removing one element
            
            # Alternative: just recompute from scratch for the candidate set
            # This is expensive but correct
            
            # Let's try a different approach: for each y to remove,
            # compute the bits of S\{y}, then try adding pairs from excluded
            
            # Compute bits of S\{y}
            remaining = [x for x in current_set if x != y]
            bits_without_y = compute_bits(remaining)
            
            # Now try to add two elements from excluded
            # First, find which excluded elements can be added to S\{y}
            addable = []
            for x in excluded:
                ok, new_bits = can_add(bits_without_y, x)
                if ok:
                    addable.append((x, new_bits))
            
            # Now try pairs from addable
            for i in range(len(addable)):
                for j in range(i + 1, len(addable)):
                    x1, bits1 = addable[i]
                    x2, _ = addable[j]
                    # Check if we can add x2 to the set that already has x1
                    ok, new_bits = can_add(bits1, x2)
                    if ok:
                        # Valid move! Remove y, add x1 and x2
                        current_set.remove(y)
                        current_set.add(x1)
                        current_set.add(x2)
                        current_bits = new_bits
                        current_size += 1
                        return True
            
        return False
    
    # Phase 4: Iterative improvement with swaps
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try remove-1-add-2 moves
        if try_remove_add2():
            improved = True
            continue
        
        # Try single additions again (in case we missed some)
        for x in range(1, 101):
            if x not in current_set:
                ok, new_bits = can_add(current_bits, x)
                if ok:
                    current_set.add(x)
                    current_bits = new_bits
                    current_size += 1
                    improved = True
                    break
        
        # Try remove-1-add-1 swaps that enable future additions
        # This is harder; skip for now and rely on remove-1-add-2
    
    # Phase 5: Final polish - try more aggressive local search
    # Try random restarts of the local search from perturbed states
    for restart in range(20):
        if (time.time() - start_time) > time_limit:
            break
        
        # Perturb: remove 2-3 random elements, then re-add greedily
        perturbed_set = set(current_set)
        perturbed_bits = current_bits
        perturbed_size = current_size
        
        # Remove 2 random elements
        to_remove = random.sample(list(perturbed_set), min(2, len(perturbed_set)))
        for y in to_remove:
            perturbed_set.remove(y)
        
        # Recompute bits
        perturbed_bits = compute_bits(list(perturbed_set))
        perturbed_size = len(perturbed_set)
        
        # Greedily add back
        for x in range(1, 101):
            if x not in perturbed_set:
                ok, new_bits = can_add(perturbed_bits, x)
                if ok:
                    perturbed_set.add(x)
                    perturbed_bits = new_bits
                    perturbed_size += 1
        
        # Try remove-1-add-2 on perturbed state
        # (simplified: just check if perturbed is better)
        if perturbed_size > current_size:
            current_set = perturbed_set
            current_bits = perturbed_bits
            current_size = perturbed_size
    
    # Ensure nonempty
    result = list(current_set)
    if not result:
        result = [1]
    
    return result
