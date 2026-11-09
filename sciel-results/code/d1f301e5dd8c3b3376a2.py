"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 16
DEADLINE = time.time() + 80  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _apply_comparator(arr, i, j):
    if arr[i] > arr[j]:
        arr[i], arr[j] = arr[j], arr[i]

def _verify_network(comparators, sample_size=0):
    """Verify network sorts all 2^16 inputs (or a sample). Returns True if valid."""
    total = 1 << N
    if sample_size == 0 or sample_size >= total:
        indices = range(total)
    else:
        indices = random.sample(range(total), sample_size)
    for mask in indices:
        arr = [(mask >> k) & 1 for k in range(N)]
        for i, j in comparators:
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
        # Check sorted
        for k in range(N - 1):
            if arr[k] > arr[k + 1]:
                return False
    return True

def _compute_depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    # Greedy layering: assign each comparator to earliest possible layer
    wire_last_layer = [-1] * N
    max_layer = 0
    for i, j in comparators:
        layer = max(wire_last_layer[i], wire_last_layer[j]) + 1
        wire_last_layer[i] = layer
        wire_last_layer[j] = layer
        if layer > max_layer:
            max_layer = layer
    return max_layer

def _batcher_merge_sort_network(n):
    """Build Batcher's odd-even merge sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comparators = []
    # Recursive construction
    if n % 2 == 0:
        # Split into two halves
        half = n // 2
        left = _batcher_merge_sort_network(half)
        right = _batcher_merge_sort_network(half)
        
        # Apply to left half
        for i, j in left:
            comparators.append((i, j))
        # Apply to right half
        for i, j in right:
            comparators.append((i + half, j + half))
        
        # Odd-even merge
        for i in range(half):
            comparators.append((i, i + half))
        
        # Recursively merge odd and even subsequences
        # Odd subsequence: wires 1, 3, 5, ..., n-1
        # Even subsequence: wires 0, 2, 4, ..., n-2
        
        # After the first pass of comparators (i, i+half), we need to merge
        # the odd positions among themselves and even positions among themselves
        
        # Actually, let's use the standard recursive odd-even merge
        # For even n, we split into two halves of size n/2
        # Merge them using odd-even merge
        
        # Standard odd-even merge for two sorted lists of size n/2:
        # 1. Compare (0, n/2), (1, n/2+1), ... (n/2-1, n-1)
        # 2. Recursively merge odd-indexed elements
        # 3. Recursively merge even-indexed elements
        
        # The comparators so far are just the two recursive sorts
        # Now add the odd-even merge comparators
        
        # Step 1: Compare (i, i + n/2) for i in 0..n/2-1
        # Already done above
        
        # Step 2: Recursively merge the odd-indexed subsequence
        # Odd indices: 1, 3, 5, ..., n-1
        # Map to 0, 1, 2, ..., n/2-1
        odd_comparators = _batcher_merge_sort_network(half)
        for i, j in odd_comparators:
            # Map back to original indices
            # Odd index i -> 2*i + 1
            comparators.append((2 * i + 1, 2 * j + 1))
        
        # Step 3: Recursively merge the even-indexed subsequence
        # Even indices: 0, 2, 4, ..., n-2
        # Map to 0, 1, 2, ..., n/2-1
        even_comparators = _batcher_merge_sort_network(half)
        for i, j in even_comparators:
            # Map back to original indices
            # Even index i -> 2*i
            comparators.append((2 * i, 2 * j))
    else:
        # For odd n, add a dummy wire
        # This is getting complex, let's just handle powers of 2 and use
        # a simpler approach for n=16
        pass
    
    return comparators

def _odd_even_merge_sort_network(n):
    """Build odd-even merge sort network for n wires (power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    comparators = []
    
    # Recursively sort left and right halves
    left = _odd_even_merge_sort_network(half)
    right = _odd_even_merge_sort_network(half)
    
    for i, j in left:
        comparators.append((i, j))
    for i, j in right:
        comparators.append((i + half, j + half))
    
    # Odd-even merge the two sorted halves
    # Step 1: Compare (i, i + half) for i in 0..half-1
    for i in range(half):
        comparators.append((i, i + half))
    
    # Step 2: Recursively merge odd-indexed subsequence
    odd = _odd_even_merge_sort_network(half)
    for i, j in odd:
        comparators.append((2 * i + 1, 2 * j + 1))
    
    # Step 3: Recursively merge even-indexed subsequence
    even = _odd_even_merge_sort_network(half)
    for i, j in even:
        comparators.append((2 * i, 2 * j))
    
    return comparators

def _remove_redundant_comparators(comparators):
    """Try to remove redundant comparators from a valid network."""
    comparators = list(comparators)
    n = len(comparators)
    # Try removing each comparator and check if still valid
    for idx in range(n - 1, -1, -1):
        if time.time() > DEADLINE - 2:
            break
        # Temporarily remove
        test = comparators[:idx] + comparators[idx + 1:]
        # Quick check with sample
        if _verify_network(test, sample_size=2000):
            # Full verification
            if _verify_network(test):
                comparators = test
    return comparators

def _local_search(comparators, time_limit=20):
    """Local search: try swapping, inserting, deleting comparators."""
    start_time = time.time()
    best = list(comparators)
    best_score = len(best)
    
    while time.time() - start_time < time_limit:
        if time.time() > DEADLINE - 2:
            break
        
        improved = False
        n = len(best)
        
        # Try deleting a random comparator
        if n > 0:
            idx = random.randint(0, n - 1)
            candidate = best[:idx] + best[idx + 1:]
            if _verify_network(candidate, sample_size=1000):
                if _verify_network(candidate):
                    best = candidate
                    best_score = len(best)
                    improved = True
        
        # Try swapping two comparators
        if not improved and n >= 2:
            a, b = random.sample(range(n), 2)
            candidate = best[:]
            candidate[a], candidate[b] = candidate[b], candidate[a]
            if _verify_network(candidate, sample_size=1000):
                if _verify_network(candidate):
                    best = candidate
                    improved = True
        
        # Try replacing a comparator with a random one
        if not improved and n > 0:
            idx = random.randint(0, n - 1)
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            candidate = best[:]
            candidate[idx] = (i, j)
            if _verify_network(candidate, sample_size=1000):
                if _verify_network(candidate):
                    best = candidate
                    improved = True
        
        # Try inserting a random comparator
        if not improved:
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            idx = random.randint(0, len(best))
            candidate = best[:idx] + [(i, j)] + best[idx:]
            if _verify_network(candidate, sample_size=1000):
                if _verify_network(candidate):
                    best = candidate
                    improved = True
        
        if not improved:
            # Try more aggressive mutations
            for _ in range(3):
                candidate = best[:]
                for _ in range(random.randint(1, 3)):
                    op = random.randint(0, 2)
                    if op == 0 and len(candidate) > 0:
                        idx = random.randint(0, len(candidate) - 1)
                        candidate.pop(idx)
                    elif op == 1:
                        i, j = random.sample(range(N), 2)
                        if i > j:
                            i, j = j, i
                        idx = random.randint(0, len(candidate))
                        candidate.insert(idx, (i, j))
                    else:
                        if len(candidate) > 0:
                            idx = random.randint(0, len(candidate) - 1)
                            i, j = random.sample(range(N), 2)
                            if i > j:
                                i, j = j, i
                            candidate[idx] = (i, j)
                if _verify_network(candidate, sample_size=1000):
                    if _verify_network(candidate):
                        best = candidate
                        best_score = len(best)
                        improved = True
                        break
    
    return best, best_score

def _genetic_algorithm(comparators, time_limit=40):
    """Genetic algorithm to improve the network."""
    start_time = time.time()
    population = [list(comparators)]
    
    # Add some diverse individuals
    for _ in range(10):
        if time.time() > DEADLINE - 5:
            break
        # Random mutation of best
        candidate = list(comparators)
        for _ in range(random.randint(1, 5)):
            op = random.randint(0, 3)
            n = len(candidate)
            if op == 0 and n > 0:
                idx = random.randint(0, n - 1)
                candidate.pop(idx)
            elif op == 1:
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                idx = random.randint(0, len(candidate))
                candidate.insert(idx, (i, j))
            elif op == 2 and n > 0:
                idx = random.randint(0, n - 1)
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                candidate[idx] = (i, j)
            elif op == 3 and n >= 2:
                a, b = random.sample(range(n), 2)
                candidate[a], candidate[b] = candidate[b], candidate[a]
        if _verify_network(candidate, sample_size=500):
            population.append(candidate)
    
    # Also add odd-even transposition as baseline
    oet = [(i, i + 1) for phase in range(N)
           for i in range(phase % 2, N - 1, 2)]
    if _verify_network(oet, sample_size=500):
        population.append(oet)
    
    # Evaluate and keep only valid
    valid_pop = []
    for comp in population:
        if time.time() > DEADLINE - 5:
            break
        if _verify_network(comp, sample_size=1000):
            if _verify_network(comp):
                valid_pop.append(comp)
    
    if not valid_pop:
        return comparators, len(comparators)
    
    valid_pop.sort(key=lambda x: len(x))
    best = valid_pop[0]
    best_score = len(best)
    
    pop_size = 20
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        if time.time() > DEADLINE - 2:
            break
        
        new_pop = []
        for _ in range(pop_size):
            if time.time() > DEADLINE - 2:
                break
            
            # Tournament selection
            sample = random.sample(valid_pop, min(3, len(valid_pop)))
            sample.sort(key=lambda x: len(x))
            parent = sample[0]
            
            # Crossover with another parent
            if len(valid_pop) > 1 and random.random() < 0.6:
                sample2 = random.sample(valid_pop, min(3, len(valid_pop)))
                sample2.sort(key=lambda x: len(x))
                parent2 = sample2[0]
                
                # Uniform crossover
                if len(parent) != len(parent2):
                    shorter = parent if len(parent) < len(parent2) else parent2
                    longer = parent2 if len(parent) < len(parent2) else parent
                    while len(shorter) < len(longer):
                        i, j = random.sample(range(N), 2)
                        if i > j:
                            i, j = j, i
                        shorter.append((i, j))
                    parent, parent2 = shorter, longer
                
                child = []
                for a, b in zip(parent, parent2):
                    child.append(a if random.random() < 0.5 else b)
            else:
                child = list(parent)
            
            # Mutation
            for _ in range(max(1, int(len(child) * 0.1))):
                op = random.randint(0, 3)
                n = len(child)
                if op == 0 and n > 0:
                    idx = random.randint(0, n - 1)
                    child.pop(idx)
                elif op == 1:
                    i, j = random.sample(range(N), 2)
                    if i > j:
                        i, j = j, i
                    idx = random.randint(0, len(child))
                    child.insert(idx, (i, j))
                elif op == 2 and n > 0:
                    idx = random.randint(0, n - 1)
                    i, j = random.sample(range(N), 2)
                    if i > j:
                        i, j = j, i
                    child[idx] = (i, j)
                elif op == 3 and n >= 2:
                    a, b = random.sample(range(n), 2)
                    child[a], child[b] = child[b], child[a]
            
            # Validate
            if _verify_network(child, sample_size=500):
                if _verify_network(child):
                    new_pop.append(child)
        
        # Add new valid individuals
        valid_pop.extend(new_pop)
        valid_pop.sort(key=lambda x: len(x))
        valid_pop = valid_pop[:pop_size]
        
        if valid_pop and len(valid_pop[0]) < best_score:
            best_score = len(valid_pop[0])
            best = valid_pop[0]
    
    return best, best_score

def construct():
    # Start with odd-even transposition (120 comparators)
    best = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    best_score = len(best)
    
    # Try Batcher's merge sort network
    if time.time() < DEADLINE - 15:
        try:
            batcher = _odd_even_merge_sort_network(N)
            if _verify_network(batcher, sample_size=1000):
                if _verify_network(batcher):
                    if len(batcher) < best_score:
                        best = batcher
                        best_score = len(batcher)
        except Exception:
            pass
    
    # Remove redundant comparators
    if time.time() < DEADLINE - 10:
        try:
            reduced = _remove_redundant_comparators(best)
            if len(reduced) < best_score:
                best = reduced
                best_score = len(best)
        except Exception:
            pass
    
    # Local search
    if time.time() < DEADLINE - 10:
        try:
            best, best_score = _local_search(best, time_limit=15)
        except Exception:
            pass
    
    # Genetic algorithm
    if time.time() < DEADLINE - 5:
        try:
            best, best_score = _genetic_algorithm(best, time_limit=30)
        except Exception:
            pass
    
    # Final: try to remove redundant comparators
    if time.time() < DEADLINE - 2:
        try:
            reduced = _remove_redundant_comparators(best)
            if len(reduced) < best_score:
                best = reduced
                best_score = len(best)
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
