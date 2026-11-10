# EVOLVE-BLOCK-START
import random
import itertools
import time

def _is_sidon(subset):
    """Check if a list of distinct ints is a Sidon set (pair sums unique, incl. diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _find_one_sidon_subset(A, k):
    """Find one Sidon subset of size k in A. Returns None if not found."""
    for subset in itertools.combinations(A, k):
        if _is_sidon(list(subset)):
            return list(subset)
    return None

def _compute_max_sidon_size(A):
    """Compute the maximum size of a Sidon subset of A."""
    for k in range(len(A), 1, -1):
        if _find_one_sidon_subset(A, k) is not None:
            return k
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    best_set = list(range(12))
    best_max_size = _compute_max_sidon_size(best_set)
    
    # If best is already 3, we can try to improve to 2, but 3 is likely the floor for random-ish sets.
    # Let's aim for 3.
    
    # Strategy: Random restarts with local search to eliminate 4-Sidon subsets.
    for restart in range(2000):
        if time.time() > deadline:
            break
        
        # Initialize with a dense set to encourage collisions
        # Dense sets have more sum collisions, making it harder to be Sidon.
        base = random.randint(0, 100)
        span = random.randint(12, 25)
        A = random.sample(range(base, min(base + span + 1, 129)), 12)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Check for Sidon subset of size 4
            sidon4 = _find_one_sidon_subset(A, 4)
            if sidon4 is None:
                # Max Sidon size is <= 3
                max_sz = _compute_max_sidon_size(A)
                if max_sz < best_max_size:
                    best_max_size = max_sz
                    best_set = A[:]
                    if best_max_size == 2:
                        return best_set
                break
            
            # Found a Sidon subset of size 4. Break it.
            # Pick a random element from the subset to replace
            idx_to_replace = random.choice(range(4))
            elem_to_replace = sidon4[idx_to_replace]
            
            # Find index in A
            try:
                pos = A.index(elem_to_replace)
            except ValueError:
                continue
            
            # Try to replace with a value that creates a collision in the subset
            # We want the new set {new_val, s1, s2, s3} to NOT be Sidon.
            # Let other 3 be O.
            others = [x for x in sidon4 if x != elem_to_replace]
            
            # Generate candidate replacements
            # Candidates should be close to existing values to maintain density/collisions
            candidates = []
            for x in A:
                for delta in range(-5, 6):
                    nx = x + delta
                    if 0 <= nx <= 128 and nx not in A:
                        candidates.append(nx)
            
            random.shuffle(candidates)
            
            found_fix = False
            for new_val in candidates[:20]:
                # Construct test subset
                test_subset = others + [new_val]
                if not _is_sidon(test_subset):
                    # This replacement breaks the Sidon property of this specific 4-set
                    # Check if this move is valid (doesn't introduce other issues? 
                    # We don't care about other 4-sets yet, just break this one)
                    A[pos] = new_val
                    improved = True
                    found_fix = True
                    break
            
            if not found_fix:
                # If no simple fix found, try a random move
                available = [x for x in range(129) if x not in A]
                if available:
                    A[pos] = random.choice(available)
                    improved = True # Force another iteration to re-check

        # After local search, evaluate
        max_sz = _compute_max_sidon_size(A)
        if max_sz < best_max_size:
            best_max_size = max_sz
            best_set = A[:]
            if best_max_size == 2:
                return best_set

    return best_set
# EVOLVE-BLOCK-END
