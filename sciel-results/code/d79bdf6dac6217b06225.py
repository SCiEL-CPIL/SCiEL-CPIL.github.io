"""Two Sidon sets with disjoint difference sets – ILS with Bitsets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin

def _rebuild_bits(elems):
    """Rebuild sum_bits and diff_bits for a list of elements."""
    sum_bits = 0
    for a, b in combinations_with_replacement(elems, 2):
        sum_bits |= (1 << (a + b))
    diff_bits = 0
    for a, b in combinations(elems, 2):
        diff_bits |= (1 << abs(a - b))
    return sum_bits, diff_bits

def _can_add(elem, elems, sum_bits, elem_mask, other_diff_bits):
    """
    Check if elem can be added to the set defined by elems, sum_bits, elem_mask.
    other_diff_bits is the diff mask of the OTHER set.
    """
    # 1. Check sum collisions
    # New sums: 2*elem and elem + a for all a in elems
    # Check 2*elem
    if (sum_bits >> (elem + elem)) & 1:
        return False
    
    # Check elem + a
    # The mask of (elem + a) for all a in elems is (elem_mask << elem)
    # We need to check if any of these bits are set in sum_bits
    new_sums_mask = elem_mask << elem
    if sum_bits & new_sums_mask:
        return False

    # 2. Check difference disjointness with the OTHER set
    # New differences: |elem - a| for all a in elems
    # We need to check if any of these are in other_diff_bits
    for a in elems:
        d = abs(elem - a)
        if (other_diff_bits >> d) & 1:
            return False
            
    return True

def _greedy_build(random_order=None):
    """Build sets A and B using a greedy approach."""
    if random_order is None:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
    
    A = []
    B = []
    sum_A = 0
    sum_B = 0
    diff_A = 0
    diff_B = 0
    mask_A = 0
    mask_B = 0
    
    for x in random_order:
        # Try adding to A
        if len(A) < MAX_EACH and _can_add(x, A, sum_A, mask_A, diff_B):
            A.append(x)
            mask_A |= (1 << x)
            # Update sum_A
            sum_A |= (1 << (x + x))
            for a in A[:-1]:
                sum_A |= (1 << (x + a))
            # Update diff_A
            for a in A[:-1]:
                diff_A |= (1 << abs(x - a))
        # Try adding to B
        elif len(B) < MAX_EACH and _can_add(x, B, sum_B, mask_B, diff_A):
            B.append(x)
            mask_B |= (1 << x)
            # Update sum_B
            sum_B |= (1 << (x + x))
            for b in B[:-1]:
                sum_B |= (1 << (x + b))
            # Update diff_B
            for b in B[:-1]:
                diff_B |= (1 << abs(x - b))
        
    return A, B

def _local_search(A, B):
    """Perform local search to improve the solution."""
    A = A[:]
    B = B[:]
    sum_A, diff_A = _rebuild_bits(A)
    sum_B, diff_B = _rebuild_bits(B)
    mask_A = 0
    for a in A: mask_A |= (1 << a)
    mask_B = 0
    for b in B: mask_B |= (1 << b)
    
    improved = True
    while improved:
        improved = False
        # Try adding elements
        for x in range(1, N + 1):
            if (mask_A >> x) & 1 or (mask_B >> x) & 1:
                continue
            # Try A
            if len(A) < MAX_EACH and _can_add(x, A, sum_A, mask_A, diff_B):
                A.append(x)
                mask_A |= (1 << x)
                sum_A |= (1 << (x + x))
                for a in A[:-1]:
                    sum_A |= (1 << (x + a))
                for a in A[:-1]:
                    diff_A |= (1 << abs(x - a))
                improved = True
                break # Restart loop with new state
            # Try B
            if len(B) < MAX_EACH and _can_add(x, B, sum_B, mask_B, diff_A):
                B.append(x)
                mask_B |= (1 << x)
                sum_B |= (1 << (x + x))
                for b in B[:-1]:
                    sum_B |= (1 << (x + b))
                for b in B[:-1]:
                    diff_B |= (1 << abs(x - b))
                improved = True
                break
        
        # If no additions, try swaps
        if not improved:
            # Try swapping an element in A with an unassigned element
            for i, a in enumerate(A):
                for x in range(1, N + 1):
                    if (mask_A >> x) & 1 or (mask_B >> x) & 1:
                        continue
                    # Simulate swap: remove a, add x to A
                    # We need to check if A without a plus x is valid
                    # And if the new differences don't clash with B
                    # This is complex to check in O(1). 
                    # Given time constraints, we might skip swaps in local search 
                    # and rely on perturbation + greedy rebuild.
                    pass
            
    return A, B

def construct():
    start_time = time.time()
    best_A, best_B = [], []
    best_size = 0
    
    # Initial greedy build
    A, B = _greedy_build()
    best_A, best_B = A, B
    best_size = len(A) + len(B)
    
    # ILS Loop
    current_A, current_B = A, B
    current_size = best_size
    
    while time.time() - start_time < DEADLINE:
        # Perturb
        perturbed_A = current_A[:]
        perturbed_B = current_B[:]
        
        # Remove some random elements
        if perturbed_A:
            for _ in range(random.randint(1, 3)):
                if perturbed_A:
                    idx = random.randint(0, len(perturbed_A) - 1)
                    perturbed_A.pop(idx)
        if perturbed_B:
            for _ in range(random.randint(1, 3)):
                if perturbed_B:
                    idx = random.randint(0, len(perturbed_B) - 1)
                    perturbed_B.pop(idx)
        
        # Rebuild greedily from perturbed state
        # We need a greedy that starts from existing sets
        # Let's just do a fresh greedy with a random order, but bias towards keeping existing elements?
        # Simpler: Just run _greedy_build with a random order.
        # To be more effective, we can try to keep the "core" of the previous solution.
        
        # Let's try a different perturbation: Random restart with random order
        A_new, B_new = _greedy_build()
        
        # Local Search
        A_ls, B_ls = _local_search(A_new, B_new)
        
        size = len(A_ls) + len(B_ls)
        if size > best_size:
            best_size = size
            best_A, best_B = A_ls, B_ls
            current_A, current_B = A_ls, B_ls
            current_size = size
            
        # If no improvement for a while, we could try stronger perturbation, 
        # but simple random restarts with local search is often sufficient.
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
