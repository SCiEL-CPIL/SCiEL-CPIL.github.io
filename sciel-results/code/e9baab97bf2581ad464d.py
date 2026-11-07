"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

N = 128
MAX_VAL = 128

def _is_valid_mask(mask):
    """
    Check if the set represented by the bitmask violates the condition.
    Bit i represents integer i+1 (1..128).
    Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
    
    Optimized: For each pair (a,b) with a+b in S, check if there's a c != a,b
    such that a+c and b+c are both in S.
    """
    # Iterate over potential pairs (a, b) in S
    # If a+b is in S, check if there exists c in S (c != a, c != b) 
    # such that a+c in S and b+c in S.
    
    # Optimization: Pre-extract set bits to iterate only over present elements
    # This is faster when the set is not full, but for ~80 elements, 
    # the overhead of list creation might be comparable to bit checks.
    # We'll stick to the bit-checking loop which is very tight.
    
    for i in range(N):
        if not (mask >> i) & 1:
            continue
        a = i + 1
        # b must satisfy a+b <= 128
        max_j = N - 1 - i # since b = j+1, a+b = i+1+j+1 = i+j+2 <= 128 => i+j <= 126 => j <= 126-i
        # Actually j goes up to N-1 (value 128).
        # a+b = i+1 + j+1 = i+j+2.
        # We need a+b in S, so a+b <= 128.
        # i+j+2 <= 128 => i+j <= 126.
        # So j <= 126 - i.
        # Also j > i.
        for j in range(i + 1, min(N, 127 - i)):
            if not (mask >> j) & 1:
                continue
            b = j + 1
            
            # Check if a+b is in S
            # a+b = i+j+2. Bit index is (i+j+2)-1 = i+j+1.
            ab_idx = i + j + 1
            if ab_idx >= N:
                continue
            if not (mask >> ab_idx) & 1:
                continue
            
            # If a+b in S, check if there exists c in S, c != a, c != b
            # such that a+c in S and b+c in S.
            # c is value k+1.
            # a+c = i+k+2 => Bit i+k+1
            # b+c = j+k+2 => Bit j+k+1
            # We need bit k in mask, mask>>(i+1), mask>>(j+1).
            # Wait, a = i+1. a+c = i+1+k+1 = i+k+2. Bit index i+k+1.
            # So we need bit (k) in (mask >> (i+1)).
            # Similarly for b.
            
            # candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            # This finds all k such that:
            # k is in S (c in S)
            # k + i + 1 is in S (a+c in S)
            # k + j + 1 is in S (b+c in S)
            
            candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            
            # We need to ensure c != a and c != b.
            # c = k+1.
            # a = i+1 => k != i
            # b = j+1 => k != j
            
            # Mask out bits i and j from candidates
            if candidates & ~((1 << i) | (1 << j)):
                return False
    return True

def _construct_baseline():
    """
    Construct the known strong baseline: {65..128} U {1,3,...,31, 64}.
    """
    mask = 0
    # Add 65..128
    for v in range(65, 129):
        mask |= (1 << (v - 1))
    
    # Add specific small numbers
    for v in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 64]:
        mask |= (1 << (v - 1))
        
    return mask

def _greedy_fill(mask, order=None):
    """
    Try to add elements from 'order' to mask if valid.
    """
    if order is None:
        order = list(range(1, N + 1))
        
    for v in order:
        bit = 1 << (v - 1)
        if mask & bit:
            continue
        trial = mask | bit
        if _is_valid_mask(trial):
            mask = trial
    return mask

def _local_search(mask, deadline):
    """
    Local search to improve mask size.
    Strategy: 
    1. Try adding any single missing element (multiple passes).
    2. Try 'remove one, add many' moves.
    3. Try 'remove two, add many' moves for escaping local minima.
    """
    current_mask = mask
    current_size = current_mask.bit_count()
    
    # First, do multiple passes of single-element additions
    while time.time() < deadline:
        improved = False
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if current_mask & bit:
                continue
            trial = current_mask | bit
            if _is_valid_mask(trial):
                current_mask = trial
                current_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Now try exchange moves
    while time.time() < deadline:
        found_better = False
        best_candidate = current_mask
        best_candidate_size = current_size
        
        # Get list of elements in current set
        elems = [i + 1 for i in range(N) if (current_mask >> i) & 1]
        
        # Phase 2: Remove one, add many
        for v in elems:
            if time.time() > deadline:
                break
            bit = 1 << (v - 1)
            reduced_mask = current_mask & ~bit
            
            # Greedy fill
            candidate = reduced_mask
            for add_v in range(1, N + 1):
                add_bit = 1 << (add_v - 1)
                if candidate & add_bit:
                    continue
                trial = candidate | add_bit
                if _is_valid_mask(trial):
                    candidate = trial
            
            if candidate.bit_count() > best_candidate_size:
                best_candidate = candidate
                best_candidate_size = candidate.bit_count()
                found_better = True
        
        # Phase 3: Remove two, add many (sampled for larger sets)
        if not found_better and len(elems) >= 2:
            # Sample pairs if set is large
            if len(elems) > 20:
                num_samples = min(200, len(elems) * (len(elems) - 1) // 2)
                for _ in range(num_samples):
                    if time.time() > deadline:
                        break
                    a, b = random.sample(elems, 2)
                    reduced = current_mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                    candidate = reduced
                    for add_v in range(1, N + 1):
                        add_bit = 1 << (add_v - 1)
                        if candidate & add_bit:
                            continue
                        trial = candidate | add_bit
                        if _is_valid_mask(trial):
                            candidate = trial
                    if candidate.bit_count() > best_candidate_size:
                        best_candidate = candidate
                        best_candidate_size = candidate.bit_count()
                        found_better = True
            else:
                # Try all pairs for small sets
                for i in range(len(elems)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(elems)):
                        a, b = elems[i], elems[j]
                        reduced = current_mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                        candidate = reduced
                        for add_v in range(1, N + 1):
                            add_bit = 1 << (add_v - 1)
                            if candidate & add_bit:
                                continue
                            trial = candidate | add_bit
                            if _is_valid_mask(trial):
                                candidate = trial
                        if candidate.bit_count() > best_candidate_size:
                            best_candidate = candidate
                            best_candidate_size = candidate.bit_count()
                            found_better = True
        
        if found_better:
            current_mask = best_candidate
            current_size = best_candidate_size
            # After improvement, try single additions again
            while time.time() < deadline:
                improved = False
                for v in range(1, N + 1):
                    bit = 1 << (v - 1)
                    if current_mask & bit:
                        continue
                    trial = current_mask | bit
                    if _is_valid_mask(trial):
                        current_mask = trial
                        current_size += 1
                        improved = True
                        break
                if not improved:
                    break
        else:
            break
    
    return current_mask

def construct():
    start_time = time.time()
    # Leave 10 seconds margin for safety (evaluator kills at 90s)
    deadline = start_time + 78.0
    
    # 1. Start with strong baseline
    best_mask = _construct_baseline()
    best_size = best_mask.bit_count()
    
    # 2. Greedy fill with multiple orderings
    if time.time() < deadline - 15:
        # Try ascending order
        filled_mask = _greedy_fill(best_mask, list(range(1, N + 1)))
        if filled_mask.bit_count() > best_size:
            best_mask = filled_mask
            best_size = best_mask.bit_count()
        
        # Try descending order
        if time.time() < deadline - 12:
            filled_desc = _greedy_fill(best_mask, list(range(N, 0, -1)))
            if filled_desc.bit_count() > best_size:
                best_mask = filled_desc
                best_size = best_mask.bit_count()
        
        # Try a few random orderings
        if time.time() < deadline - 10:
            for seed in range(5):
                if time.time() > deadline - 10:
                    break
                random.seed(seed)
                perm = list(range(1, N + 1))
                random.shuffle(perm)
                filled_r = _greedy_fill(best_mask, perm)
                if filled_r.bit_count() > best_size:
                    best_mask = filled_r
                    best_size = best_mask.bit_count()
    
    # 3. Local search
    if time.time() < deadline - 8:
        searched_mask = _local_search(best_mask, deadline - 8)
        if searched_mask.bit_count() > best_size:
            best_mask = searched_mask
            best_size = searched_mask.bit_count()
            
    # 4. Iterative Local Search (ILS) with perturbation
    # This is the main improvement: keep perturbing the best solution and refining.
    if time.time() < deadline - 2:
        current_best = best_mask
        current_best_size = best_size
        stagnation = 0
        
        while time.time() < deadline - 2:
            # Perturbation: remove k random elements
            # k is adaptive: start small, increase if stagnating
            elems = [i + 1 for i in range(N) if (current_best >> i) & 1]
            if len(elems) < 3:
                break
            
            k = random.randint(2, 5)
            if stagnation > 10:
                k = random.randint(5, 10)
            
            to_remove = random.sample(elems, min(k, len(elems)))
            perturbed = current_best
            for r in to_remove:
                perturbed &= ~(1 << (r - 1))
            
            # Greedy fill with random ordering
            random.seed(int(time.time() * 1000) % 100000)
            perm = list(range(1, N + 1))
            random.shuffle(perm)
            filled_p = _greedy_fill(perturbed, perm)
            
            # Local search
            if time.time() < deadline - 3:
                searched_p = _local_search(filled_p, deadline - 3)
                
                if searched_p.bit_count() > current_best_size:
                    current_best = searched_p
                    current_best_size = searched_p.bit_count()
                    stagnation = 0
                else:
                    stagnation += 1
                    
                # Update global best
                if current_best_size > best_size:
                    best_mask = current_best
                    best_size = current_best_size
            else:
                break
                
    # Final validity check
    if not _is_valid_mask(best_mask):
        # Fallback to baseline
        best_mask = _construct_baseline()
        
    # Convert mask to list of ints
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
