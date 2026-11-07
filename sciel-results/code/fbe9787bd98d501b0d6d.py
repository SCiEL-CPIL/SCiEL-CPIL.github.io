"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128

def construct():
    start_time = time.time()
    # Evaluator kills at 90s; use 55s budget to be safe
    deadline = start_time + 55.0

    # Precompute: for each pair (a, b) with a < b and a+b <= N,
    # store the bitmask of c values (c > b) such that a+c <= N and b+c <= N.
    # If a, b, a+b are all in the set, then having any such c in the set
    # creates a forbidden triple.
    forbidden_c_masks = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s_ab = a + b
            if s_ab > N:
                break
            max_c = min(N - a, N - b)
            if max_c <= b:
                continue
            c_mask = 0
            for c in range(b + 1, max_c + 1):
                c_mask |= (1 << c)
            if c_mask != 0:
                forbidden_c_masks[(a, b)] = c_mask

    # Also precompute for pairs where a+b > N (no sum in range, so no constraint from that pair)
    # These don't need checking.

    def _is_valid_mask(mask):
        """
        Check if the set represented by the bitmask violates the condition.
        Bit i represents integer i (1..128). Note: bit 0 is unused.
        """
        for (a, b), c_mask in forbidden_c_masks.items():
            # Check if a, b, a+b are all in mask
            if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> (a + b)) & 1):
                # Check if any c in c_mask is also in mask
                if (mask & c_mask) != 0:
                    return False
        return True

    def _construct_baseline():
        """
        Construct the known strong baseline: {65..128} U {1,3,...,31, 64}.
        """
        mask = 0
        # Add 65..128
        for v in range(65, 129):
            mask |= (1 << v)
        
        # Add specific small numbers
        for v in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 64]:
            mask |= (1 << v)
            
        return mask

    def _greedy_fill(mask, order=None):
        """
        Try to add elements from 'order' to mask if valid.
        """
        if order is None:
            order = list(range(1, N + 1))
            
        for v in order:
            bit = 1 << v
            if mask & bit:
                continue
            trial = mask | bit
            if _is_valid_mask(trial):
                mask = trial
        return mask

    def _local_search(mask, deadline):
        """
        Local search to improve mask size.
        """
        current_mask = mask
        current_size = current_mask.bit_count()
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Try adding single elements
            for v in range(1, N + 1):
                bit = 1 << v
                if current_mask & bit:
                    continue
                trial = current_mask | bit
                if _is_valid_mask(trial):
                    current_mask = trial
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                continue
                
            # Phase 2: Try 'remove one, add many'
            found_better = False
            best_candidate = current_mask
            best_candidate_size = current_size
            
            for v in range(1, N + 1):
                bit = 1 << v
                if not (current_mask & bit):
                    continue
                
                reduced_mask = current_mask & ~bit
                
                candidate = reduced_mask
                for add_v in range(1, N + 1):
                    add_bit = 1 << add_v
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
            else:
                break
                
        return current_mask

    # 1. Start with strong baseline
    best_mask = _construct_baseline()
    best_size = best_mask.bit_count()
    
    # 2. Greedy fill
    if time.time() < deadline - 10:
        filled_mask = _greedy_fill(best_mask, list(range(1, N + 1)))
        if filled_mask.bit_count() > best_size:
            best_mask = filled_mask
            best_size = best_mask.bit_count()
    
    # 3. Local search
    if time.time() < deadline - 5:
        searched_mask = _local_search(best_mask, deadline - 5)
        if searched_mask.bit_count() > best_size:
            best_mask = searched_mask
            best_size = searched_mask.bit_count()
            
    # 4. Random restarts if time remains
    random.seed(42)
    if time.time() < deadline - 2:
        for restart in range(50):
            if time.time() > deadline - 2:
                break
            perm = list(range(1, N + 1))
            random.shuffle(perm)
            
            # Perturbation: remove 5 random elements, then local search.
            if best_size > 0:
                elems = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
                if len(elems) >= 5:
                    to_remove = random.sample(elems, 5)
                    perturbed = best_mask
                    for r in to_remove:
                        perturbed &= ~(1 << r)
                    
                    # Greedy fill on perturbed
                    filled_p = _greedy_fill(perturbed, perm)
                    
                    # Local search on filled_p
                    searched_p = _local_search(filled_p, deadline - 2)
                    
                    if searched_p.bit_count() > best_size:
                        best_mask = searched_p
                        best_size = searched_p.bit_count()

    # Final validity check
    if not _is_valid_mask(best_mask):
        best_mask = _construct_baseline()
        
    # Convert mask to list of ints
    result = []
    for i in range(1, N + 1):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result
# EVOLVE-BLOCK-END
