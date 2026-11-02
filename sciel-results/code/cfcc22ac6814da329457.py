"""Covering code K_3(7,1) via greedy set cover, redundancy removal, and swap optimization."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

# Deadline: evaluator kills at 150s; we return well before that.
_DEADLINE = 140.0  # seconds


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    total = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # Precompute all words
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))
        
    # Precompute ball masks
    ball_masks = [0] * total
    for center_idx in range(total):
        center = all_words[center_idx]
        mask = 1 << center_idx
        if radius >= 1:
            for pos in range(n):
                for alt in range(q):
                    if alt == center[pos]:
                        continue
                    new_idx = center_idx + (alt - center[pos]) * powers[pos]
                    mask |= (1 << new_idx)
        ball_masks[center_idx] = mask
        
    all_mask = (1 << total) - 1
    best_code = None
    best_size = total # Upper bound
    
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = all_mask
        selected = []
        remaining = list(range(total))
        
        while uncovered != 0:
            if time.time() > deadline - 5:
                break
                
            best_count = -1
            best_candidates = []
            
            for c in remaining:
                covered = ball_masks[c] & uncovered
                cnt = covered.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [c]
                elif cnt == best_count and cnt > 0:
                    best_candidates.append(c)
                    
            if not best_candidates:
                break
                
            # Pick randomly from best candidates to allow different paths
            best_idx = random.choice(best_candidates)
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            remaining.remove(best_idx)
            
        if uncovered != 0:
            return None
        return selected

    def local_search(code):
        code = code[:]
        changed = True
        while changed:
            changed = False
            
            # 1. Redundancy Removal
            for i in range(len(code) - 1, -1, -1):
                if time.time() > deadline - 5:
                    return code
                c = code[i]
                # Compute union of all other balls
                others_mask = 0
                for j in range(len(code)):
                    if j != i:
                        others_mask |= ball_masks[code[j]]
                
                # If c covers nothing unique, remove it
                if (ball_masks[c] & ~others_mask) == 0:
                    code.pop(i)
                    changed = True
                    break # Restart loop as indices changed
            
            # 2. Swap Optimization
            # Try to replace one codeword with another to potentially enable removal
            # We iterate through current codewords and try to swap
            for i in range(len(code)):
                if time.time() > deadline - 5:
                    return code
                
                c_out = code[i]
                # Compute mask of all others
                others_mask = 0
                for j in range(len(code)):
                    if j != i:
                        others_mask |= ball_masks[code[j]]
                
                deficit = all_mask & ~others_mask
                if deficit == 0:
                    # Should have been caught by redundancy removal, but just in case
                    code.pop(i)
                    changed = True
                    break
                
                # Look for c_in not in code that covers deficit
                # To speed up, we can check candidates more likely to be good?
                # For now, iterate all not in code.
                code_set = set(code)
                found_swap = False
                for c_in in range(total):
                    if c_in in code_set:
                        continue
                    if (deficit & ~ball_masks[c_in]) == 0:
                        # Swap
                        code[i] = c_in
                        changed = True
                        found_swap = True
                        break # Restart local search loop
                        
                if found_swap:
                    break # Restart local search loop
                    
        return code

    # Run multiple restarts
    # Estimate: Greedy is fast. Local search is slower.
    # Let's do 10-20 restarts depending on time.
    
    for restart in range(50):
        if time.time() > deadline - 15:
            break
            
        code = greedy_cover(random_seed=restart)
        if code is None:
            continue
            
        # Local search
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    # Fallback if no valid code found (should not happen)
    if best_code is None:
        # Simple greedy without randomization
        code = greedy_cover(random_seed=0)
        if code:
            best_code = code
        else:
            # Direct sum construction fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code_indices = []
            # Map tetracode words to indices
            # This fallback is complex to map to indices quickly, so we return the list of digits directly?
            # No, construct returns list of codewords (lists of ints).
            # The greedy returns indices. We need to convert.
            # If we reach here, we have a problem.
            # Let's just return the tetracode expansion directly.
            result = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    result.append(h + list(tail))
            return result

    # Convert best_code indices to lists of digits
    result = []
    for idx in best_code:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        result.append(digits)
        
    return result
# EVOLVE-BLOCK-END
