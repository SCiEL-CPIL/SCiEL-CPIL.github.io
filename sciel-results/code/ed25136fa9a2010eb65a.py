"""Covering code K_3(7,1) via Iterated Local Search with bitset acceleration."""
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
    
    # Precompute ball masks
    ball_masks = [0] * total
    for center_idx in range(total):
        # Extract digits
        x = center_idx
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        # digits[0] is LSB (pos n-1 in standard notation if 0-indexed from left)
        # powers[0] is q^(n-1) (MSB weight)
        # Wait, the original code did:
        # digits.append(x % q) -> LSB first
        # digits.reverse() -> MSB first
        # powers[0] = q^(n-1)
        # So digits[0] (MSB) corresponds to powers[0].
        
        # My extraction above:
        # digits[0] = center_idx % q (LSB)
        # This corresponds to power q^0.
        # But powers[0] is q^(n-1).
        # So I need to be careful.
        
        # Let's stick to the original logic to be safe.
        x = center_idx
        temp_digits = []
        for _ in range(n):
            temp_digits.append(x % q)
            x //= q
        temp_digits.reverse() # Now temp_digits[0] is MSB
        
        mask = 1 << center_idx
        if radius >= 1:
            for pos in range(n):
                for alt in range(q):
                    if alt == temp_digits[pos]:
                        continue
                    new_idx = center_idx + (alt - temp_digits[pos]) * powers[pos]
                    mask |= (1 << new_idx)
        ball_masks[center_idx] = mask
        
    all_mask = (1 << total) - 1
    best_code = None
    best_size = total # Upper bound
    
    def local_search(selected):
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow removal without index shifting issues
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                others_mask = 0
                for j in range(len(selected)):
                    if j != i:
                        others_mask |= ball_masks[selected[j]]
                # Check if c covers anything unique
                unique = ball_masks[c] & ~others_mask
                if unique == 0:
                    selected.pop(i)
                    changed = True
                    break # Restart the loop as indices changed
        return selected

    def greedy(uncovered_mask, initial_selected):
        selected = initial_selected[:]
        selected_set = set(selected)
        uncovered = uncovered_mask
        
        while uncovered != 0:
            best_count = -1
            best_candidates = []
            
            for c in range(total):
                if c in selected_set:
                    continue
                covered = ball_masks[c] & uncovered
                cnt = covered.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [c]
                elif cnt == best_count and cnt > 0:
                    best_candidates.append(c)
                    
            if not best_candidates:
                break
                
            best_idx = random.choice(best_candidates)
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
        return selected if uncovered == 0 else None

    # Main Loop
    for restart in range(2000):
        if time.time() > deadline - 10:
            break
            
        # Phase 1: Pure Greedy
        uncovered = all_mask
        selected = greedy(uncovered, [])
        
        if selected is None:
            continue
            
        selected = local_search(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
            
        # Phase 2: Perturbation (if we have a best code)
        if best_code is not None and len(best_code) > 0:
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed version
            perturbed = best_code[:]
            # Remove 1 or 2 random codewords
            num_remove = random.choice([1, 2])
            for _ in range(num_remove):
                if not perturbed: break
                idx_to_remove = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx_to_remove)
            
            # Calculate uncovered mask
            covered_mask = 0
            for c in perturbed:
                covered_mask |= ball_masks[c]
            uncovered = all_mask & ~covered_mask
            
            # Repair with greedy
            repaired = greedy(uncovered, perturbed)
            
            if repaired is not None:
                repaired = local_search(repaired)
                if len(repaired) < best_size:
                    best_size = len(repaired)
                    best_code = repaired[:]

    # Convert best_code to list of lists
    if best_code is None:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
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
