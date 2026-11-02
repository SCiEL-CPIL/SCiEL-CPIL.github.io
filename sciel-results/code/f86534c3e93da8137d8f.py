"""Covering code K_3(7,1) via greedy set cover with bitset acceleration and time guard."""
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
    
    # Number of restarts
    # We will run multiple random restarts of the greedy algorithm
    # and then apply a local search (removal of redundant codewords)
    
    # Estimate time per restart
    # Greedy: ~200 iterations * 2187 candidates * bit_count
    # Local search: ~200^2 bit ops
    # Should be fast enough for ~50 restarts
    
    for restart in range(100):
        if time.time() > deadline - 10:
            break
            
        # Randomize the greedy choice among ties
        # We do this by shuffling the remaining candidates list initially?
        # No, we need to pick the one with max coverage.
        # If there are ties, we pick randomly.
        
        uncovered = all_mask
        selected = []
        remaining = list(range(total))
        
        # Shuffle remaining to ensure we don't always pick the same index in case of ties?
        # Actually, the loop `for c in remaining` will find the max.
        # If multiple have the same max, we need to pick one.
        # We can collect all candidates with max coverage and pick one randomly.
        
        while uncovered != 0:
            if time.time() > deadline - 5:
                break
                
            best_count = -1
            best_candidates = []
            
            for c in remaining:
                covered = ball_masks[c] & uncovered
                cnt = bin(covered).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [c]
                elif cnt == best_count and cnt > 0:
                    best_candidates.append(c)
                    
            if not best_candidates:
                break
                
            # Pick randomly from best candidates
            best_idx = random.choice(best_candidates)
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            remaining.remove(best_idx)
            
        if uncovered != 0:
            continue # Failed to cover, skip this restart
            
        # Local Search: Remove redundant codewords
        # A codeword c is redundant if its ball is covered by the union of balls of other selected codewords.
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow removal
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                c = selected[i]
                # Compute coverage of others
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
                    
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
            
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
