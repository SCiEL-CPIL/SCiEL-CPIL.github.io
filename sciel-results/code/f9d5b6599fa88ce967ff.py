"""Covering code K_3(7,1) via greedy set cover with bitset acceleration and local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

# Deadline: evaluator kills at 150s; we return well before that.
_DEADLINE = 140.0  # seconds


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Total space
    total = q ** n

    # Precompute digit arrays for all words
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))

    # Precompute powers for fast index computation
    powers = [q ** (n - 1 - i) for i in range(n)]

    # Precompute ball masks: for each center, a bitset of words within distance <= radius
    # Use numpy for vectorized computation of neighbor indices
    # Ball size for q=3, n=7, r=1: 1 + 7*2 = 15 words per ball
    ball_size = 1
    if radius >= 1:
        ball_size += n * (q - 1)
    
    # Precompute all ball masks as a list of integers
    # For each center, the ball contains the center itself plus all words at distance 1
    ball_masks = [0] * total
    
    # Vectorized approach: for each position and each alternative value, compute all neighbor indices at once
    # But since we need individual bit masks, let's use a semi-vectorized approach
    # Actually, for 2187 centers, the loop is manageable. Let's optimize the inner loop.
    
    for center_idx in range(total):
        center = all_words[center_idx]
        # Start with center bit
        mask = 1 << center_idx
        if radius >= 1:
            for pos in range(n):
                cur_val = center[pos]
                # For each alternative value at this position
                for alt in range(q):
                    if alt == cur_val:
                        continue
                    new_idx = center_idx + (alt - cur_val) * powers[pos]
                    mask |= (1 << new_idx)
        ball_masks[center_idx] = mask

    # Greedy set cover with numpy-accelerated coverage counting
    all_mask = (1 << total) - 1
    uncovered = all_mask
    selected = []
    remaining = list(range(total))
    remaining_set = set(range(total))
    
    # For faster iteration, use a list and remove by swapping
    remaining_indices = list(range(total))
    rem_pos = {i: i for i in range(total)}  # position of i in remaining_indices
    
    while uncovered != 0:
        if time.time() > deadline - 10:
            break

        best_count = -1
        best_idx = -1
        best_pos_in_rem = -1
        
        # Iterate through remaining candidates
        for pos in range(len(remaining_indices)):
            c = remaining_indices[pos]
            covered = ball_masks[c] & uncovered
            cnt = covered.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = c
                best_pos_in_rem = pos

        if best_idx == -1 or best_count == 0:
            break

        selected.append(best_idx)
        uncovered &= ~ball_masks[best_idx]
        
        # Remove best_idx from remaining_indices by swapping with last element
        last_pos = len(remaining_indices) - 1
        removed_idx = remaining_indices[best_pos_in_rem]
        remaining_indices[best_pos_in_rem] = remaining_indices[last_pos]
        rem_pos[remaining_indices[best_pos_in_rem]] = best_pos_in_rem
        remaining_indices.pop()

    # If greedy didn't finish, fall back to direct sum construction
    if uncovered != 0:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Local search: try to remove redundant codewords
    # A codeword is redundant if its ball is fully covered by the union of other balls
    # Use a more efficient approach: maintain coverage counts
    for _ in range(5):  # Multiple passes of removal
        if time.time() > deadline - 5:
            break
        removed_any = False
        # Try removing each codeword, starting from the end
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Check if selected[i] is redundant
            # Compute union of all other balls
            union_mask = 0
            for j in range(len(selected)):
                if j != i:
                    union_mask |= ball_masks[selected[j]]
            # If ball_masks[selected[i]] is subset of union_mask, it's redundant
            if ball_masks[selected[i]] & ~union_mask == 0:
                selected.pop(i)
                removed_any = True
                break  # Restart the loop since indices changed
        if not removed_any:
            break

    # Local search: try to swap codewords for better coverage
    # Try random restarts with different orderings
    best_solution = selected[:]
    best_size = len(selected)

    # Try several random restarts with different orderings
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            random.seed(42 + restart * 7)
            # Shuffle the order of candidates
            shuffled = list(range(total))
            random.shuffle(shuffled)

            uncovered_r = all_mask
            selected_r = []
            remaining_r_indices = list(shuffled)
            rem_pos_r = {i: i for i in range(total)}

            while uncovered_r != 0:
                if time.time() > deadline - 10:
                    break
                best_count = -1
                best_idx = -1
                best_pos_in_rem = -1
                
                for pos in range(len(remaining_r_indices)):
                    c = remaining_r_indices[pos]
                    covered = ball_masks[c] & uncovered_r
                    cnt = covered.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = c
                        best_pos_in_rem = pos

                if best_idx == -1 or best_count == 0:
                    break
                selected_r.append(best_idx)
                uncovered_r &= ~ball_masks[best_idx]
                
                # Remove best_idx from remaining_r_indices by swapping with last element
                last_pos = len(remaining_r_indices) - 1
                remaining_r_indices[best_pos_in_rem] = remaining_r_indices[last_pos]
                rem_pos_r[remaining_r_indices[best_pos_in_rem]] = best_pos_in_rem
                remaining_r_indices.pop()

            if uncovered_r == 0 and len(selected_r) < best_size:
                best_solution = selected_r
                best_size = len(selected_r)
                # Also run redundancy removal on this solution
                for _ in range(3):
                    if time.time() > deadline - 5:
                        break
                    removed_any = False
                    for i in range(len(best_solution) - 1, -1, -1):
                        if time.time() > deadline - 5:
                            break
                        union_mask = 0
                        for j in range(len(best_solution)):
                            if j != i:
                                union_mask |= ball_masks[best_solution[j]]
                        if ball_masks[best_solution[i]] & ~union_mask == 0:
                            best_solution.pop(i)
                            best_size = len(best_solution)
                            removed_any = True
                            break
                    if not removed_any:
                        break

    # Convert best_solution indices back to lists of digits
    result = []
    for idx in best_solution:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        result.append(digits)

    return result
# EVOLVE-BLOCK-END
