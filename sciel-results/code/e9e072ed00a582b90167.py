"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: all words (valid but large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    total_words = 3 ** 7  # 2187

    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(3), repeat=7)]
    word_index = {w: i for i, w in enumerate(all_words)}

    # Precompute balls as index arrays for fast numpy operations
    # Ball size: 1 + 7*2 = 15
    balls_indices = np.zeros((total_words, 15), dtype=np.int32)
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(word_index[tuple(new_w)])
        balls_indices[i] = ball

    # Greedy set cover using numpy for speed
    covered = np.zeros(total_words, dtype=np.int32)
    chosen = []
    uncovered_count = total_words

    while uncovered_count > 0 and time.time() < deadline - 30:
        # Compute gain for each word: number of uncovered words in its ball
        gains = np.zeros(total_words, dtype=np.int32)
        for i in range(total_words):
            ball_indices = balls_indices[i]
            gains[i] = np.sum(covered[ball_indices] == 0)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
        
        chosen.append(best_idx)
        
        # Update coverage
        ball_indices = balls_indices[best_idx]
        for idx in ball_indices:
            if covered[idx] == 0:
                uncovered_count -= 1
            covered[idx] += 1

    # Local optimization: try to remove redundant codewords
    # Then try swaps: remove one, add another that covers what was lost
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # First pass: try removing each codeword if redundant
        for i in range(len(chosen)):
            if time.time() > deadline - 10:
                break
            # Check if codeword i is redundant
            ball_i = balls_indices[chosen[i]]
            redundant = True
            for idx in ball_i:
                if covered[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for idx in ball_i:
                    covered[idx] -= 1
                chosen.pop(i)
                improved = True
                break

        # Second pass: try swaps
        if not improved:
            for i in range(len(chosen)):
                if time.time() > deadline - 10:
                    break
                # Temporarily remove codeword i
                ball_i = balls_indices[chosen[i]]
                lost = []
                for idx in ball_i:
                    if covered[idx] == 1:
                        lost.append(idx)
                
                if not lost:
                    # Redundant, remove
                    for idx in ball_i:
                        covered[idx] -= 1
                    chosen.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement that covers all lost words
                lost_set = set(lost)
                best_replace = None
                best_new_gain = 0
                
                for j in range(total_words):
                    if j in chosen:
                        continue
                    ball_j = balls_indices[j]
                    # Check if ball_j covers all lost words
                    covers_all = all(idx in set(ball_j) for idx in lost)
                    if not covers_all:
                        continue
                    # Count new coverage
                    new_gain = sum(1 for idx in ball_j if covered[idx] == 0 and idx not in lost_set)
                    if new_gain > best_new_gain:
                        best_new_gain = new_gain
                        best_replace = j
                
                if best_replace is not None:
                    # Perform swap
                    for idx in ball_i:
                        covered[idx] -= 1
                    chosen[i] = best_replace
                    for idx in balls_indices[best_replace]:
                        covered[idx] += 1
                    improved = True
                    break

    # Convert to lists
    return [list(all_words[idx]) for idx in chosen]
# EVOLVE-BLOCK-END
