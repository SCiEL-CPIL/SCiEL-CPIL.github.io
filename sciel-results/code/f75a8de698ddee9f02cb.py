"""Ternary covering code K_3(8,1) via simulated annealing with pair swaps."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin before 150s kill

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # 17 for q=3, n=8, radius=1

    # Convert index to tuple and back
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls using vectorized numpy for speed
    # For each word i, its ball contains i and all words at Hamming distance 1
    # We precompute as a list of numpy arrays for fast iteration
    balls = []
    for i in range(total_words):
        t = idx_to_tuple(i)
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val == t[pos]:
                    continue
                new_t = list(t)
                new_t[pos] = val
                ball.append(tuple_to_idx(tuple(new_t)))
        balls.append(ball)

    # Greedy set cover to get initial solution
    # uncovered[w] = True if word w is not yet covered
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    coverage = np.zeros(total_words, dtype=np.int32)

    # For efficiency, maintain a list of uncovered words
    uncovered_list = list(range(total_words))

    while uncovered_list:
        if time.time() > deadline - 60:  # reserve time for SA
            break

        # Pick a random uncovered word to guide the greedy choice
        # Actually, better: pick the codeword that covers the most uncovered words
        # But that's O(total_words * ball_size) per step which is expensive.
        # Use a randomized greedy: pick a random uncovered word w, then pick a random
        # codeword that covers w and has high coverage of other uncovered words.

        # Simpler approach: iterate through all possible codewords, score by uncovered coverage
        # For 6561 words, this is feasible if done carefully.

        # Actually, let's use a faster heuristic:
        # Pick a random uncovered word w
        # Find all codewords that cover w (w itself and its neighbors at distance 1)
        # Among those, pick the one that covers the most currently-uncovered words

        w = random.choice(uncovered_list)
        # Candidates are: w itself and all words at distance 1 from w
        # i.e., balls[w]
        best_c = -1
        best_score = -1
        for c in balls[w]:
            # Score = number of uncovered words in balls[c]
            # Quick approximation: count uncovered in balls[c]
            # This is expensive. Let's just use a simpler scoring.
            # For now, pick randomly from candidates.
            pass

        # Simpler: just pick w itself as the codeword (covers w and its neighbors)
        # This is a valid greedy step.
        c = w
        code.append(c)
        for word in balls[c]:
            coverage[word] += 1
            if uncovered[word]:
                uncovered[word] = False
                uncovered_list.remove(word)

    # After greedy, we have a covering. Now do removal pass.
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        random.shuffle(code)
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline - 30:
                break
            c = code[i]
            can_remove = True
            for w in balls[c]:
                if coverage[w] <= 1:
                    can_remove = False
                    break
            if can_remove:
                for w in balls[c]:
                    coverage[w] -= 1
                code.pop(i)
                improved = True

    # Now simulate annealing with pair-swap moves
    # State: code (list of indices), coverage (array)
    # Energy: number of codewords (we want to minimize)
    # But we must maintain validity: all words covered.
    # Moves:
    # 1. Remove a codeword if possible (reduces energy by 1)
    # 2. Swap: remove c1, add c2, then try to remove more
    # 3. Pair swap: remove c1 and c2, add c3 and c4, if net reduction

    best_code = list(code)
    best_coverage = coverage.copy()
    best_len = len(code)

    # SA parameters
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.995

    while time.time() < deadline:
        if T < T_min:
            break

        current_len = len(code)

        # Try different move types
        move_type = random.random()

        if move_type < 0.3 and len(code) > 1:
            # Pair swap: remove two codewords, add two new ones
            # Pick two random codewords to remove
            idx1 = random.randint(0, len(code) - 1)
            idx2 = random.randint(0, len(code) - 1)
            if idx1 == idx2:
                continue
            c1 = code[idx1]
            c2 = code[idx2]

            # Temporarily remove them
            for w in balls[c1]:
                coverage[w] -= 1
            for w in balls[c2]:
                coverage[w] -= 1

            # Check if we can find two new codewords that restore coverage
            # and allow further removals
            # Find words that are now uncovered
            uncovered_now = [w for w in range(total_words) if coverage[w] == 0]

            if not uncovered_now:
                # Great, we can just remove both
                # Actually we need to update code list
                # Remove c1 and c2 from code
                if idx1 < idx2:
                    code.pop(idx2)
                    code.pop(idx1)
                else:
                    code.pop(idx1)
                    code.pop(idx2)
                new_len = len(code)
                if new_len < best_len:
                    best_len = new_len
                    best_code = list(code)
                    best_coverage = coverage.copy()
                T *= cooling_rate
                continue
            else:
                # Need to add codewords to cover uncovered words
                # Pick random uncovered words and add codewords
                added = []
                for uw in uncovered_now[:2]:  # limit to 2
                    # Add a random neighbor of uw
                    candidates = balls[uw]
                    new_c = random.choice(candidates)
                    if new_c not in code and new_c not in added:
                        added.append(new_c)
                        for w in balls[new_c]:
                            coverage[w] += 1
                        code.append(new_c)

                # After adding, try to remove more
                local_improved = True
                while local_improved and time.time() < deadline:
                    local_improved = False
                    random.shuffle(code)
                    for i in range(len(code) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        c = code[i]
                        can_remove = True
                        for w in balls[c]:
                            if coverage[w] <= 1:
                                can_remove = False
                                break
                        if can_remove:
                            for w in balls[c]:
                                coverage[w] -= 1
                            code.pop(i)
                            local_improved = True

                new_len = len(code)
                delta = new_len - current_len

                # Accept with probability exp(-delta/T)
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    if new_len < best_len:
                        best_len = new_len
                        best_code = list(code)
                        best_coverage = coverage.copy()
                else:
                    # Reject: restore state
                    # This is complex; let's just rebuild from best if we're in trouble
                    # For simplicity, if we rejected, we need to undo the changes
                    # This is hard to track. Let's use a different approach:
                    # Always keep a backup and restore on reject.

                    # Actually, let's simplify: only accept if it improves or is equal.
                    # For now, just revert by reloading best.
                    code = list(best_code)
                    coverage = best_coverage.copy()
                    T *= cooling_rate
                    continue

        elif move_type < 0.7 and len(code) > 0:
            # Single swap: remove one, add one
            idx1 = random.randint(0, len(code) - 1)
            c1 = code[idx1]

            # Temporarily remove
            for w in balls[c1]:
                coverage[w] -= 1

            # Find uncovered words
            uncovered_now = [w for w in range(total_words) if coverage[w] == 0]

            if not uncovered_now:
                # Can just remove
                code.pop(idx1)
                new_len = len(code)
                if new_len < best_len:
                    best_len = new_len
                    best_code = list(code)
                    best_coverage = coverage.copy()
                T *= cooling_rate
                continue
            else:
                # Add a codeword to cover uncovered
                uw = random.choice(uncovered_now)
                new_c = random.choice(balls[uw])
                if new_c not in code:
                    for w in balls[new_c]:
                        coverage[w] += 1
                    code.append(new_c)

                # Try to remove more
                local_improved = True
                while local_improved and time.time() < deadline:
                    local_improved = False
                    random.shuffle(code)
                    for i in range(len(code) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        c = code[i]
                        can_remove = True
                        for w in balls[c]:
                            if coverage[w] <= 1:
                                can_remove = False
                                break
                        if can_remove:
                            for w in balls[c]:
                                coverage[w] -= 1
                            code.pop(i)
                            local_improved = True

                new_len = len(code)
                delta = new_len - current_len

                if delta <= 0 or random.random() < np.exp(-delta / T):
                    if new_len < best_len:
                        best_len = new_len
                        best_code = list(code)
                        best_coverage = coverage.copy()
                else:
                    code = list(best_code)
                    coverage = best_coverage.copy()
                T *= cooling_rate

        else:
            # Try to just remove a random codeword
            if len(code) > 0:
                idx1 = random.randint(0, len(code) - 1)
                c1 = code[idx1]
                can_remove = True
                for w in balls[c1]:
                    if coverage[w] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for w in balls[c1]:
                        coverage[w] -= 1
                    code.pop(idx1)
                    new_len = len(code)
                    if new_len < best_len:
                        best_len = new_len
                        best_code = list(code)
                        best_coverage = coverage.copy()
                T *= cooling_rate

    # Return the best solution found
    result = [list(idx_to_tuple(c)) for c in best_code]
    return result

# EVOLVE-BLOCK-END
