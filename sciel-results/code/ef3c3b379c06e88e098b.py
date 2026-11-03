import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish well before.
    deadline = start_time + 140.0

    # 1. Recursive Construction
    word = [1]

    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        L = len(word)
        for i in range(L - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        # If this is the last step (m == n) and n > 4, try to optimize the order
        # of permutations to maximize overlap.
        if m == n and n > 4:
            # We have a list of perms (tuples of length n-1)
            # We want to order them to maximize sum of overlaps.

            def calc_overlap(u, v):
                max_k = min(len(u), len(v))
                for k in range(max_k, 0, -1):
                    if u[-k:] == v[:k]:
                        return k
                return 0

            base_order = perms
            base_score = 0
            for i in range(len(base_order) - 1):
                base_score += calc_overlap(base_order[i], base_order[i+1])

            # Try to improve with local search
            current_order = list(base_order)
            current_score = base_score
            improved = True
            # Limit iterations to ensure we finish in time
            max_iterations = 500
            iter_count = 0

            while improved and time.time() < deadline - 10.0 and iter_count < max_iterations:
                improved = False
                iter_count += 1

                # Random 2-opt: pick a segment and reverse it
                if len(current_order) < 2:
                    break

                i = random.randint(0, len(current_order) - 2)
                j = random.randint(i + 1, len(current_order) - 1)

                # Calculate score change for reversing current_order[i:j+1]
                # We recalculate the total score for simplicity as N is small (720 for n=7)
                new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]

                score = 0
                for k in range(len(new_order) - 1):
                    score += calc_overlap(new_order[k], new_order[k+1])

                if score > current_score:
                    current_score = score
                    current_order = new_order
                    improved = True

            if current_score > base_score:
                perms = current_order
            else:
                perms = base_order
        else:
            # For smaller m, just use the standard order
            pass

        # Construct new word
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)

            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break

            new_word.extend(block[max_overlap:])

        word = new_word

        if time.time() > deadline:
            break

    # If n is small, just return the constructed word
    if n <= 4:
        return "".join(map(str, word))

    # 2. Polishing Phase: Character Deletion
    # This phase attempts to shorten the word by removing redundant characters.

    # Map permutations to IDs for fast counting
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)

    # Initialize counts
    counts = [0] * num_perms
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_id:
            counts[perm_to_id[p]] += 1

    # Check if valid initially (should be)
    if any(c == 0 for c in counts):
        return "".join(map(str, word))

    # Precompute window IDs for the current word for O(1) lookup
    # window_ids[i] = id of permutation at position i, or -1 if not a valid permutation
    window_ids = []
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p, -1)
        window_ids.append(pid)

    def is_deletion_valid_fast(i, current_word, current_counts, current_window_ids):
        """
        Check if deleting character at index i is valid.
        Uses precomputed window_ids for old windows.
        Returns (valid, changes) where changes is a list of (pid, delta) applied.
        """
        L = len(current_word)
        if L <= n:
            return False, []

        # Old windows to remove: k in [max(0, i-n+1), min(i, L-n)]
        old_start_min = max(0, i - n + 1)
        old_start_max = min(i, L - n)

        # New windows to add: k in [max(0, i-n+1), min(i-1, L-n-1)]
        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, L - n - 1)

        changes = []

        # 1. Decrement old windows
        for k in range(old_start_min, old_start_max + 1):
            pid = current_window_ids[k]
            if pid >= 0:
                current_counts[pid] -= 1
                changes.append((pid, -1))
                if current_counts[pid] < 0:
                    for cp, delta in changes:
                        current_counts[cp] -= delta
                    return False, []

        # 2. Increment new windows
        for k in range(new_start_min, new_start_max + 1):
            # New window at position k in the word after deletion
            # consists of current_word[k:i] + current_word[i+1:k+n+1]
            w1 = current_word[k:i]
            w2 = current_word[i+1:k+n+1]
            p = tuple(w1 + w2)
            pid = perm_to_id.get(p)
            if pid is not None:
                current_counts[pid] += 1
                changes.append((pid, 1))

        # Check if any decremented count became 0
        valid = True
        for pid, delta in changes:
            if delta == -1:
                if current_counts[pid] == 0:
                    valid = False
                    break

        if not valid:
            for pid, delta in changes:
                current_counts[pid] -= delta
            return False, []

        return True, changes

    # Perform multiple passes of deletion to find more reductions
    # Each pass goes through the word and deletes any character that can be removed
    # Multiple passes are needed because deleting one character can enable deleting another

    max_passes = 10
    for pass_num in range(max_passes):
        if time.time() > deadline - 5.0:
            break

        i = 0
        L = len(word)
        # Rebuild window_ids at the start of each pass
        window_ids = []
        for pos in range(L - n + 1):
            p = tuple(word[pos:pos+n])
            pid = perm_to_id.get(p, -1)
            window_ids.append(pid)

        deletions_in_pass = 0
        while i < len(word):
            if time.time() > deadline - 5.0:
                break

            valid, changes = is_deletion_valid_fast(i, word, counts, window_ids)
            if valid:
                # Commit deletion
                word.pop(i)
                # Update window_ids: windows before i are unchanged, windows after i shift left
                # But since we're iterating and will rebuild at next pass, we can just continue
                # However, we need to be careful: window_ids is now stale
                # For simplicity, if we delete, we rebuild window_ids and restart the pass
                # This is expensive but correct. Alternatively, we can update window_ids incrementally.
                # Let's update incrementally:
                # Old windows at positions [i-n+1, i] are removed.
                # New windows at positions [i-n+1, i-1] are added.
                # This is complex. Let's just rebuild and restart the pass for correctness.
                # But that's O(L*n) per deletion which is too slow.
                # Instead, let's note that after deletion at i, the window_ids array is invalid.
                # We'll set a flag to rebuild.
                # Actually, for the inner loop, let's just break and restart the pass.
                deletions_in_pass += 1
                # Restart the pass with fresh window_ids
                break
            else:
                i += 1

        if deletions_in_pass == 0:
            # No more deletions possible in this pass
            break

    return "".join(map(str, word))

# Keep the function name and signature as required
# The task says: Interface: construct(n) -> a word ...
# The current program defines construct(n).
