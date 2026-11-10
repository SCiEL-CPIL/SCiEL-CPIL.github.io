import time
import random

def construct():
    start_time = time.time()
    time_limit = 58.0  # Leave margin before the 60s hard limit (evaluator kills at 90s, but task says 60s finish)

    def check_conflict(seq, x, current_sums):
        """Check if appending x to seq creates duplicate consecutive-block sums."""
        total = x
        if total in current_sums:
            return True
        for a in reversed(seq):
            total += a
            if total in current_sums:
                return True
        return False

    def get_new_sums(seq, x, current_sums):
        """Return the new sums added when appending x to seq."""
        new_sums = set()
        total = x
        new_sums.add(total)
        for a in reversed(seq):
            total += a
            new_sums.add(total)
        return new_sums

    # 1. Greedy baseline: build a valid sequence greedily
    values = []
    sums_set = set()
    for x in range(1, 257):
        if not check_conflict(values, x, sums_set):
            values.append(x)
            new_s = set()
            total = x
            new_s.add(total)
            for a in reversed(values[:-1]):
                total += a
                new_s.add(total)
            sums_set.update(new_s)

    best_seq = values[:]
    best_len = len(best_seq)

    # 2. Local search: try to improve by swapping/inserting
    # Strategy: iteratively try to replace an element with a larger one,
    # or insert new elements, to extend the sequence.

    def try_extend(seq, sums_set, max_time):
        """Try to extend seq by finding elements that can be appended."""
        current_seq = seq[:]
        current_sums = sums_set.copy()
        last_val = current_seq[-1] if current_seq else 0

        # Try appending from last_val+1 to 256
        for x in range(last_val + 1, 257):
            if time.time() > max_time:
                break
            if not check_conflict(current_seq, x, current_sums):
                current_seq.append(x)
                new_s = get_new_sums(current_seq[:-1], x, current_sums)
                current_sums.update(new_s)

        return current_seq, current_sums

    # 3. Backtracking with randomization to find longer sequences
    def dfs_search(current_seq, current_sums, depth, target_len, max_time):
        """Search for a valid sequence of length target_len."""
        if time.time() > max_time:
            return None

        if depth == target_len:
            return current_seq[:]

        last_val = current_seq[-1] if current_seq else 0

        # Pruning: not enough numbers left
        if (target_len - depth) > (256 - last_val):
            return None

        candidates = list(range(last_val + 1, 257))
        random.shuffle(candidates)

        for x in candidates:
            if time.time() > max_time:
                return None

            if check_conflict(current_seq, x, current_sums):
                continue

            new_sums = current_sums.union(get_new_sums(current_seq, x, current_sums))
            result = dfs_search(current_seq + [x], new_sums, depth + 1, target_len, max_time)
            if result is not None:
                return result

        return None

    # 4. Iteratively try to find longer sequences
    target_len = best_len + 1
    max_search_time = time_limit - 5  # Keep 5s margin

    while time.time() < max_search_time - 1:
        # Try DFS to find sequence of target_len
        result = dfs_search([], set(), 0, target_len, max_search_time)
        if result is not None:
            best_seq = result
            best_len = target_len
            target_len += 1
        else:
            # If DFS fails quickly, try a few more times with different random seeds
            # But don't waste too much time
            if time.time() > max_search_time - 2:
                break
            # Try a few more random restarts
            found = False
            for _ in range(3):
                if time.time() > max_search_time - 2:
                    break
                result = dfs_search([], set(), 0, target_len, max_search_time)
                if result is not None:
                    best_seq = result
                    best_len = target_len
                    target_len += 1
                    found = True
                    break
            if not found:
                # Give up on this target length, try to see if we can do better with local search
                break

    # 5. Final local search: try to improve best_seq by small modifications
    # Try inserting elements between existing ones
    if time.time() < max_search_time - 1:
        current_seq = best_seq[:]
        current_sums = set()
        for i in range(len(current_seq)):
            s = 0
            for j in range(i, len(current_seq)):
                s += current_seq[j]
                current_sums.add(s)

        # Try to insert new elements
        improved = True
        while improved and time.time() < max_search_time - 1:
            improved = False
            for i in range(len(current_seq) + 1):
                if time.time() > max_search_time - 1:
                    break
                # Try inserting a value at position i
                prev_val = current_seq[i-1] if i > 0 else 0
                next_val = current_seq[i] if i < len(current_seq) else 257
                for x in range(prev_val + 1, min(next_val, 257)):
                    if x in current_seq:
                        continue
                    # Check if inserting x at position i is valid
                    # Need to check all consecutive block sums
                    test_seq = current_seq[:i] + [x] + current_seq[i:]
                    # Verify strictly increasing
                    for k in range(len(test_seq) - 1):
                        if test_seq[k] >= test_seq[k+1]:
                            break
                    else:
                        # Check all consecutive block sums are distinct
                        test_sums = set()
                        valid = True
                        for a in range(len(test_seq)):
                            s = 0
                            for b in range(a, len(test_seq)):
                                s += test_seq[b]
                                if s in test_sums:
                                    valid = False
                                    break
                                test_sums.add(s)
                            if not valid:
                                break
                        if valid:
                            current_seq = test_seq
                            current_sums = test_sums
                            best_seq = current_seq[:]
                            best_len = len(best_seq)
                            improved = True
                            break
                if improved:
                    break

    return best_seq
