import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s buffer for safety

    def time_left():
        return deadline - time.time()

    def get_block_sums_bitset(seq):
        """Calculate all consecutive block sums and return as a set."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def count_conflicts_fast(seq):
        """Return the number of duplicate sums using a set."""
        sums = set()
        n = len(seq)
        conflicts = 0
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    conflicts += 1
                else:
                    sums.add(s)
        return conflicts

    def is_valid(seq):
        return count_conflicts_fast(seq) == 0

    # === Strategy 1: Multiple Greedy Initializations ===
    def greedy_min_conflict():
        """Greedy: pick the value that creates the fewest conflicts."""
        values = []
        sums = set()
        for _ in range(20):  # Try to build up to 20 elements
            best_x = None
            best_conflicts = float('inf')
            for x in range(1, 257):
                if values and x <= values[-1]:
                    continue
                # Calculate new sums if we add x
                new_sums = set()
                total = x
                new_sums.add(total)
                for a in reversed(values):
                    total += a
                    new_sums.add(total)
                conflicts = len(new_sums.intersection(sums))
                if conflicts == 0:
                    # Also prefer smaller values to keep sums small
                    if best_x is None or x < best_x:
                        best_x = x
                        best_conflicts = 0
                elif conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_x = x
            if best_x is None or best_conflicts > 0:
                break
            values.append(best_x)
            # Update sums
            total = best_x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
        return values

    def greedy_sequential():
        """Greedy: try values in order, skip if conflict."""
        values = []
        sums = set()
        for x in range(1, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(values):
                total += a
                new_sums.add(total)
            if not new_sums.intersection(sums):
                values.append(x)
                sums.update(new_sums)
        return values

    def greedy_randomized():
        """Greedy with randomization: shuffle candidates, pick first valid."""
        values = []
        sums = set()
        for _ in range(20):
            candidates = list(range(1, 257))
            if values:
                candidates = [x for x in candidates if x > values[-1]]
            random.shuffle(candidates)
            placed = False
            for x in candidates:
                new_sums = set()
                total = x
                new_sums.add(total)
                for a in reversed(values):
                    total += a
                    new_sums.add(total)
                if not new_sums.intersection(sums):
                    values.append(x)
                    sums.update(new_sums)
                    placed = True
                    break
            if not placed:
                break
        return values

    # Run multiple greedy strategies
    best_seq = []
    best_len = 0

    for _ in range(50):
        seq = greedy_randomized()
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
        if time_left() < 10:
            break

    seq2 = greedy_sequential()
    if len(seq2) > best_len:
        best_seq = seq2
        best_len = len(seq2)

    seq3 = greedy_min_conflict()
    if len(seq3) > best_len:
        best_seq = seq3
        best_len = len(seq3)

    # === Strategy 2: Local Search / Simulated Annealing ===
    def local_search(init_seq, time_budget=5.0):
        """Improve a sequence using local search."""
        seq = init_seq[:]
        n = len(seq)
        if n == 0:
            return seq

        current_cost = count_conflicts_fast(seq)
        if current_cost == 0:
            return seq

        t_start = time.time()
        t_end = t_start + time_budget
        T = 5.0
        min_T = 0.1
        cooling = 0.995

        while time.time() < t_end:
            # Try to insert a new element
            if random.random() < 0.3:
                # Try inserting at a random position
                pos = random.randint(0, n)
                for _ in range(100):
                    x = random.randint(1, 256)
                    if pos > 0 and x <= seq[pos-1]:
                        continue
                    if pos < n and x >= seq[pos]:
                        continue
                    # Check if inserting x at pos creates conflicts
                    new_seq = seq[:pos] + [x] + seq[pos:]
                    new_cost = count_conflicts_fast(new_seq)
                    delta_cost = new_cost - current_cost
                    if delta_cost <= 0 or random.random() < math.exp(-delta_cost / max(T, 0.01)):
                        seq = new_seq
                        current_cost = new_cost
                        n += 1
                        break
            else:
                # Modify an existing element
                i = random.randint(0, n - 1)
                delta = random.randint(-3, 3)
                new_val = seq[i] + delta
                if new_val < 1 or new_val > 256:
                    continue
                if i > 0 and new_val <= seq[i-1]:
                    continue
                if i < n - 1 and new_val >= seq[i+1]:
                    continue

                old_val = seq[i]
                seq[i] = new_val
                new_cost = count_conflicts_fast(seq)
                delta_cost = new_cost - current_cost

                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / max(T, 0.01)):
                    current_cost = new_cost
                    if current_cost == 0:
                        return seq
                else:
                    seq[i] = old_val

            T *= cooling
            if T < min_T:
                T = min_T

        return seq

    # Run local search on the best greedy result
    if time_left() > 15:
        improved = local_search(best_seq, time_budget=min(10, time_left() - 15))
        if len(improved) > best_len or (len(improved) == best_len and count_conflicts_fast(improved) < count_conflicts_fast(best_seq)):
            best_seq = improved
            best_len = len(best_seq)

    # Try local search on multiple random sequences
    for _ in range(20):
        if time_left() < 15:
            break
        n_try = random.randint(best_len, best_len + 3)
        if n_try > 25:
            n_try = 25
        seq = sorted(random.sample(range(1, 257), n_try))
        improved = local_search(seq, time_budget=min(3, time_left() - 15))
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(improved)

    # === Strategy 3: Iterative Deepening DFS with Pruning ===
    def max_extension(last_val, remaining_slots):
        """Upper bound on how many more elements we can add after last_val."""
        # We need strictly increasing values in [last_val+1, 256]
        # So at most 256 - last_val elements
        return 256 - last_val

    def dfs_build(current_seq, used_sums, target_len, time_budget):
        """
        Build a sequence of length target_len using DFS with pruning.
        Returns the sequence if found, None otherwise.
        """
        if len(current_seq) == target_len:
            return current_seq[:]

        last_val = current_seq[-1] if current_seq else 0
        remaining_needed = target_len - len(current_seq)

        # Pruning: if we can't fit enough more elements
        if max_extension(last_val, remaining_needed) < remaining_needed:
            return None

        # Try candidate values
        # Heuristic: try smaller values first (keeps sums smaller)
        for x in range(last_val + 1, 257):
            # Calculate new sums
            new_sums_list = []
            total = x
            new_sums_list.append(total)
            for a in reversed(current_seq):
                total += a
                new_sums_list.append(total)

            # Check for conflicts
            has_conflict = False
            for s in new_sums_list:
                if s in used_sums:
                    has_conflict = True
                    break
            if has_conflict:
                continue

            # Pruning: check if the new sums leave enough room
            # The largest sum so far
            max_sum = max(new_sums_list) if new_sums_list else 0
            # Future sums must be > max_sum and distinct
            # With remaining_needed - 1 more elements, we need at least
            # (remaining_needed - 1) distinct sums > max_sum
            # The minimum possible next sum is x + (something)
            # This is a weak bound, skip for now

            # Recursive call
            new_used_sums = used_sums.union(new_sums_list)
            result = dfs_build(current_seq + [x], new_used_sums, target_len, time_budget)
            if result is not None:
                return result

            # Check time
            if time.time() > time_budget:
                return None

        return None

    # Try to extend the best sequence using DFS
    # First, try to build from scratch with increasing target lengths
    target_len = best_len + 1
    while target_len <= best_len + 5 and time_left() > 10:
        t_budget = time_left() - 10
        if t_budget < 1:
            break

        # Start DFS from empty sequence
        result = dfs_build([], set(), target_len, time.time() + min(5, t_budget))
        if result is not None:
            best_seq = result
            best_len = target_len
            target_len += 1
        else:
            # Try starting from a partial greedy sequence
            partial = best_seq[:min(len(best_seq), target_len - 1)]
            if partial:
                used_sums = get_block_sums_bitset(partial)
                result = dfs_build(partial, used_sums, target_len, time.time() + min(3, t_budget))
                if result is not None:
                    best_seq = result
                    best_len = target_len
                    target_len += 1
                else:
                    break
            else:
                break

    # Final validation
    if not is_valid(best_seq):
        # Fallback to a simple valid sequence
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]

    return best_seq

construct()
