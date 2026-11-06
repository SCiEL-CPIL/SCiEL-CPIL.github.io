import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    max_val = 4096

    # Helper to check if adding x is valid
    def can_add(S, used_sums, pair_sums, x):
        # 1. x + x + x
        s1 = 3 * x
        if s1 in used_sums:
            return False

        # 2. x + x + s
        for s in S:
            s2 = 2 * x + s
            if s2 in used_sums:
                return False
            # Pre-check: x + s must not be a pair sum in S
            if x + s in pair_sums:
                return False

        # 3. x + s_i + s_j
        for p in pair_sums:
            s3 = x + p
            if s3 in used_sums:
                return False

        # 4. 2x must not be a pair sum in S
        if 2 * x in pair_sums:
            return False

        return True

    def get_sums_for(x, S_old):
        """Returns list of sums contributed by x given S_old (which does not contain x)."""
        sums = [3 * x]
        for s in S_old:
            sums.append(2 * x + s)
        for i in range(len(S_old)):
            for j in range(i, len(S_old)):
                sums.append(x + S_old[i] + S_old[j])
        return sums

    def get_pair_sums_for(x, S_old):
        """Returns set of pair sums involving x, given S_old."""
        ps = {2 * x}
        for s in S_old:
            ps.add(x + s)
        return ps

    # Initial Greedy to get a baseline
    S = []
    S_set = set()
    used_sums = set()
    pair_sums = set()

    for x in range(1, max_val + 1):
        if x not in S_set:
            if can_add(S, used_sums, pair_sums, x):
                S_old = S
                S.append(x)
                S_set.add(x)

                for s_val in get_sums_for(x, S_old):
                    used_sums.add(s_val)

                for p_val in get_pair_sums_for(x, S_old):
                    pair_sums.add(p_val)

    best_set = list(S)
    best_len = len(S)

    def add_element(S, S_set, used_sums, pair_sums, x):
        S.append(x)
        S_set.add(x)
        S_old = S[:-1]
        for s_val in get_sums_for(x, S_old):
            used_sums.add(s_val)
        for p_val in get_pair_sums_for(x, S_old):
            pair_sums.add(p_val)

    def remove_element(S, S_set, used_sums, pair_sums, idx):
        y = S[idx]
        S_rest = S[:idx] + S[idx+1:]

        S.pop(idx)
        S_set.remove(y)

        for s_val in get_sums_for(y, S_rest):
            used_sums.remove(s_val)

        for p_val in get_pair_sums_for(y, S_rest):
            pair_sums.remove(p_val)

        return y

    def greedy_add_pass(S, S_set, used_sums, pair_sums):
        added = []
        for x in range(1, max_val + 1):
            if x not in S_set:
                if can_add(S, used_sums, pair_sums, x):
                    add_element(S, S_set, used_sums, pair_sums, x)
                    added.append(x)
        return added

    def try_swap(S, S_set, used_sums, pair_sums, y, candidates=None):
        """Try to swap y with some x. Returns True if successful."""
        if candidates is None:
            candidates = range(1, max_val + 1)
        # Remove y
        idx = S.index(y)
        remove_element(S, S_set, used_sums, pair_sums, idx)
        swapped = False
        for x in candidates:
            if x in S_set:
                continue
            if can_add(S, used_sums, pair_sums, x):
                add_element(S, S_set, used_sums, pair_sums, x)
                swapped = True
                break
        if not swapped:
            add_element(S, S_set, used_sums, pair_sums, y)
        return swapped

    # Local Search Loop
    while time.time() - start_time < DEADLINE:
        improved = False

        # Phase 1: Greedy Add
        added = greedy_add_pass(S, S_set, used_sums, pair_sums)
        if added:
            improved = True
            if len(S) > best_len:
                best_set = list(S)
                best_len = len(S)

        if not improved and len(S) > 0:
            # Phase 2: Swap (1-for-1) - try to escape local optima
            # Try multiple random swaps per iteration
            for _ in range(20):
                if time.time() - start_time > DEADLINE:
                    break
                idx = random.randint(0, len(S) - 1)
                y = S[idx]
                if try_swap(S, S_set, used_sums, pair_sums, y):
                    # After swap, try greedy add
                    added2 = greedy_add_pass(S, S_set, used_sums, pair_sums)
                    if added2:
                        improved = True
                        if len(S) > best_len:
                            best_set = list(S)
                            best_len = len(S)
                        break

        if not improved and len(S) > 1:
            # Phase 3: 2-for-1 removal then add
            # Remove two random elements, then try to add
            for _ in range(10):
                if time.time() - start_time > DEADLINE:
                    break
                idx1 = random.randint(0, len(S) - 1)
                idx2 = random.randint(0, len(S) - 2)
                if idx2 >= idx1:
                    idx2 += 1

                y1 = S[idx1]
                y2 = S[idx2]

                # Remove y1
                remove_element(S, S_set, used_sums, pair_sums, idx1)
                # Adjust index for y2
                if idx2 > idx1:
                    idx2 -= 1
                remove_element(S, S_set, used_sums, pair_sums, idx2)

                added = greedy_add_pass(S, S_set, used_sums, pair_sums)
                if len(added) >= 1:
                    improved = True
                    if len(S) > best_len:
                        best_set = list(S)
                        best_len = len(S)
                    break
                else:
                    # Restore y1, y2
                    add_element(S, S_set, used_sums, pair_sums, y2)
                    add_element(S, S_set, used_sums, pair_sums, y1)

    return best_set

# Quick sanity check / test
if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
    print(res[:10])
