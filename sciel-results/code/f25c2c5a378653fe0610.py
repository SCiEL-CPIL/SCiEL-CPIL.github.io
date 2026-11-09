import time
import random

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit

    TARGET = set(range(1, 31))
    TARGET_MASK = (1 << 30) - 1

    def get_diffs(S):
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])

    def is_sidon(S):
        n = len(S)
        if n < 2:
            return True
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2

    def covers_all(S):
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)

    def popcount(x):
        return bin(x).count('1')

    def solve_for_D(D, time_limit):
        """
        Find a Sidon set in [0, D] containing 0, covering all diffs 1..30.
        Uses DFS with bitmasks and strong pruning.
        """
        best_result = [None]
        best_size = [100]

        # Precompute for each x in 1..D, the mask of diffs 1..30 it would cover
        # when added to a set containing 0 (i.e., diff is x itself if 1<=x<=30)
        # But this depends on the current set. We'll compute on the fly but efficiently.

        def dfs(elements, used_diffs, covered_mask, last_elem, depth):
            if time.time() > time_limit:
                return False

            if covered_mask == TARGET_MASK:
                if len(elements) < best_size[0]:
                    best_size[0] = len(elements)
                    best_result[0] = elements[:]
                return True

            # Pruning: can we cover remaining diffs with remaining elements?
            # Max elements allowed is 32.
            if len(elements) >= 32:
                return False

            # Lower bound on elements needed:
            # Each new element adds at most len(elements) new diffs.
            # Remaining diffs to cover: popcount(TARGET_MASK ^ covered_mask)
            # But some might be covered by future elements in ways we don't know yet.
            # Simple bound: if remaining_diffs > (32 - len(elements)) * len(elements), prune?
            # Actually, each pair contributes 1 diff. Total pairs in final set of size k is k(k-1)/2.
            # We need at least 30 distinct diffs in 1..30.
            # Let's just rely on the search.

            # Try candidates from last_elem+1 to D
            # To find smallest D, we want to find ANY solution.
            # To find smallest max element, we are iterating D.
            # Within a fixed D, we want to find a solution.

            # Optimization: try smaller x first? Or larger?
            # Smaller x keeps max element small, which is good for the global objective.
            # But we are already fixing D. So any solution in [0,D] is valid.
            # We want to find one quickly.

            for x in range(last_elem + 1, D + 1):
                # Check if adding x is valid
                new_diffs = []
                valid = True
                for y in elements:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if not valid:
                    continue

                # Update covered mask
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))

                # Prune: if this move doesn't increase coverage, it might still be necessary
                # to enable other moves, but if we are deep, it's risky.
                # Let's allow it but maybe deprioritize?
                # Actually, if new_mask == covered_mask, we are adding an element that
                # doesn't help cover 1..30 directly. This is often bad for minimizing size.
                # But it might be needed to avoid conflicts? No, Sidon is about distinctness.
                # If it doesn't cover new target diffs, it's likely useless unless it's
                # a "bridge". But with 0 fixed, adding x creates diffs x-y.
                # If none of these are in 1..30, then x > max(elements)+30.
                # This is unlikely to be useful for covering 1..30.
                if new_mask == covered_mask:
                    continue

                new_used = used_diffs | set(new_diffs)
                new_elements = elements + [x]

                if dfs(new_elements, new_used, new_mask, x, depth + 1):
                    return True

            return False

        # Start with [0]
        success = dfs([0], set(), 0, 0, 0)
        if success and best_result[0] is not None:
            return best_result[0]
        return None

    def local_search(S, time_limit):
        """Reduce max element via swaps and removals."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)
        
        for _ in range(5000):
            if time.time() > time_limit:
                break
            
            improved = False
            
            # Try removing an element if we still cover all diffs
            if len(S) > 2:
                for i in range(len(S)):
                    if S[i] == 0:
                        continue
                    trial = S[:i] + S[i+1:]
                    if covers_all(trial):
                        if max(trial) < best_max:
                            S = trial
                            best = S[:]
                            best_max = max(S)
                            improved = True
                            break
                if improved:
                    continue
            
            # Try swapping an element with a smaller value
            for i in range(1, len(S)):
                # Only try values below the current element to reduce max
                # Randomize the range to avoid getting stuck in local minima
                search_range = range(1, S[i])
                candidates = list(search_range)
                random.shuffle(candidates)
                
                for new_val in candidates:
                    if new_val in S:
                        continue
                    trial = S[:i] + [new_val] + S[i+1:]
                    trial.sort()
                    if is_sidon(trial) and covers_all(trial):
                        if max(trial) < best_max:
                            S = trial
                            best = S[:]
                            best_max = max(S)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return best

    best_set = None
    best_max = float('inf')

    # Phase 1: Find a valid solution by increasing D
    # Known lower bounds suggest D around 70-100 for size ~9-10
    # Let's start from 60 and go up.
    D = 60
    while D < 200:
        if time.time() > deadline - 20:
            break
        
        result = solve_for_D(D, deadline - 10)
        if result is not None:
            best_set = result
            best_max = max(result)
            break
        D += 5

    # Fallback if no solution found yet
    if best_set is None:
        # Greedy construction
        a = [0]
        all_diffs = set()
        for d_target in range(1, 31):
            if d_target in all_diffs:
                continue
            found = False
            for x in range(1, 500):
                if x in a:
                    continue
                new_diffs = [x - y for y in a]
                if any(nd in all_diffs for nd in new_diffs):
                    continue
                trial = sorted(a + [x])
                trial_diffs = set()
                ok = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        dd = trial[j] - trial[i]
                        if dd in trial_diffs:
                            ok = False
                            break
                        trial_diffs.add(dd)
                    if not ok:
                        break
                if ok:
                    a = trial
                    all_diffs = trial_diffs
                    found = True
                    break
            if not found:
                y = 2 * max(a) + d_target + 1
                while True:
                    trial = sorted(a + [y, y + d_target])
                    trial_diffs = set()
                    ok = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            dd = trial[j] - trial[i]
                            if dd in trial_diffs:
                                ok = False
                                break
                            trial_diffs.add(dd)
                        if not ok:
                            break
                    if ok:
                        a = trial
                        all_diffs = trial_diffs
                        break
                    y += 1
        best_set = a
        best_max = max(a)

    # Phase 2: Local search to minimize max element
    if time.time() < deadline - 10:
        improved = local_search(best_set[:], deadline - 5)
        if improved and max(improved) < best_max:
            best_set = improved
            best_max = max(improved)

    return best_set

# Quick self-test to ensure it runs and returns something valid
# if __name__ == "__main__":
#     s = construct()
#     print(s)
#     print(max(s))
