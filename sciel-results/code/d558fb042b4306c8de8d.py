import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin before 90s kill

    # Precompute all APs (length >= 2) and for each position, which APs contain it
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)

    num_aps = len(ap_terms)

    # For each position, list of AP indices containing it
    pos_to_aps = [[] for _ in range(N)]
    for j, terms in enumerate(ap_terms):
        for i in terms:
            pos_to_aps[i].append(j)
    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]

    # Precompute AP lengths (for histogram bounds)
    ap_lengths = np.array([len(t) for t in ap_terms], dtype=np.int32)
    max_len = int(ap_lengths.max())

    # Build incidence as sparse-like: for each AP, list of positions
    # We'll use a dense float32 matrix for initial sums computation
    M = np.zeros((num_aps, N), dtype=np.float32)
    for j, terms in enumerate(ap_terms):
        M[j, terms] = 1.0

    def compute_sums(signs):
        return M @ signs

    def make_histogram(sums):
        # sums: int32 array of AP sums (can be negative)
        # histogram[|s|] = count of APs with absolute sum |s|
        abs_sums = np.abs(sums).astype(np.int32)
        hist = np.zeros(max_len + 1, dtype=np.int64)
        np.add.at(hist, abs_sums, 1)
        return hist, abs_sums

    def get_max_from_hist(hist):
        # Find highest index with nonzero count
        # hist is small (max_len+1 <= 256), so just scan
        for i in range(len(hist) - 1, -1, -1):
            if hist[i] > 0:
                return i
        return 0

    def update_hist_for_flip(hist, abs_sums, ap_indices, delta_abs_old, delta_abs_new):
        """
        When we flip a position, the sums of APs in ap_indices change.
        For each AP, old_abs = abs_sums[ap], new_abs = abs(sums[ap] + delta)
        We update hist[old_abs] -= 1, hist[new_abs] += 1
        """
        # This is called with the actual new sums already computed for those APs
        pass

    # Better: maintain sums array and hist. On flip, update sums for affected APs,
    # then update hist for each affected AP.

    class HistTracker:
        def __init__(self, sums):
            self.sums = sums.copy()  # int32 or float32, we'll use int32 for exact
            self.abs_sums = np.abs(self.sums).astype(np.int32)
            self.hist = np.zeros(max_len + 1, dtype=np.int64)
            np.add.at(self.hist, self.abs_sums, 1)
            self.current_max = get_max_from_hist(self.hist)

        def flip(self, pos, sign):
            """
            Flip position pos. sign is the current sign (+1 or -1).
            delta = -2 * sign
            Update sums for all APs containing pos.
            """
            delta = -2.0 * sign
            ap_idxs = pos_to_aps_arr[pos]
            for ap in ap_idxs:
                old_sum = self.sums[ap]
                old_abs = self.abs_sums[ap]
                new_sum = old_sum + delta
                new_abs = int(abs(new_sum))
                # Update hist
                self.hist[old_abs] -= 1
                self.hist[new_abs] += 1
                # Update arrays
                self.sums[ap] = new_sum
                self.abs_sums[ap] = new_abs
            # Update current_max
            if self.current_max > 0 and self.hist[self.current_max] == 0:
                self.current_max = get_max_from_hist(self.hist)
            elif new_abs > self.current_max:
                self.current_max = new_abs

        def get_max(self):
            return self.current_max

    def local_search(signs, max_time):
        """
        Greedy local search: repeatedly sweep all positions, flip if it reduces max.
        Returns best signs found.
        """
        signs = signs.copy()
        sums = compute_sums(signs).astype(np.float32)
        tracker = HistTracker(sums)
        best_max = tracker.get_max()
        best_signs = signs.copy()

        no_improve_rounds = 0
        max_rounds = 50

        for rnd in range(max_rounds):
            if time.time() > max_time:
                break
            improved = False
            for k in range(N):
                if time.time() > max_time:
                    break
                # Try flipping k
                old_max = tracker.get_max()
                tracker.flip(k, signs[k])
                signs[k] = -signs[k]
                new_max = tracker.get_max()
                if new_max < old_max:
                    improved = True
                    if new_max < best_max:
                        best_max = new_max
                        best_signs = signs.copy()
                else:
                    # Revert
                    signs[k] = -signs[k]
                    tracker.flip(k, signs[k])
            if not improved:
                break

        return best_signs, best_max

    def random_init(rng):
        return np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(N)], dtype=np.float32)

    def balanced_init(rng):
        # Exactly 128 +1 and 128 -1
        arr = [1.0] * 128 + [-1.0] * 128
        rng.shuffle(arr)
        return np.array(arr, dtype=np.float32)

    def thue_morse_init():
        # Thue-Morse sequence: t(n) = parity of popcount(n)
        # Map to signs: +1 if even, -1 if odd
        signs = np.ones(N, dtype=np.float32)
        for i in range(N):
            if bin(i).count('1') % 2 == 1:
                signs[i] = -1.0
        return signs

    best_signs = None
    best_global_max = N + 1

    rng = random.Random(42)

    # Generate diverse initializations
    inits = []
    inits.append(thue_morse_init())
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        r = random.Random(seed)
        inits.append(random_init(r))
        inits.append(balanced_init(r))

    # Shuffle inits for diversity
    rng.shuffle(inits)

    for init in inits:
        if time.time() > deadline - 3:
            break

        signs, local_max = local_search(init, deadline - 2)

        if local_max < best_global_max:
            best_global_max = local_max
            best_signs = signs.copy()
            if best_global_max <= 1:
                break

    # If we still have time, do additional random restarts
    if time.time() < deadline - 5 and best_signs is not None:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            r = random.Random(rng.randint(0, 10**9))
            init = random_init(r)
            signs, local_max = local_search(init, deadline - 2)
            if local_max < best_global_max:
                best_global_max = local_max
                best_signs = signs.copy()
                if best_global_max <= 1:
                    break

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]

    return [int(s) for s in best_signs]
