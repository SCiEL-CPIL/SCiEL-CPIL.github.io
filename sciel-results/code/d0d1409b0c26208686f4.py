"""Structural change: difference-vector simulated annealing for increasing sequences
with distinct consecutive-block sums.

Representation: a sequence a_0 < a_1 < ... < a_{n-1} in [1, 256] is encoded by
differences d_i = a_{i+1} - a_i >= 1 and a starting value a_0 >= 1.
We perform simulated annealing on (a_0, d_0, ..., d_{n-2}) accepting moves based
on the number of duplicate consecutive-block sums, so the search can escape local
optima that trap a strict-validity greedy/DFS.  We also try to change length by
inserting/removing a difference.
"""

import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def block_sums(seq):
        """Return list of all non-empty consecutive block sums."""
        sums = []
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.append(s)
        return sums

    def dup_count(sums_list):
        return len(sums_list) - len(set(sums_list))

    def is_valid(seq):
        if not seq:
            return True
        for i in range(1, len(seq)):
            if seq[i] <= seq[i - 1]:
                return False
        s = block_sums(seq)
        return len(s) == len(set(s))

    def seq_from_diffs(a0, ds):
        """Build sequence from start value and differences."""
        seq = [a0]
        cur = a0
        for d in ds:
            cur += d
            seq.append(cur)
        return seq

    def greedy():
        """Fast greedy construction of a valid increasing sequence."""
        values = []
        sums = set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    # Initial valid sequence from greedy
    best_seq = greedy()
    best_len = len(best_seq)

    # Convert to difference representation
    def to_diffs(seq):
        if len(seq) <= 1:
            return (seq[0] if seq else 1, [])
        a0 = seq[0]
        ds = [seq[i] - seq[i - 1] for i in range(1, len(seq))]
        return (a0, ds)

    def diff_to_seq(a0, ds):
        return seq_from_diffs(a0, ds)

    # Simulated annealing on difference vectors
    def anneal(init_seq, max_iters=40000, t0=8.0, t1=0.1):
        nonlocal best_seq, best_len
        if not init_seq:
            return init_seq
        a0, ds = to_diffs(init_seq)
        cur_seq = init_seq[:]
        cur_sums = set(block_sums(cur_seq))
        cur_dup = len(block_sums(cur_seq)) - len(cur_sums)
        cur_len = len(cur_seq)

        best_local = cur_seq[:]
        best_local_len = cur_len
        best_local_dup = cur_dup

        # Temperature schedule
        t = t0
        cooling = (t1 / t0) ** (1.0 / max(1, max_iters))

        used = set(cur_seq)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            r = random.random()

            if r < 0.25 and len(ds) < 255:
                # Insert a new difference at a random position
                pos = random.randint(0, len(ds))
                # Choose a difference value: between 1 and 20
                d_new = random.randint(1, 20)
                new_ds = ds[:pos] + [d_new] + ds[pos:]
                # Keep start the same
                new_seq = diff_to_seq(a0, new_ds)
                # Check strictly increasing and within [1, 256]
                if new_seq and 1 <= new_seq[0] and all(1 <= x <= 256 for x in new_seq):
                    if all(new_seq[i] < new_seq[i + 1] for i in range(len(new_seq) - 1)):
                        new_sums_list = block_sums(new_seq)
                        new_sums = set(new_sums_list)
                        new_dup = len(new_sums_list) - len(new_sums)
                        delta = new_dup - cur_dup
                        if delta <= 0 or random.random() < (t / (t + 1)) ** (delta if delta > 0 else 0):
                            # Update used set
                            # Remove old values not in new_seq
                            old_used = used
                            new_used = set(new_seq)
                            used = new_used
                            a0, ds = a0, new_ds
                            cur_seq = new_seq
                            cur_sums = new_sums
                            cur_dup = new_dup
                            cur_len = len(cur_seq)
                            if (cur_dup < best_local_dup or
                                (cur_dup == best_local_dup and cur_len > best_local_len)):
                                best_local = cur_seq[:]
                                best_local_len = cur_len
                                best_local_dup = cur_dup
            elif r < 0.45 and len(ds) > 0:
                # Remove a difference at a random position
                pos = random.randint(0, len(ds) - 1)
                new_ds = ds[:pos] + ds[pos + 1:]
                new_seq = diff_to_seq(a0, new_ds)
                if new_seq and 1 <= new_seq[0] and all(1 <= x <= 256 for x in new_seq):
                    if all(new_seq[i] < new_seq[i + 1] for i in range(len(new_seq) - 1)):
                        new_sums_list = block_sums(new_seq)
                        new_sums = set(new_sums_list)
                        new_dup = len(new_sums_list) - len(new_sums)
                        delta = new_dup - cur_dup
                        if delta <= 0 or random.random() < (t / (t + 1)) ** (delta if delta > 0 else 0):
                            used = set(new_seq)
                            ds = new_ds
                            cur_seq = new_seq
                            cur_sums = new_sums
                            cur_dup = new_dup
                            cur_len = len(cur_seq)
                            if (cur_dup < best_local_dup or
                                (cur_dup == best_local_dup and cur_len > best_local_len)):
                                best_local = cur_seq[:]
                                best_local_len = cur_len
                                best_local_dup = cur_dup
            elif r < 0.85:
                # Modify a random difference
                if not ds:
                    t *= cooling
                    continue
                pos = random.randint(0, len(ds) - 1)
                old_d = ds[pos]
                # Try to change by a random amount
                d_new = max(1, old_d + random.randint(-10, 10))
                new_ds = ds[:]
                new_ds[pos] = d_new
                new_seq = diff_to_seq(a0, new_ds)
                if new_seq and 1 <= new_seq[0] and all(1 <= x <= 256 for x in new_seq):
                    if all(new_seq[i] < new_seq[i + 1] for i in range(len(new_seq) - 1)):
                        new_sums_list = block_sums(new_seq)
                        new_sums = set(new_sums_list)
                        new_dup = len(new_sums_list) - len(new_sums)
                        delta = new_dup - cur_dup
                        if delta <= 0 or random.random() < (t / (t + 1)) ** (delta if delta > 0 else 0):
                            used = set(new_seq)
                            ds = new_ds
                            cur_seq = new_seq
                            cur_sums = new_sums
                            cur_dup = new_dup
                            cur_len = len(cur_seq)
                            if (cur_dup < best_local_dup or
                                (cur_dup == best_local_dup and cur_len > best_local_len)):
                                best_local = cur_seq[:]
                                best_local_len = cur_len
                                best_local_dup = cur_dup
            else:
                # Perturb start value
                a0_new = max(1, min(256, a0 + random.randint(-15, 15)))
                new_seq = diff_to_seq(a0_new, ds)
                if new_seq and 1 <= new_seq[0] and all(1 <= x <= 256 for x in new_seq):
                    if all(new_seq[i] < new_seq[i + 1] for i in range(len(new_seq) - 1)):
                        new_sums_list = block_sums(new_seq)
                        new_sums = set(new_sums_list)
                        new_dup = len(new_sums_list) - len(new_sums)
                        delta = new_dup - cur_dup
                        if delta <= 0 or random.random() < (t / (t + 1)) ** (delta if delta > 0 else 0):
                            a0 = a0_new
                            used = set(new_seq)
                            cur_seq = new_seq
                            cur_sums = new_sums
                            cur_dup = new_dup
                            cur_len = len(cur_seq)
                            if (cur_dup < best_local_dup or
                                (cur_dup == best_local_dup and cur_len > best_local_len)):
                                best_local = cur_seq[:]
                                best_local_len = cur_len
                                best_local_dup = cur_dup

            t *= cooling

        # Return the best valid sequence found (or best with minimal dups if none valid)
        if best_local_dup == 0 and len(best_local) > len(best_seq):
            best_seq = best_local
            best_len = len(best_local)
        elif is_valid(best_local) and len(best_local) > len(best_seq):
            best_seq = best_local
            best_len = len(best_local)
        return best_local

    # Run several annealing restarts
    random.seed(123)
    for restart in range(30):
        if time.time() > deadline:
            break

        # Create a perturbed starting sequence
        if random.random() < 0.3 and best_len > 2:
            # Randomly drop a few elements
            start_seq = list(best_seq)
            for _ in range(random.randint(1, min(3, len(start_seq) - 1))):
                if len(start_seq) > 1:
                    idx = random.randint(0, len(start_seq) - 1)
                    start_seq.pop(idx)
        else:
            start_seq = list(best_seq)

        # Also try a fresh greedy with random start
        if random.random() < 0.2:
            start_val = random.randint(1, 80)
            fresh = []
            sums_fresh = set()
            for x in range(start_val, 257):
                tails = [x]
                total = x
                for a in reversed(fresh):
                    total += a
                    tails.append(total)
                if not sums_fresh.intersection(tails):
                    fresh.append(x)
                    sums_fresh.update(tails)
            if len(fresh) > len(start_seq):
                start_seq = fresh

        # Ensure strictly increasing
        start_seq = sorted(set(start_seq))
        start_seq = [x for x in start_seq if 1 <= x <= 256]
        # Enforce strictly increasing (remove non-increasing)
        filtered = []
        last = 0
        for x in start_seq:
            if x > last:
                filtered.append(x)
                last = x
        start_seq = filtered

        if not start_seq:
            start_seq = [1]

        anneal(start_seq, max_iters=8000)

        # Try to extend best_seq by appending
        if len(best_seq) < 256:
            for x in range(best_seq[-1] + 1, 257):
                new_seq = best_seq + [x]
                s = block_sums(new_seq)
                if len(s) == len(set(s)):
                    best_seq = new_seq
                    best_len = len(best_seq)
                    break

    # Final validation and fallback
    if not is_valid(best_seq):
        # Fallback to greedy
        best_seq = greedy()

    return best_seq
