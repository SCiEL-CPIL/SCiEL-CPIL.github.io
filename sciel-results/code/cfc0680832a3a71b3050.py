import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def is_time_up():
        return time.time() > deadline

    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails = []
            total = 0
            for a in reversed(values + [x]):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums

    base_seq, base_sums = greedy()
    best_seq = base_seq
    base_len = len(base_seq)

    # Beam search with better pruning and heuristics
    # State: (last_val, seq_tuple, sums_set, tails_list)
    # tails_list = list of current tail sums (sums ending at last element)
    # This allows incremental update: new_tails = [x] + [x + t for t in reversed(old_tails)]

    beam = [(0, (), set(), [])]
    max_len = 0
    best_state = (0, (), set(), [])

    B = 200  # beam width

    while beam and not is_time_up():
        new_beam = []
        for last, seq, sums, tails in beam:
            # Try possible next values
            # Skip x if x is already in sums (single element block)
            # Also skip if any new tail would be in sums
            for x in range(last + 1, 257):
                if x in sums:
                    continue

                # Compute new tails: [x, x+tails[-1], x+tails[-1]+tails[-2], ...]
                # Actually tails are in order: tails[0] = last element, tails[1] = last two, etc.
                # New tails = [x] + [x + t for t in reversed(tails)]
                # Wait, let me reconsider. If seq = [a1, a2, ..., an], tails = [an, an+an-1, ..., an+...+a1]
                # New seq = [a1, ..., an, x]
                # New tails = [x, x+an, x+an+an-1, ..., x+an+...+a1]
                # So new_tails = [x] + [x + t for t in reversed(tails)]
                # But reversed(tails) = [a1+...+an, ..., an]
                # So x + reversed(tails)[i] = x + (sum of last i+1 elements of seq)
                # That's correct.

                new_tails = [x]
                s = x
                for t in reversed(tails):
                    s += t
                    new_tails.append(s)

                # Check if any new tail is in sums
                valid = True
                for nt in new_tails:
                    if nt in sums:
                        valid = False
                        break

                if valid:
                    new_sums = sums | set(new_tails)
                    new_seq = seq + (x,)
                    new_beam.append((x, new_seq, new_sums, new_tails))

        if not new_beam:
            break

        # Select top B states
        # Heuristic: smaller last value is better (more room for future extensions)
        # Also prefer sequences with smaller total sum (less "crowding")
        # Sort by last value, then by sum of sequence
        new_beam.sort(key=lambda s: (s[0], sum(s[1])))
        beam = new_beam[:B]

        # Update best
        if len(beam[0][1]) > max_len:
            max_len = len(beam[0][1])
            best_state = beam[0]
            best_seq = list(best_state[1])

    if max_len > base_len:
        return best_seq
    else:
        return base_seq
