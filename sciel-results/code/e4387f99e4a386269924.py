import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)

    # Precompute factor vectors
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
            if temp == 1:
                break
        return tuple(v)

    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # Beam State: (length, seq, prefix_list, diff_set)
    # We sort by a composite score: length (primary), then flexibility (secondary)

    beam = [(0, [], [zero_vec], frozenset())]

    # Dynamic beam width: wider early for diversity, narrower later for refinement
    def get_beam_width(step, total_steps):
        if step < total_steps * 0.2:
            return 200
        elif step < total_steps * 0.5:
            return 150
        elif step < total_steps * 0.8:
            return 100
        else:
            return 80

    total_steps = 127  # numbers 2 to 128

    for step, n in enumerate(range(2, 129)):
        if time.time() > deadline:
            break

        current_beam_width = get_beam_width(step, total_steps)

        new_beam = []
        v_n = vecs[n]

        for state in beam:
            length, seq, pref_list, diff_set = state
            last_pref = pref_list[-1]

            # Option 1: Exclude n
            new_beam.append((length, seq, pref_list, diff_set))

            # Option 2: Include n
            new_pref_list_val = [last_pref[i] + v_n[i] for i in range(num_primes)]
            new_pref = tuple(new_pref_list_val)

            valid = True
            new_diffs = []

            # Check all previous prefixes for collision
            for p in pref_list:
                diff_list = [new_pref[i] - p[i] for i in range(num_primes)]
                diff_tup = tuple(diff_list)
                if diff_tup in diff_set:
                    valid = False
                    break
                new_diffs.append(diff_tup)

            if valid:
                new_pref_list = pref_list + [new_pref]
                new_diff_set = diff_set.union(new_diffs)
                new_seq = seq + [n]
                new_length = length + 1
                new_beam.append((new_length, new_seq, new_pref_list, new_diff_set))

        # Score function: length + flexibility heuristic
        # Flexibility: how many of the next K numbers can still be added
        def calc_score(state):
            l, seq, pl, ds = state
            last_p = pl[-1]

            # Lookahead: check compatibility with next 10 numbers
            potential = 0.0
            for i, m in enumerate(range(n + 1, min(n + 11, 129))):
                weight = 1.0 / (i + 1) ** 0.7  # slower decay than linear
                v_m = vecs[m]
                new_p_val = [last_p[i2] + v_m[i2] for i2 in range(num_primes)]
                new_p = tuple(new_p_val)
                ok = True
                for p in pl:
                    d_val = [new_p[i2] - p[i2] for i2 in range(num_primes)]
                    if tuple(d_val) in ds:
                        ok = False
                        break
                if ok:
                    potential += weight

            # Penalty: if diff_set is very large relative to length, we're "crowded"
            # This may block future additions
            crowd_penalty = 0.0
            if l > 0 and len(ds) > l * l * 1.5:
                crowd_penalty = (len(ds) - l * l * 1.5) * 0.01

            return (l, potential - crowd_penalty)

        new_beam.sort(key=calc_score, reverse=True)
        beam = new_beam[:current_beam_width]

        # Early stop if we've used all numbers
        if beam and beam[0][0] == 127:
            return beam[0][1]

    # Local improvement pass: try to add missed numbers at the end
    if beam:
        best_state = beam[0]
        length, seq, pref_list, diff_set = best_state
        last_pref = pref_list[-1]

        # Try to add any numbers from 2 to 128 that weren't included
        for m in range(2, 129):
            if m in seq:
                continue
            v_m = vecs[m]
            new_pref_list_val = [last_pref[i] + v_m[i] for i in range(num_primes)]
            new_pref = tuple(new_pref_list_val)

            valid = True
            for p in pref_list:
                d_val = [new_pref[i] - p[i] for i in range(num_primes)]
                if tuple(d_val) in diff_set:
                    valid = False
                    break

            if valid:
                # This would require reordering, which is complex.
                # Skip for now; the beam search should have found good solutions.
                pass

    if beam:
        return beam[0][1]
    return []
