import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

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
    # We sort by length (desc) and then by potential (desc).
    
    beam = [(0, [], [zero_vec], frozenset())]
    
    # Dynamic beam width: Start wide, narrow down
    # Total steps: 127
    # We want to spend more time on early steps to establish good structure
    # and less time on later steps to refine.
    
    # Let's use a simple linear decrease or exponential decay
    # Beam width from 200 down to 20
    
    best_seq = []
    best_length = 0
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        # Calculate dynamic beam width
        # progress = (n - 2) / 126
        # width = int(200 * (1 - progress) + 20)
        # Or simply:
        if n < 20:
            beam_width = 300
        elif n < 50:
            beam_width = 150
        elif n < 100:
            beam_width = 80
        else:
            beam_width = 40
            
        new_beam = []
        v_n = vecs[n]
        
        for state in beam:
            length, seq, pref_list, diff_set = state
            last_pref = pref_list[-1]
            
            # Option 1: Exclude
            new_beam.append((length, seq, pref_list, diff_set))
            
            # Option 2: Include
            new_pref_list_val = [last_pref[i] + v_n[i] for i in range(num_primes)]
            new_pref = tuple(new_pref_list_val)
            
            valid = True
            new_diffs = []
            
            # Check all previous prefixes
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
        
        # Sort new_beam by Length (desc), then by "Flexibility"
        # Flexibility: How many of the next few numbers could be added?
        
        def calc_score(state):
            l, seq, pl, ds = state
            # Potential: count compatible next 5 numbers
            potential = 0
            last_p = pl[-1]
            # Next numbers are n+1, n+2, n+3, n+4, n+5
            # We limit the lookahead to avoid excessive computation
            max_lookahead = min(n + 5, 128)
            for m in range(n + 1, max_lookahead + 1):
                v_m = vecs[m]
                new_p_val = [last_p[i] + v_m[i] for i in range(num_primes)]
                new_p = tuple(new_p_val)
                ok = True
                for p in pl:
                    d_val = [new_p[i] - p[i] for i in range(num_primes)]
                    if tuple(d_val) in ds:
                        ok = False
                        break
                if ok:
                    potential += 1
            # Score: Length is primary, Potential is secondary
            return (l, potential)

        new_beam.sort(key=calc_score, reverse=True)
        beam = new_beam[:beam_width]
        
        # Update best
        if beam:
            current_best = beam[0]
            if current_best[0] > best_length:
                best_length = current_best[0]
                best_seq = current_best[1]
        
        # Early stop if best possible is reached
        if best_length == 127:
            return best_seq

    # Local refinement: try to add more elements by checking compatibility
    if best_seq and time.time() < deadline - 10:
        # Try to extend the sequence by checking which remaining numbers can be added
        remaining = [i for i in range(2, 129) if i not in best_seq]
        # Sort by how "compatible" they are (fewer conflicts)
        current_seq = list(best_seq)
        
        # Rebuild prefix and diff sets for the current sequence
        pref_list = [zero_vec]
        diff_set = frozenset()
        for x in current_seq:
            v_x = vecs[x]
            last = pref_list[-1]
            new_p = tuple(last[i] + v_x[i] for i in range(num_primes))
            pref_list.append(new_p)
            for p in pref_list[:-1]:
                d = tuple(new_p[i] - p[i] for i in range(num_primes))
                diff_set = diff_set.union([d])
        
        # Try to insert remaining numbers one by one
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in sorted(remaining, key=lambda y: sum(1 for z in current_seq if z > y)):
                if x in current_seq:
                    continue
                # Try inserting x at each possible position
                v_x = vecs[x]
                for pos in range(len(current_seq) + 1):
                    # Check if inserting x at pos is valid
                    # Build prefix list up to pos
                    if pos == 0:
                        pref_before = [zero_vec]
                    else:
                        pref_before = pref_list[:pos + 1]
                    
                    last_before = pref_before[-1]
                    new_p = tuple(last_before[i] + v_x[i] for i in range(num_primes))
                    
                    # Check conflicts with all prefixes before pos
                    ok = True
                    for p in pref_before:
                        d = tuple(new_p[i] - p[i] for i in range(num_primes))
                        if d in diff_set:
                            ok = False
                            break
                    
                    if not ok:
                        continue
                    
                    # Check conflicts with suffix
                    # We need to verify that adding x doesn't create conflicts with existing suffix blocks
                    # This is complex, so we'll do a full validation
                    test_seq = current_seq[:pos] + [x] + current_seq[pos:]
                    
                    # Full validation
                    test_pref = [zero_vec]
                    test_diffs = set()
                    valid = True
                    for val in test_seq:
                        v_val = vecs[val]
                        last = test_pref[-1]
                        np = tuple(last[i] + v_val[i] for i in range(num_primes))
                        test_pref.append(np)
                        for p in test_pref[:-1]:
                            d = tuple(np[i] - p[i] for i in range(num_primes))
                            if d in test_diffs:
                                valid = False
                                break
                            test_diffs.add(d)
                        if not valid:
                            break
                    
                    if valid:
                        current_seq = test_seq
                        remaining.remove(x)
                        changed = True
                        break
        
        if len(current_seq) > best_length:
            best_seq = current_seq
            best_length = len(current_seq)

    if best_seq:
        return best_seq
    return []
