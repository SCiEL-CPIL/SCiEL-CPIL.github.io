import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors as numpy arrays
    vecs = {}
    for i in range(2, 129):
        v = np.zeros(num_primes, dtype=np.int8)
        temp = i
        for idx, p in enumerate(primes):
            while temp % p == 0:
                v[idx] += 1
                temp //= p
            if temp == 1:
                break
        vecs[i] = v
    zero_vec = np.zeros(num_primes, dtype=np.int8)
    
    # Beam State: (length, seq, prefix_array, diff_set)
    beam = [(0, [], np.array([zero_vec]), frozenset())]
    
    best_seq = []
    best_length = 0
    
    for n in range(2, 129):
        if time.time() > deadline:
            break
            
        # Dynamic beam width
        if n < 15:
            beam_width = 250
        elif n < 40:
            beam_width = 120
        elif n < 80:
            beam_width = 60
        else:
            beam_width = 30
            
        new_beam = []
        v_n = vecs[n]
        
        for state in beam:
            length, seq, pref_arr, diff_set = state
            last_pref = pref_arr[-1]
            
            # Option 1: Exclude
            new_beam.append((length, seq, pref_arr, diff_set))
            
            # Option 2: Include
            new_pref = last_pref + v_n
            
            valid = True
            new_diffs = []
            
            # Check all previous prefixes using numpy
            diffs = new_pref - pref_arr  # shape (len, num_primes)
            for i in range(len(pref_arr)):
                diff_tup = tuple(diffs[i])
                if diff_tup in diff_set:
                    valid = False
                    break
                new_diffs.append(diff_tup)
            
            if valid:
                new_pref_arr = np.vstack([pref_arr, new_pref.reshape(1, -1)])
                new_diff_set = diff_set.union(new_diffs)
                new_seq = seq + [n]
                new_length = length + 1
                new_beam.append((new_length, new_seq, new_pref_arr, new_diff_set))
        
        # Score function: length + limited lookahead
        def calc_score(state):
            l, seq, pa, ds = state
            last_p = pa[-1]
            potential = 0
            # Lookahead: next 3 numbers
            max_look = min(n + 3, 128)
            for m in range(n + 1, max_look + 1):
                v_m = vecs[m]
                new_p = last_p + v_m
                # Check against all prefixes
                diffs = new_p - pa
                ok = True
                for i in range(len(pa)):
                    if tuple(diffs[i]) in ds:
                        ok = False
                        break
                if ok:
                    potential += 1
            return (l, potential)

        new_beam.sort(key=calc_score, reverse=True)
        beam = new_beam[:beam_width]
        
        # Update best
        if beam:
            current_best = beam[0]
            if current_best[0] > best_length:
                best_length = current_best[0]
                best_seq = list(current_best[1])
        
        # Early stop if best possible is reached
        if best_length == 127:
            return best_seq

    # Local insertion pass: try to insert missed numbers into best_seq
    if best_seq and best_length < 127:
        # Build prefix array and diff set for best_seq
        best_arr_list = [zero_vec.copy()]
        for x in best_seq:
            best_arr_list.append(best_arr_list[-1] + vecs[x])
        best_arr = np.array(best_arr_list)
        
        # Build diff set
        best_diffs = set()
        for i in range(len(best_arr)):
            for j in range(i + 1, len(best_arr)):
                best_diffs.add(tuple(best_arr[j] - best_arr[i]))
        
        # Try inserting each missing number
        in_seq = set(best_seq)
        missing = [x for x in range(2, 129) if x not in in_seq]
        
        # Repeatedly try to insert the best candidate
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            best_insert = None
            best_pos = -1
            
            for x in missing:
                # Find valid insertion positions
                for pos in range(len(best_seq) + 1):
                    if pos > 0 and x <= best_seq[pos - 1]:
                        continue
                    if pos < len(best_seq) and x >= best_seq[pos]:
                        continue
                    
                    # Check validity of inserting x at position pos
                    # New prefix at pos: best_arr[pos] + vecs[x]
                    # All subsequent prefixes shift by vecs[x]
                    
                    new_pref = best_arr[pos] + vecs[x]
                    
                    # Check new_pref against all existing prefixes (0..pos-1)
                    ok = True
                    for i in range(pos):
                        d = tuple(new_pref - best_arr[i])
                        if d in best_diffs:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # Check new_pref against shifted suffix (pos+1..end)
                    # The shifted suffix prefixes are best_arr[k] + vecs[x] for k > pos
                    # Diff between new_pref and shifted prefix at k: (best_arr[pos]+vecs[x]) - (best_arr[k]+vecs[x]) = best_arr[pos]-best_arr[k]
                    # These are already in best_diffs (as negatives of existing diffs)
                    # Actually we need: new_pref - (best_arr[k]+vecs[x]) = best_arr[pos] - best_arr[k]
                    # And (best_arr[k]+vecs[x]) - new_pref = best_arr[k] - best_arr[pos]
                    # Both of these are already in best_diffs since they existed before
                    # So the only NEW diffs are between new_pref and prefixes 0..pos-1
                    # AND between shifted suffix and prefixes 0..pos-1
                    # Shifted suffix prefix at k (k>pos): best_arr[k]+vecs[x]
                    # Diff with prefix i (i<pos): (best_arr[k]+vecs[x]) - best_arr[i] = (best_arr[k]-best_arr[i]) + vecs[x]
                    # This is a NEW diff not in best_diffs
                    
                    for i in range(pos):
                        for k in range(pos + 1, len(best_arr)):
                            d = tuple(best_arr[k] + vecs[x] - best_arr[i])
                            if d in best_diffs:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    
                    # Also check: shifted suffix internal diffs are unchanged (same as before)
                    # And new_pref to shifted suffix diffs are same as old (best_arr[pos]-best_arr[k])
                    # So we just need to verify the cross terms above
                    
                    if best_insert is None or x < best_insert:
                        best_insert = x
                        best_pos = pos
                        improved = True
            
            if improved:
                # Perform insertion
                new_seq = best_seq[:best_pos] + [best_insert] + best_seq[best_pos:]
                best_seq = new_seq
                best_length += 1
                
                # Rebuild arrays
                best_arr_list = [zero_vec.copy()]
                for x in best_seq:
                    best_arr_list.append(best_arr_list[-1] + vecs[x])
                best_arr = np.array(best_arr_list)
                
                best_diffs = set()
                for i in range(len(best_arr)):
                    for j in range(i + 1, len(best_arr)):
                        best_diffs.add(tuple(best_arr[j] - best_arr[i]))
                
                missing.remove(best_insert)
                
                if best_length == 127:
                    return best_seq

    return best_seq
