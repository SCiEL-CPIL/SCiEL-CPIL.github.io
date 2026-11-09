import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # State: (sequence, sums_set, suffix_sums)
    # suffix_sums: list of sums of suffixes of sequence
    # e.g., seq=[1, 3], suffix_sums=[4, 3]
    # sums_set: set of all consecutive sums
    
    beam = [([], set(), [])]
    best_seq = []
    max_len = 0
    beam_width = 300  # Moderate beam width, lookahead allows for smaller width potentially
    
    while True:
        if time.time() > deadline:
            break
            
        next_beam = []
        
        for seq, sums_set, suffix_sums in beam:
            last_val = seq[-1] if seq else 0
            n = len(seq)
            
            # Try all x > last_val
            for x in range(last_val + 1, 257):
                # Calculate new suffix sums and check validity
                # new_suffix_sums = [x + s for s in reversed(suffix_sums)] + [x] ? 
                # Wait, suffix_sums is defined as [sum(seq), ..., last(seq)] in current code?
                # Let's check current code definition:
                # "suffix_sums: list of sums of suffixes of sequence"
                # "e.g., seq=[1, 3], suffix_sums=[4, 3]"
                # So suffix_sums[0] is total sum, suffix_sums[-1] is last element.
                
                # New sequence is seq + [x]
                # New suffixes:
                # 1. x
                # 2. x + last(seq)
                # 3. x + second_last(seq) + last(seq)
                # ...
                # k. x + sum(seq)
                
                # In current code:
                # new_suffix_sums_list starts with x.
                # Then appends x + s for s in suffix_sums.
                # Since suffix_sums is [total, ..., last], x + total is the largest new sum.
                # So new_suffix_sums_list is [x, x+last, ..., x+total].
                # This is NOT sorted. The set of sums doesn't care about order, but for lookahead,
                # we just need the set of new sums.
                
                valid = True
                new_sums_added = set()
                
                # Check x
                if x in sums_set:
                    continue
                new_sums_added.add(x)
                
                # Check x + s for s in suffix_sums
                for s in suffix_sums:
                    ns = x + s
                    if ns in sums_set:
                        valid = False
                        break
                    new_sums_added.add(ns)
                
                if not valid:
                    continue
                
                # New state components
                new_seq = seq + [x]
                new_sums_set = sums_set.union(new_sums_added)
                # new_suffix_sums for next step:
                # We need [new_total, ..., x]
                # new_total = x + suffix_sums[0] (if suffix_sums not empty) else x
                # The list of suffix sums for new_seq is [x + s for s in suffix_sums] + [x]
                # But we need it in the same format: [total, ..., last]
                # So: new_suffix_sums = [x + s for s in suffix_sums] + [x]
                # Note: suffix_sums[0] is total of seq. So x + suffix_sums[0] is total of new_seq.
                # This matches the format.
                
                new_suffix_sums = [x + s for s in suffix_sums] + [x]
                
                # --- Lookahead Check ---
                # Check if there exists y > x such that y is valid for new_seq
                # If not, this is a dead end. Discard.
                # If yes, record the smallest such y as heuristic score.
                
                min_valid_next = 257 # Sentinel
                found_next = False
                
                # Iterate y > x
                # Optimization: We can break early if we find a valid y?
                # We want the SMALLEST valid y for the heuristic.
                # So we must check all y from x+1 upwards until we find one?
                # Yes, the first valid one is the smallest.
                
                for y in range(x + 1, 257):
                    # Check if y is valid for new_seq
                    # 1. y not in new_sums_set
                    if y in new_sums_set:
                        continue
                    
                    # 2. y + s not in new_sums_set for all s in new_suffix_sums
                    valid_y = True
                    for s in new_suffix_sums:
                        if y + s in new_sums_set:
                            valid_y = False
                            break
                    
                    if valid_y:
                        min_valid_next = y
                        found_next = True
                        break # We only need the smallest one for the heuristic, and we know it's live
                
                if not found_next:
                    # Dead end, do not add to next_beam
                    continue
                
                # Heuristic Score:
                # 1. min_valid_next (smaller is better)
                # 2. total_sum (smaller is better) -> new_suffix_sums[0]
                # 3. last_val (smaller is better) -> x
                
                score = (min_valid_next, new_suffix_sums[0], x)
                
                next_beam.append((score, new_seq, new_sums_set, new_suffix_sums))
        
        if not next_beam:
            break
        
        # Sort by score
        next_beam.sort(key=lambda item: item[0])
        
        if len(next_beam) > beam_width:
            next_beam = next_beam[:beam_width]
            
        # Update best
        # The best sequence in the current layer is the first one (smallest score)
        # But we should track the global best length.
        # All sequences in next_beam have the same length.
        cur_len = len(next_beam[0][1])
        if cur_len > max_len:
            max_len = cur_len
            best_seq = next_beam[0][1]
            
        # Prepare beam for next iteration
        # Remove the score tuple
        beam = [(seq, sums, suffix) for (_, seq, sums, suffix) in next_beam]
        
    return best_seq
