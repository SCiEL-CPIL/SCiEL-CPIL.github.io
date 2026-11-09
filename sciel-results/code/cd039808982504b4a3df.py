import time
import sys

def construct():
    start_time = time.time()
    # The evaluator kills at 90s. We should aim to finish well before.
    # Let's use 80s as a hard deadline for the search logic.
    deadline = start_time + 80.0

    def _greedy():
        """
        Greedy construction: pick smallest x that doesn't conflict.
        Returns the sequence and the set of used sums (as a bitset int).
        """
        values = []
        used_sums = 0  # Bitset of used sums
        
        for x in range(1, 257):
            # Calculate new sums introduced by x
            # Blocks ending at x: [x], [prev, x], [prev2, prev, x], ...
            # These are suffix sums of the new sequence ending at x.
            # Let's compute them and check for conflicts.
            
            # New sums are: x, x+values[-1], x+values[-1]+values[-2], ...
            # We need to check if any of these bits are already set in used_sums.
            
            # Optimization: If x itself is in used_sums, we can skip immediately?
            # Not necessarily, but x is the smallest new sum.
            
            new_sums_mask = 0
            s = x
            # Check x
            if (used_sums >> x) & 1:
                continue
            new_sums_mask |= (1 << x)
            
            valid = True
            # Iterate backwards through values to build suffix sums
            for a in reversed(values):
                s += a
                # Max sum is roughly 15*256 = 3840. 
                # Python ints handle this, but shifting by 3840 is fine.
                if (used_sums >> s) & 1:
                    valid = False
                    break
                new_sums_mask |= (1 << s)
            
            if valid:
                values.append(x)
                used_sums |= new_sums_mask
                
        return values, used_sums

    # Get baseline
    best_seq, best_sums = _greedy()
    best_len = len(best_seq)
    
    # If greedy is already very long, we might not need heavy search.
    # But let's assume it's around 9-10.

    def _beam_search(target_len, width=200):
        """
        Beam search to find a sequence of length target_len.
        State: (sequence, used_sums_bitset, last_val)
        """
        # Initialize beam with empty sequence
        # We can also initialize with the greedy prefix to speed up?
        # No, beam search from scratch is more robust for finding *different* structures.
        
        beam = [([], 0, 0)]
        best_found = None
        
        for depth in range(target_len):
            new_beam = []
            if not beam:
                break
            
            # Sort beam by some heuristic? 
            # We want to keep the most "promising" states.
            # Promising = smaller last_val (more room), smaller max_sum?
            # Let's just keep the first 'width' valid extensions, 
            # but prioritize smaller last_val to allow more future options.
            
            # To ensure we don't miss the best, we should sort candidates 
            # before truncating the beam.
            
            for seq, sums, last_val in beam:
                # Try to extend with x > last_val
                # Heuristic: Try smaller x first? 
                # In beam search, we generate all valid extensions, then sort and truncate.
                
                # Optimization: 
                # If we are at depth d, we need target_len - d - 1 more elements.
                # Max possible elements remaining is 256 - last_val.
                # If 256 - last_val < target_len - depth - 1, we can't reach target_len.
                # But we are checking this per candidate x.
                
                for x in range(last_val + 1, 257):
                    # Check time
                    if time.time() > deadline - 1.0:
                        return best_found # Return whatever we have, or None
                    
                    # Calculate new sums
                    # New sums: x, x+last_val, ...
                    # We need to check collisions with 'sums'.
                    
                    # Quick check: if x is in sums, skip
                    if (sums >> x) & 1:
                        continue
                    
                    new_sums_mask = 1 << x
                    s = x
                    valid = True
                    # Iterate backwards through seq
                    # seq is a list. Accessing from end is O(1).
                    for a in reversed(seq):
                        s += a
                        if (sums >> s) & 1:
                            valid = False
                            break
                        new_sums_mask |= (1 << s)
                    
                    if not valid:
                        continue
                    
                    new_sums = sums | new_sums_mask
                    new_seq = seq + [x]
                    
                    # Pruning: Can we still reach target_len?
                    # We need (target_len - depth - 1) more elements.
                    # Available numbers: 256 - x.
                    if 256 - x < target_len - depth - 1:
                        continue
                        
                    new_beam.append((new_seq, new_sums, x))
            
            if not new_beam:
                return None
            
            # Sort new_beam to keep the best candidates
            # Heuristic: 
            # 1. Prefer smaller last_val (more flexibility)
            # 2. Prefer smaller total sum? (less collision risk later)
            # Let's sort by last_val ascending.
            # If we want to find *any* solution of length L, the order of truncation matters.
            # Keeping smaller last_val is generally good for extending further.
            
            new_beam.sort(key=lambda state: state[2])
            
            # Truncate to width
            beam = new_beam[:width]
            
            # If we reached the target depth, any state in beam is a valid solution
            if depth == target_len - 1:
                # Return the first one (smallest last_val) or the best one?
                # They are all valid.
                return beam[0][0]

        return None

    # Try to find longer sequences
    # Start from best_len + 1
    # We can increase the width for longer targets if time permits, 
    # but usually a fixed width is fine.
    
    found_longer = False
    for target_len in range(best_len + 1, 15): # Cap at 14/15 for safety
        if time.time() > deadline - 5.0:
            break
        
        # Increase width for harder targets? 
        # Let's start with 100 and increase if it fails quickly?
        # Or just use a reasonably large width like 500.
        # Memory: 500 states * list of ~15 ints. Very small.
        
        res = _beam_search(target_len, width=500)
        
        if res is not None:
            best_seq = res
            best_len = target_len
            found_longer = True
        else:
            # If beam search fails to find length L, it's unlikely to find L+1 
            # with the same width, but not impossible if the structure is very specific.
            # However, if the search space is exhausted (not just pruned by width), 
            # we should stop.
            # Beam search returning None means no path of that length was found in the beam.
            # It's safer to break here to save time for other attempts if we had them.
            # Since we are iterating increasing lengths, if L fails, L+1 is harder.
            break

    # If we didn't find a longer sequence with beam search, 
    # we might try a DFS extension of the greedy sequence specifically?
    # The current program did this. 
    # Let's add a quick DFS extension check if we have time left.
    
    if time.time() < deadline - 2.0:
        # Try to extend the current best_seq using DFS
        # This is a targeted search.
        
        def _dfs_extend(vals, last_val, used_bits, depth, target_len):
            if time.time() > deadline - 1.0:
                return None
            if depth == target_len:
                return vals[:]
            
            # Pruning
            if 256 - last_val < target_len - depth:
                return None
                
            for x in range(last_val + 1, 257):
                if (used_bits >> x) & 1:
                    continue
                
                new_mask = 1 << x
                s = x
                valid = True
                for a in reversed(vals):
                    s += a
                    if (used_bits >> s) & 1:
                        valid = False
                        break
                    new_mask |= (1 << s)
                
                if not valid:
                    continue
                
                res = _dfs_extend(vals + [x], x, used_bits | new_mask, depth + 1, target_len)
                if res:
                    return res
            return None

        # Try to extend best_seq by 1
        # We need the bitset for best_seq
        # Recalculate bitset for best_seq
        bs = 0
        # Efficiently calculate all consecutive sums for best_seq
        # O(L^2)
        for i in range(len(best_seq)):
            s = 0
            for j in range(i, len(best_seq)):
                s += best_seq[j]
                bs |= (1 << s)
        
        ext = _dfs_extend(best_seq, best_seq[-1], bs, len(best_seq), len(best_seq) + 1)
        if ext:
            best_seq = ext
            best_len = len(best_seq)

    return best_seq

# Set recursion limit just in case, though we are using iterative beam search
sys.setrecursionlimit(10000)

if __name__ == "__main__":
    # Test locally
    seq = construct()
    print(f"Length: {len(seq)}")
    print(f"Sequence: {seq}")
    
    # Verify
    if seq:
        sums = set()
        valid = True
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        
        print(f"Valid: {valid}")
        if valid:
            # Check strictly increasing
            is_inc = all(seq[i] < seq[i+1] for i in range(len(seq)-1))
            in_range = all(1 <= x <= 256 for x in seq)
            print(f"Increasing: {is_inc}, In Range: {in_range}")
