import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def greedy_build(start_order=None):
        """Greedy: try candidates in given order, add if valid."""
        values = []
        sums = set()
        candidates = list(start_order) if start_order else list(range(1, 257))
        for x in candidates:
            if values and x <= values[-1]:
                continue
            # Compute all new consecutive block sums ending at x
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums
    
    # Phase 1: Greedy baseline
    best_seq, best_sums = greedy_build()
    best_len = len(best_seq)
    
    # Phase 2: Try to improve with backtracking
    # At each step, we can try different candidates
    # Use iterative deepening or limited backtracking
    
    def try_extend(values, sums, max_depth=3, timeout=None):
        """Try to extend the sequence by backtracking up to max_depth levels."""
        if timeout is not None and time.time() - start_time > timeout:
            return values, sums
        
        if max_depth <= 0:
            return values, sums
        
        last_val = values[-1] if values else 0
        # Try candidates from last_val+1 to 256
        # Use a heuristic: try candidates that create "spread" in sums
        best_local = list(values)
        best_local_sums = set(sums)
        
        for x in range(last_val + 1, 257):
            if time_left() < 5:
                break
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
            if sums.intersection(tails):
                continue
            
            new_values = values + [x]
            new_sums = sums | set(tails)
            
            # Recurse
            ext_values, ext_sums = try_extend(new_values, new_sums, max_depth - 1, timeout)
            if len(ext_values) > len(best_local):
                best_local = ext_values
                best_local_sums = ext_sums
        
        return best_local, best_local_sums
    
    # Try to improve from greedy result
    if time_left() > 10:
        improved, imp_sums = try_extend(best_seq, best_sums, max_depth=2, timeout=start_time + DEADLINE - 10)
        if len(improved) > best_len:
            best_seq, best_sums, best_len = improved, imp_sums, len(improved)
    
    # Phase 3: Try different greedy orderings
    if time_left() > 15:
        # Try descending
        desc_order = list(range(256, 0, -1))
        desc_seq, desc_sums = greedy_build(desc_order)
        if len(desc_seq) > best_len:
            best_seq, best_sums, best_len = desc_seq, desc_sums, len(desc_seq)
        
        # Try a few random orderings
        rng = random.Random(42)
        for trial in range(50):
            if time_left() < 10:
                break
            order = list(range(1, 257))
            rng.shuffle(order)
            # Ensure strictly increasing: sort the chosen subset
            # Actually greedy with random order doesn't work directly since we need strictly increasing
            # We need to pick a subset and sort it
            # Alternative: try different "gaps" strategies
            pass
    
    # Phase 4: More sophisticated search - try to find longer sequences
    # Key insight: we want elements that are "far apart" to avoid sum collisions
    # Try exponential spacing or other patterns
    
    if time_left() > 10:
        # Try geometric-like sequences
        for base in [2, 3, 4, 5]:
            if time_left() < 5:
                break
            seq = []
            v = 1
            while v <= 256:
                seq.append(v)
                v *= base
            # Now try to fill gaps
            # Actually, let's just use greedy but with a bias toward larger gaps
            pass
    
    # Phase 5: Local search / hill climbing on the greedy result
    if time_left() > 10:
        # Try replacing elements to allow longer sequences
        # Or try to insert elements into the existing sequence
        current = list(best_seq)
        current_sums = set(best_sums)
        
        # Try inserting elements between existing ones
        improved = False
        while not improved and time_left() > 5:
            improved = True
            for i in range(len(current)):
                if time_left() < 5:
                    break
                lo = current[i-1] + 1 if i > 0 else 1
                hi = current[i+1] - 1 if i < len(current) - 1 else 256
                # For inserting at position i, the new element must be > current[i-1] and < current[i]
                # Wait, the sequence is strictly increasing, so inserting at position i means
                # the new element goes between current[i-1] and current[i]
                # Actually, let me reconsider: current is sorted, inserting between index i-1 and i
                for j in range(i, len(current)):
                    if time_left() < 5:
                        break
                    # Insert x at position j: must be > current[j-1] (if j>0) and < current[j] (if j<len)
                    lo_ins = current[j-1] + 1 if j > 0 else 1
                    hi_ins = current[j] - 1 if j < len(current) else 256
                    for x in range(lo_ins, hi_ins + 1):
                        # Check if inserting x at position j creates valid sums
                        new_seq = current[:j] + [x] + current[j:]
                        # Compute all consecutive block sums
                        new_sums = set()
                        valid = True
                        for a in range(len(new_seq)):
                            s = 0
                            for b in range(a, len(new_seq)):
                                s += new_seq[b]
                                if s in new_sums:
                                    valid = False
                                    break
                                new_sums.add(s)
                            if not valid:
                                break
                        if valid:
                            current = new_seq
                            current_sums = new_sums
                            improved = False
                            break
                    if not improved:
                        break
                if not improved:
                    break
        
        if len(current) > best_len:
            best_seq, best_len = current, len(current)
    
    return best_seq
