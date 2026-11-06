"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85  # 85s safety margin (limit is 90s)
    
    def greedy_randomized(seed):
        """Randomized greedy: at each step, pick from available candidates with some randomness."""
        rng = random.Random(seed)
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums
    
    def greedy_standard():
        """Standard greedy: pick smallest available x at each step."""
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums
    
    def backtrack_extend(seq, sums, best_holder):
        """Backtracking to try to extend a given sequence beyond its current length."""
        if time.time() > deadline:
            return
        
        current_len = len(seq)
        if current_len > len(best_holder[0]):
            best_holder[0] = list(seq)
        
        last = seq[-1] if seq else 0
        
        # Pruning: if we can't possibly beat the best, return
        if current_len + (256 - last) <= len(best_holder[0]):
            return
        
        # Try candidates from last+1 to 256
        for x in range(last + 1, 257):
            if time.time() > deadline:
                return
            
            # Compute new tails
            tails = []
            total = x
            tails.append(total)
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            # Check if any tail is in sums
            if any(t in sums for t in tails):
                continue
            
            # Valid - recurse
            seq.append(x)
            sums.update(tails)
            backtrack_extend(seq, sums, best_holder)
            sums.difference_update(tails)
            seq.pop()
    
    def local_search(seq, sums, best_holder):
        """Try local improvements: swap, remove-and-reinsert, etc."""
        if time.time() > deadline:
            return
        
        n = len(seq)
        if n == 0:
            return
        
        # Try removing each element and re-adding a different one
        for i in range(n):
            if time.time() > deadline:
                return
            
            # Remove element i
            removed = seq[i]
            removed_tails = []
            total = 0
            for j in range(i, -1, -1):
                total += seq[j]
                removed_tails.append(total)
            
            sums.difference_update(removed_tails)
            seq.pop(i)
            
            # Try adding different elements at position i
            prev_val = seq[i-1] if i > 0 else 0
            next_val = seq[i] if i < len(seq) else 257
            
            for x in range(prev_val + 1, next_val):
                if time.time() > deadline:
                    break
                if x == removed:
                    continue
                
                # Check if x can be inserted at position i
                valid = True
                # Check tails ending at x
                total = x
                if total in sums:
                    valid = False
                else:
                    for a in reversed(seq[:i]):
                        total += a
                        if total in sums:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Check sums that start after x (affected by insertion)
                # Actually, we need to check all new consecutive block sums
                # This is complex; let's use a simpler check
                # Insert x and verify
                seq.insert(i, x)
                
                # Recompute all sums and check for duplicates
                all_sums = set()
                for start in range(len(seq)):
                    s = 0
                    for end in range(start, len(seq)):
                        s += seq[end]
                        if s in all_sums:
                            valid = False
                            break
                        all_sums.add(s)
                    if not valid:
                        break
                
                if valid:
                    if len(seq) > len(best_holder[0]):
                        best_holder[0] = list(seq)
                    # Keep the insertion
                    sums = all_sums
                    break
                else:
                    seq.pop(i)
            
            # If we didn't find a better insertion, restore original
            if len(seq) < n:
                seq.insert(i, removed)
                # Recompute sums
                all_sums = set()
                for start in range(len(seq)):
                    s = 0
                    for end in range(start, len(seq)):
                        s += seq[end]
                        all_sums.add(s)
                sums = all_sums
    
    # Get baseline
    base_seq, base_sums = greedy_standard()
    best = base_seq
    best_holder = [best]
    
    # Try backtracking extension from baseline
    backtrack_extend(list(base_seq), set(base_sums), best_holder)
    
    # Try randomized greedy searches
    num_attempts = 0
    for seed in range(100000):
        if time.time() > deadline:
            break
        
        seq, sums = greedy_randomized(seed)
        if len(seq) > len(best_holder[0]):
            best_holder[0] = list(seq)
            # Try to extend this better sequence
            backtrack_extend(list(seq), set(sums), best_holder)
        
        num_attempts += 1
        if num_attempts % 500 == 0:
            # Occasionally try local search on best
            if len(best_holder[0]) > len(base_seq):
                local_search(list(best_holder[0]), set(), best_holder)
    
    # Final: try backtracking from the best found
    if len(best_holder[0]) > len(base_seq):
        # Recompute sums for best
        best_seq = best_holder[0]
        best_sums = set()
        for start in range(len(best_seq)):
            s = 0
            for end in range(start, len(best_seq)):
                s += best_seq[end]
                best_sums.add(s)
        
        if time.time() < deadline - 10:
            backtrack_extend(list(best_seq), best_sums, best_holder)
    
    return best_holder[0]
# EVOLVE-BLOCK-END
