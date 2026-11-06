import time
import sys
import random

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def is_time_up():
        return time.time() > deadline

    def greedy(order):
        values, sums = [], set()
        for x in order:
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        values.sort()
        return values

    # Multiple greedy restarts
    best_seq = greedy(range(1, 257))
    best_len = len(best_seq)

    # Try descending
    seq_desc = greedy(range(256, 0, -1))
    if len(seq_desc) > best_len:
        best_seq = seq_desc
        best_len = len(best_seq)

    if is_time_up():
        return best_seq

    # Random restarts
    rng = random.Random(42)
    for trial in range(30):
        if is_time_up():
            break
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        seq = greedy(candidates)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)

    if is_time_up():
        return best_seq

    # Local search: try to insert additional elements into the best sequence
    # This is the key improvement from inspiration 1

    def compute_all_sums(seq):
        n = len(seq)
        sums = set()
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def can_insert_at_position(seq, x, pos, current_sums):
        """Check if inserting x at position pos keeps all block sums distinct."""
        n = len(seq)
        # New blocks involving x:
        # 1. Blocks ending at x: sum of seq[i..pos-1] + x for i in [0..pos]
        # 2. Blocks starting at x: x + sum of seq[pos..j] for j in [pos..n-1]
        # 3. Blocks spanning x: sum of seq[i..pos-1] + x + sum of seq[pos..j]
        
        new_sums = set(current_sums)
        
        # Check blocks ending at x (suffixes of prefix ending at pos-1, plus x)
        total = x
        for i in range(pos - 1, -1, -1):
            if total in new_sums:
                return False
            new_sums.add(total)
            total += seq[i]
        # Also just x itself
        if x in new_sums:
            return False
        new_sums.add(x)
        
        # Check blocks starting at x (prefixes of suffix starting at pos, plus x)
        total = x
        for j in range(pos, n):
            if total in new_sums:
                return False
            new_sums.add(total)
            total += seq[j]
        
        # Check blocks spanning x
        # For each i < pos, j >= pos: sum(seq[i..pos-1]) + x + sum(seq[pos..j])
        prefix_sums = {}
        total = 0
        for i in range(pos - 1, -1, -1):
            total += seq[i]
            prefix_sums[i] = total
        
        suffix_totals = [0]
        total = 0
        for j in range(pos, n):
            total += seq[j]
            suffix_totals.append(total)
        
        for i in range(pos):
            p = prefix_sums[i]
            for j in range(pos, n):
                s = p + x + suffix_totals[j - pos + 1]
                if s in new_sums:
                    return False
                new_sums.add(s)
        
        return True

    # More efficient insertion check
    def try_insertion(current, current_sums, unused, rng, max_trials=50):
        """Try to insert elements from unused into current to increase length."""
        improved = True
        trials = 0
        while improved and trials < max_trials and not is_time_up():
            improved = False
            trials += 1
            
            # Try a random subset of unused elements
            trial_unused = unused[:min(20, len(unused))]
            rng.shuffle(trial_unused)
            
            for x in trial_unused:
                # Try random positions
                n = len(current)
                positions = list(range(n + 1))
                rng.shuffle(positions)
                
                for pos in positions:
                    # Check if x can be inserted at pos
                    # Quick check: x itself must not be in current_sums
                    if x in current_sums:
                        continue
                    
                    # Full check
                    if can_insert_at_position(current, x, pos, current_sums):
                        # Insert
                        new_current = current[:pos] + [x] + current[pos:]
                        new_current.sort()
                        
                        # Verify
                        new_sums = compute_all_sums(new_current)
                        if len(new_sums) == len(new_current) * (len(new_current) + 1) // 2:
                            current = new_current
                            current_sums = new_sums
                            unused.remove(x)
                            improved = True
                            break
                
                if improved:
                    break
        
        return current, current_sums, unused

    # Run local search from the best sequence
    current = best_seq[:]
    current_sums = compute_all_sums(current)
    used = set(current)
    unused = [x for x in range(1, 257) if x not in used]
    
    current, current_sums, unused = try_insertion(current, current_sums, unused, rng, max_trials=100)
    
    if len(current) > best_len:
        best_seq = current
        best_len = len(best_seq)

    if is_time_up():
        return best_seq

    # Try beam search from scratch for a potentially better structure
    # Using the beam search approach from the current program
    beam = [(0, (), frozenset())]
    max_beam_len = 0
    best_beam_seq = []

    while beam and not is_time_up():
        new_beam = []
        for last, seq, sums in beam:
            for x in range(last + 1, 257):
                if x in sums:
                    continue
                
                valid = True
                total = x
                if total in sums:
                    valid = False
                else:
                    s = total
                    for a in reversed(seq):
                        s += a
                        if s in sums:
                            valid = False
                            break
                
                if valid:
                    new_seq = seq + (x,)
                    new_sums_set = set(sums)
                    total = x
                    new_sums_set.add(total)
                    for a in reversed(seq):
                        total += a
                        new_sums_set.add(total)
                    
                    new_beam.append((x, new_seq, frozenset(new_sums_set)))
        
        if not new_beam:
            break
            
        B = 50
        if len(new_beam) > B:
            new_beam.sort(key=lambda s: s[0])
            new_beam = new_beam[:B]
        
        beam = new_beam
        
        if len(beam[0][1]) > max_beam_len:
            max_beam_len = len(beam[0][1])
            best_beam_seq = list(beam[0][1])

    if max_beam_len > best_len:
        best_seq = best_beam_seq
        best_len = max_beam_len

    return best_seq
