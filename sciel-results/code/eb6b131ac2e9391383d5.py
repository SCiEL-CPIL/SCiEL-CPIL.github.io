import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Heuristic: We want to maximize length.
    # In distinct sums problems, larger gaps are often better to avoid collisions.
    # We will try to build the sequence by picking values from the high end downwards
    # or using a smart greedy that prioritizes values that don't collide.
    
    # Strategy 1: Greedy Backtracking with Forward Checking
    # State: current sequence, set of used sums, list of suffix sums.
    # When adding x, new suffix sums are {x, x + s_1, x + s_2, ...} where s_i are old suffix sums.
    # Check if any new sum is in used_sums. If not, add.
    
    def solve_greedy_backtrack(max_val=256, min_val=1, time_limit=20.0):
        """
        Uses a greedy approach with backtracking.
        We iterate through potential values. To maximize length, we might want to
        pick values that allow more future additions.
        However, a simple greedy "pick smallest valid" or "pick largest valid" might get stuck.
        
        A better heuristic for "distinct sums" is often to pick values that are powers of 2
        or similar, but we are restricted to [1, 256].
        
        Let's use a recursive backtracking that tries to insert values.
        Since n is likely small (< 15), we can afford some depth.
        
        Actually, let's stick to a robust Greedy with local search refinement if greedy fails to reach high lengths.
        But the prompt asks for a DIFFERENT strategy.
        
        Let's try a "Greedy with Lookahead" or "Constraint Propagation".
        
        Revised Strategy:
        1. Generate a set of candidate sequences using different greedy heuristics (e.g., prefer large gaps, prefer small values).
        2. Use a fast incremental validity check.
        3. If time permits, run a targeted local search on the best greedy result to see if we can insert more elements.
        """
        best_seq = []
        
        def try_greedy(prefer_large=True):
            seq = []
            used_sums = set()
            suffix_sums = [] # List of suffix sums of current seq
            
            # We iterate through all possible values 1..256
            # To make it deterministic-ish but varied, we can shuffle the order of consideration
            # or just iterate 1..256.
            
            candidates = list(range(1, 257))
            if prefer_large:
                candidates.reverse()
            
            for x in candidates:
                # Check if x can be appended to seq
                # New suffix sums: x, x + suffix_sums[-1], x + suffix_sums[-2], ...
                # Wait, suffix_sums contains sums of suffixes of the CURRENT seq.
                # If seq = [a, b, c], suffix sums are [c, b+c, a+b+c].
                # If we append x, new suffix sums are [x, x+c, x+b+c, x+a+b+c].
                
                new_suffixes = [x]
                current_sum = x
                for s in suffix_sums:
                    current_sum += s
                    new_suffixes.append(current_sum)
                
                # Check collision
                if not used_sums.isdisjoint(new_suffixes):
                    continue
                
                # Also need to check if x itself is distinct?
                # x is in new_suffixes, so it's covered.
                
                # Accept
                seq.append(x)
                used_sums.update(new_suffixes)
                suffix_sums = new_suffixes
            
            return seq

        # Try a few variations
        seq1 = try_greedy(prefer_large=False)
        seq2 = try_greedy(prefer_large=True)
        
        best = seq1 if len(seq1) > len(seq2) else seq2
        
        # Try to improve best using a "Insertion Heuristic"
        # Try to insert any missing number into the best sequence at any position?
        # No, the sequence must be strictly increasing. So we can only insert x between a[i] and a[i+1].
        
        # Let's do a simple local search: try to add one more element to the best sequence found.
        # This is hard because adding one element changes all suffix sums from that point on.
        
        # Alternative: Randomized Restart Greedy
        # Shuffling the candidate order might find better solutions.
        
        for _ in range(100):
            if time.time() > time_limit:
                break
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            seq = try_greedy_with_order(candidates)
            if len(seq) > len(best):
                best = seq
                
        return best

    def try_greedy_with_order(candidates):
        seq = []
        used_sums = set()
        suffix_sums = []
        
        for x in candidates:
            new_suffixes = [x]
            current_sum = x
            for s in suffix_sums:
                current_sum += s
                new_suffixes.append(current_sum)
            
            if not used_sums.isdisjoint(new_suffixes):
                continue
            
            seq.append(x)
            used_sums.update(new_suffixes)
            suffix_sums = new_suffixes
        return seq

    # Initial solution
    current_best = solve_greedy_backtrack(time_limit=30.0)
    
    # Now, try to improve current_best using a "Local Search" that is more targeted.
    # The previous SA was slow because it checked O(n^2) sums.
    # Here, we can use the incremental check for neighbors.
    
    def get_suffix_sums(seq):
        suffixes = []
        s = 0
        for x in reversed(seq):
            s += x
            suffixes.append(s)
        return suffixes

    def check_append(seq, x):
        # Returns True if x can be appended, and the new suffix sums if so
        suffixes = get_suffix_sums(seq)
        new_suffixes = [x]
        s = x
        for ss in suffixes:
            s += ss
            new_suffixes.append(s)
        
        # We need to check if any new_suffix is in the set of ALL existing sums.
        # But we don't have the set of ALL sums, only suffixes?
        # No, the condition is "Sums of ALL nonempty consecutive blocks must be distinct".
        # This includes blocks that don't end at the last element.
        # So we MUST maintain the set of ALL sums.
        
        # Let's maintain used_sums set.
        pass

    # Let's refine the local search to use incremental checks properly.
    
    def local_search_improve(seq, time_limit=10.0):
        if not seq:
            return seq
        
        current_seq = list(seq)
        current_len = len(current_seq)
        
        # Precompute sums for current_seq
        def compute_all_sums(s):
            sums = set()
            n = len(s)
            for i in range(n):
                ss = 0
                for j in range(i, n):
                    ss += s[j]
                    sums.add(ss)
            return sums
        
        current_sums = compute_all_sums(current_seq)
        
        # We want to try to INSERT an element into the sequence.
        # Inserting at position k changes the sums of all blocks starting at or before k.
        # This is complex.
        
        # Simpler: Try to REPLACE an element with a larger/smaller one to free up space for another?
        # Or just try to add one more element at the end?
        # If we can't add at the end, we can't add anywhere else easily without breaking the "increasing" property
        # or causing massive sum collisions.
        
        # Actually, if greedy failed to add a value at the end, it might be because of collision.
        # Maybe a different combination of earlier values would allow a longer sequence?
        # The randomized greedy above handles this by shuffling.
        
        # Let's try a "Hill Climbing" on the sequence of VALUES.
        # Move: change one value to a nearby value (maintaining increasing order).
        # Check if valid. If valid and same length, accept (to explore).
        # If we find a valid sequence of same length, we can try to append more.
        
        T = 1.0
        while time.time() < time_limit:
            # Pick a random position to change
            i = random.randint(0, len(current_seq) - 1)
            
            # Determine valid range for new value
            if i == 0:
                low, high = 1, current_seq[1] - 1
            elif i == len(current_seq) - 1:
                low, high = current_seq[i-1] + 1, 256
            else:
                low, high = current_seq[i-1] + 1, current_seq[i+1] - 1
            
            if low > high:
                continue
                
            new_val = random.randint(low, high)
            if new_val == current_seq[i]:
                continue
                
            new_seq = list(current_seq)
            new_seq[i] = new_val
            
            # Validate
            new_sums = compute_all_sums(new_seq)
            if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                # Valid!
                # Try to extend
                # Compute suffix sums of new_seq
                suffixes = get_suffix_sums(new_seq)
                all_sums = new_sums # We already have this
                
                # Try to append any value
                extended = False
                for x in range(1, 257):
                    # Check if x can be appended
                    # New suffixes: x, x+s1, x+s2...
                    # Check against all_sums
                    # Optimization: x must not be in all_sums.
                    # x + s1 must not be in all_sums, etc.
                    
                    # Quick check: x not in all_sums
                    if x in all_sums:
                        continue
                    
                    # Check extended suffixes
                    ok = True
                    s = x
                    # The new sums are the existing sums PLUS the new suffixes.
                    # We need to check if the new suffixes collide with EXISTING sums.
                    # Note: new suffixes are distinct from each other because seq is increasing (positive).
                    # So we just need to check if any new_suffix is in all_sums.
                    
                    for ss in suffixes:
                        s += ss
                        if s in all_sums:
                            ok = False
                            break
                    if ok:
                        # Also check x itself (already done)
                        # Wait, the loop above checks x+s1, x+s2...
                        # We need to check x itself too.
                        # My loop:
                        # s = x
                        # for ss in suffixes: s += ss; check s
                        # This checks x+s1, x+s2...
                        # It does NOT check x.
                        # I checked x in all_sums before the loop.
                        # So this is correct.
                        
                        # If ok, we can extend!
                        current_seq.append(x)
                        current_len += 1
                        # Update state
                        current_sums = all_sums | {x, s} # Wait, s is the last one.
                        # Better to recompute or update set properly.
                        # Let's just update the set.
                        # New sums added: x, x+s1, ..., x+total
                        # We know they are distinct from each other.
                        # We know they are distinct from old sums (checked).
                        
                        # Let's recompute to be safe and simple, as n is small.
                        current_sums = compute_all_sums(current_seq)
                        extended = True
                        break
                        
                if not extended:
                    # If we couldn't extend, we might still accept the move to explore
                    # But we only care about maximizing length.
                    # If length is same, we accept with prob exp(-0/T) = 1?
                    # No, we want to find a state that CAN be extended.
                    # So we should accept moves that keep validity, hoping to find a "loose" configuration.
                    current_seq = new_seq
                    current_sums = new_sums
                    T *= 0.99
            else:
                T *= 0.99
                
        return current_seq

    # Run local search on the best greedy result
    improved = local_search_improve(current_best, time_limit=10.0)
    
    if len(improved) > len(current_best):
        current_best = improved
        
    return current_best

# Quick test
if __name__ == "__main__":
    import sys
    seq = construct()
    print(f"Length: {len(seq)}")
    print(seq)
