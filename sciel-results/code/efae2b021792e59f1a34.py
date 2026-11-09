import sys
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0

    # We are looking for a Golomb Ruler with max gap <= 256.
    # Marks: x_0, x_1, ..., x_N
    # Gaps: a_i = x_i - x_{i-1} <= 256
    # All pairwise differences x_j - x_i (j > i) must be distinct.
    
    # We will use backtracking.
    # To make it fast, we use a set of used differences.
    # Max possible difference: If we have N marks, max sum is roughly N * 256.
    # For N=20, sum ~ 5000. Set is fine.
    
    def search(max_len, time_limit):
        marks = [0]
        used_diffs = set()
        
        def backtrack(idx, last_mark):
            if time.time() > time_limit:
                return False
            
            if idx == max_len:
                return True
            
            # Heuristic: Try gaps from 1 to 256
            # Why 1 to 256? Because we want to pack densely.
            # However, we must ensure that the new differences don't conflict.
            
            # Pruning: 
            # If we have idx marks, we have idx*(idx-1)/2 differences used.
            # We need to add (max_len - idx) more marks.
            # This will add (max_len - idx) * (max_len + 1) / 2 - idx*(idx-1)/2 new differences?
            # No, just check time and basic feasibility.
            
            for gap in range(1, 257):
                new_mark = last_mark + gap
                
                # Check conflicts
                conflict = False
                new_diffs = []
                for i in range(idx):
                    d = new_mark - marks[i]
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                # Accept
                marks.append(new_mark)
                used_diffs.update(new_diffs)
                
                if backtrack(idx + 1, new_mark):
                    return True
                
                # Backtrack
                marks.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False

        # We want to maximize length.
        # Start from a reasonable guess.
        # Greedy with small gaps might fail early.
        # Let's try to find length 15, 16, 17...
        
        best_marks = None
        
        # Estimate max possible length.
        # Number of differences is N(N-1)/2.
        # Max difference is bounded by total length L.
        # L <= 256 * N.
        # So N(N-1)/2 <= 256*N => N-1 <= 512 => N <= 513.
        # This is a loose bound. The "density" of differences matters.
        # In practice, Golomb rulers are sparse.
        # Optimal Golomb ruler of order N has length ~ N^2.
        # Here length L <= 256 N.
        # So N^2 <= 256 N => N <= 256.
        # So theoretically N could be up to 256.
        # But we need to check if such a ruler exists with max gap 256.
        
        # Let's try to extend a known good sequence.
        # Start with greedy (small gaps first) to get a baseline.
        # The previous code's greedy was powers of 2 (large gaps).
        # Here we try small gaps.
        
        # Try lengths from high to low? Or low to high?
        # Low to high is safer to find *a* solution, then extend.
        # But we want the MAX.
        # Let's try to find a sequence of length L.
        
        # We can use randomization if deterministic fails?
        # For now, deterministic "small gap first" is a strong heuristic for packing.
        
        # Let's try up to 25.
        for target_len in range(15, 30):
            if time.time() > time_limit - 2:
                break
            
            # Reset state for new search
            marks = [0]
            used_diffs = set()
            
            success = False
            def bt(idx, last_mark):
                if time.time() > time_limit - 2:
                    return False
                if idx == target_len:
                    return True
                # Try gaps 1..256
                for gap in range(1, 257):
                    new_mark = last_mark + gap
                    conflict = False
                    new_diffs = []
                    for i in range(idx):
                        d = new_mark - marks[i]
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.append(d)
                    if conflict:
                        continue
                    marks.append(new_mark)
                    used_diffs.update(new_diffs)
                    if bt(idx + 1, new_mark):
                        return True
                    marks.pop()
                    used_diffs.difference_update(new_diffs)
                return False
            
            if bt(1, 0):
                best_marks = marks[:]
                # If we found length L, we can try L+1?
                # The loop goes 15..30, so it will try higher.
                # But we should keep the best found so far in case higher fails.
            else:
                # If we failed for L, do we stop?
                # Not necessarily, but it's likely.
                # However, with "small gap first", it might get stuck in a local minimum.
                # We can break if we fail for a few consecutive lengths.
                pass

        if best_marks:
            # Convert marks to gaps (the required output)
            gaps = [best_marks[i] - best_marks[i-1] for i in range(1, len(best_marks))]
            return gaps
        else:
            # Fallback to a simple valid sequence
            # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256
            return [1, 2, 4, 8, 16, 32, 64, 128, 256]

    return search(0, TIME_LIMIT)
