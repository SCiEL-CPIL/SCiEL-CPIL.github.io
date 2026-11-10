import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin (evaluator kills at 90s)
    
    n = 128
    # targets: 128..256 inclusive
    # A subset of [0, 128]
    # We want to minimize the max multiplicity of sums in [128, 256]
    # Multiplicity is the number of unordered pairs (a, b) with a <= b, a, b in A, such that a+b = t.
    
    # Pre-compute pairs for each target sum t in [128, 256]
    # pairs[t] is a list of (a, b) with a <= b, a+b=t, 0<=a,b<=128
    # We only care about a, b in [0, 128]
    
    # Actually, we can compute the counts on the fly or precompute a matrix.
    # Let's use a bitset or array representation for A.
    # A is a set of integers.
    
    # Initial A:
    # We need A to cover [128, 256].
    # A simple valid A is [0, 128]. Sums are 0..256.
    # But multiplicity might be high.
    # For A = [0, 128], count of sum t is:
    # if t <= 128: t+1
    # if t > 128: 257-t
    # Max is 129 at t=128.
    
    # Let's try to find a better A.
    # Heuristic: Start with a structured set.
    # e.g., A = {0, 2, 4, ..., 128} (evens).
    # Sums are even numbers 0..256.
    # This does NOT cover odd numbers in [128, 256]. So invalid.
    
    # We need to cover all integers.
    # Maybe A = [0, 128] is a good starting point for local search.
    
    def compute_counts(A):
        """
        A is a set of integers.
        Returns array of counts for sums 0..256.
        """
        counts = np.zeros(257, dtype=int)
        A_list = sorted(A)
        # Optimization: use vectorized addition?
        # For i in range(len(A_list)):
        #    for j in range(i, len(A_list)):
        #        counts[A_list[i] + A_list[j]] += 1
        # This is O(|A|^2). If |A| is ~64, that's 4096 ops. Very fast.
        # If |A| is 129, that's ~16k ops. Still fast.
        
        # Let's just do the double loop in Python for clarity and correctness, 
        # unless it's too slow. 
        # 16k ops per evaluation. If we do 1000 evaluations, 16M ops. 
        # Python might be slow (1s). 
        # We can optimize by using numpy broadcasting if A is small enough?
        # Or just stick to Python sets and loops but be careful.
        
        # Let's try a faster way:
        # Convert A to numpy array.
        # a = np.array(list(A))
        # This doesn't directly help with the double loop easily without creating a matrix of sums.
        # Sums matrix is |A|x|A|.
        # If |A|=129, matrix is 129x129 = 16k elements.
        # We can compute the upper triangle.
        
        a = np.array(list(A))
        # Create meshgrid
        # i, j = np.triu_indices(len(a))
        # sums = a[i] + a[j]
        # np.add.at(counts, sums, 1)
        
        i, j = np.triu_indices(len(a))
        sums = a[i] + a[j]
        # Filter sums <= 256 (though a,b <= 128 implies sum <= 256)
        # np.add.at is unbuffered, so it handles duplicates correctly.
        np.add.at(counts, sums, 1)
        return counts

    def get_peak(counts):
        # Only care about 128..256
        return np.max(counts[128:257])

    def is_valid(counts):
        return np.all(counts[128:257] > 0)

    # Initial A
    # Let's start with A = set(range(129))
    A = set(range(129))
    counts = compute_counts(A)
    curr_peak = get_peak(counts)
    best_peak = curr_peak
    best_A = set(A)
    
    # Local Search
    # Moves:
    # 1. Remove x from A (if valid)
    # 2. Add y to A (if y not in A) - usually increases peak, but might allow removals later?
    #    Actually, we want to minimize peak. Adding elements increases counts.
    #    So we should focus on REMOVING elements.
    #    But we must maintain validity (coverage).
    #    So: Try to remove x. If still valid, accept.
    #    If no removals possible, we are stuck?
    #    Maybe we need to swap? Remove x, add y?
    #    Or remove x1, x2, add y1, y2?
    
    # Let's implement:
    # 1. Greedy Removal:
    #    While there exists x in A such that A\{x} is valid:
    #       Remove x that minimizes the new peak (or just any valid removal).
    #       Actually, removing x decreases counts. Peak might stay same or decrease.
    #       We should pick the x that results in the lowest peak.
    
    # 2. If stuck (no single removal valid), try swaps?
    #    Or random restarts?
    
    # Let's try Greedy Removal first.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try all removals
        # To speed up, we can precompute which x are "critical" (removal breaks coverage).
        # But checking validity requires computing counts or updating them.
        
        # Optimization:
        # Instead of recomputing full counts for each x,
        # we can update counts incrementally.
        
        # Let's maintain counts array.
        # When removing x:
        # For each y in A (y != x): counts[x+y] -= 1
        # counts[2*x] -= 1
        
        A_list = list(A)
        random.shuffle(A_list)
        
        best_x = None
        best_new_peak = curr_peak
        best_new_counts = None
        
        for x in A_list:
            # Simulate removal
            new_counts = counts.copy()
            
            # Decrement pairs involving x
            for y in A:
                if y == x:
                    continue
                s = x + y
                if s <= 256:
                    new_counts[s] -= 1
            
            s = 2 * x
            if s <= 256:
                new_counts[s] -= 1
                
            if is_valid(new_counts):
                peak = get_peak(new_counts)
                if peak < best_new_peak:
                    best_new_peak = peak
                    best_x = x
                    best_new_counts = new_counts
                    # If peak is 1, we can't do better (since valid means >=1)
                    # Actually min peak is 1.
                    if best_new_peak == 1:
                        break
        
        if best_x is not None:
            A.remove(best_x)
            counts = best_new_counts
            curr_peak = best_new_peak
            improved = True
            if curr_peak < best_peak:
                best_peak = curr_peak
                best_A = set(A)
                # If best_peak is 1, we are done? 
                # Wait, is it possible to have peak 1?
                # If every sum in [128,256] has exactly 1 representation.
                # That would be a perfect difference set or similar.
                # Unlikely for this size, but possible?
                # If we reach 1, we can break.
                if best_peak == 1:
                    break
            continue # Restart loop with new A
        
        # If no removal improved, try swaps?
        # Or random perturbation?
        # Let's try a random swap: remove x, add y (y not in A).
        # This keeps |A| same.
        # It might reduce peak if x was contributing to high counts and y contributes to low counts.
        
        # We need to be careful with time.
        # If we are stuck in local minimum, we need to escape.
        
        # Let's try limited random swaps.
        if len(A) > 0:
            # Pick random x in A
            x = random.choice(list(A))
            # Pick random y not in A
            candidates = [i for i in range(129) if i not in A]
            if candidates:
                y = random.choice(candidates)
                
                new_counts = counts.copy()
                # Remove x
                for z in A:
                    if z == x: continue
                    s = x + z
                    if s <= 256: new_counts[s] -= 1
                s = 2*x
                if s <= 256: new_counts[s] -= 1
                
                # Add y
                # Note: A is modified? No, we are simulating.
                # New set is (A - {x}) U {y}
                # Pairs involving y: (y, z) for z in (A - {x}) U {y}
                # So z can be any in A except x, plus y itself.
                
                for z in A:
                    if z == x: continue
                    s = y + z
                    if s <= 256: new_counts[s] += 1
                s = 2*y
                if s <= 256: new_counts[s] += 1
                
                if is_valid(new_counts):
                    peak = get_peak(new_counts)
                    # Accept if better or equal (to allow wandering)
                    # For now, accept if better.
                    if peak < curr_peak:
                        A.remove(x)
                        A.add(y)
                        counts = new_counts
                        curr_peak = peak
                        if curr_peak < best_peak:
                            best_peak = curr_peak
                            best_A = set(A)
                            if best_peak == 1:
                                break
                        improved = True
                        continue # Restart
        
        # If still stuck, try 2-swap?
        # Or just restart with a different initial A?
        # Given time limit, maybe we should just do one pass of greedy removal.
        # The greedy removal is likely to find a good solution quickly.
        
        # If we didn't improve, we might be in a local minimum.
        # Let's try a more aggressive move: remove 2, add 0? No, size must change?
        # Actually, removing elements reduces size.
        # If we can't remove any single element, we are at a minimal valid set (w.r.t removals).
        # Further optimization requires swaps.
        
        # Since swaps are expensive to search exhaustively, and we have a time limit,
        # let's just break if no improvement in a while?
        # Or keep trying random swaps for the remaining time.
        
        # For now, if no improvement, we continue the loop but it will just keep checking removals (which will fail) and random swaps.
        # To avoid infinite loop doing nothing, we should track stagnation.
        
        # Let's add a stagnation counter.
        # If no improvement for 100 iterations, break?
        # Or just let it run until time limit, doing random swaps.
        
        # The current code will loop until time limit.
        # Inside, if no removal, it tries ONE random swap.
        # This is inefficient. It should try more swaps or break.
        
        # Let's modify: if no removal, try a batch of random swaps.
        
        # Actually, the `continue` statements restart the while loop.
        # If no improvement, it falls through to the end of the while block.
        # Then the while condition is checked.
        # It will repeat.
        
        # To make it more effective, let's try multiple random swaps in one iteration if stuck.
        
        # But for simplicity and safety, let's just return the best found so far if we are stuck and time is running out.
        
        # The current logic:
        # 1. Try all removals. If one works, do it and continue.
        # 2. If none work, try 1 random swap. If it works (improves peak), do it and continue.
        # 3. If swap doesn't improve, do nothing and loop again.
        
        # This is a bit weak. Let's enhance step 3.
        # If stuck, try a random 2-element removal? (Only if valid).
        # Or just accept that we found a local minimum and stop?
        
        # Given the problem size, greedy removal is the main driver.
        # Let's stick with that.
        
    return sorted(list(best_A))
