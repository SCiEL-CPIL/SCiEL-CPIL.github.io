import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    n = 4096
    max_sum = 3 * n
    
    def fast_greedy(order=None):
        if order is None:
            order = range(1, n + 1)
        
        selected = []
        sums_mask = np.zeros(max_sum + 1, dtype=bool)
        
        for x in order:
            if time.time() > deadline:
                break
            
            # 1. Check 3x
            if sums_mask[3 * x]:
                continue
            
            # 2. Check 2x + s
            # If selected is empty, skip
            if selected:
                sel_arr = np.array(selected)
                new_sums_2 = 2 * x + sel_arr
                if np.any(sums_mask[new_sums_2]):
                    continue
                
            # 3. Check x + s + t
            if len(selected) >= 2:
                sel_arr = np.array(selected)
                # Compute all pair sums
                # To save memory, we can do it in chunks or just let numpy handle it
                # K is small (~50), so KxK matrix is small
                pair_sums = sel_arr[:, None] + sel_arr[None, :]
                new_sums_3 = x + pair_sums
                if np.any(sums_mask[new_sums_3]):
                    continue
            
            # Add x
            selected.append(x)
            sums_mask[3 * x] = True
            
            if selected[:-1]: # If we added x, selected[:-1] is the old set
                old_sel = np.array(selected[:-1])
                sums_mask[2 * x + old_sel] = True
                
            if len(selected) >= 2:
                # Update sums for x + s + t where s,t < x
                # We need to add sums involving x and two previous elements
                # And sums involving x, x, s? No, 2x+s was handled.
                # We need x + s + t for s, t in old_sel (s <= t)
                # And x + x + s? No, that's 2x+s.
                # We need x + s + t where s, t are from the set BEFORE adding x.
                # Also x + x + x? 3x.
                
                # Wait, the check `new_sums_3` used `selected` BEFORE adding x?
                # In my loop:
                # `selected` contains elements added so far.
                # When checking x, `selected` does NOT contain x.
                # So `sel_arr` is the set of existing elements.
                # `pair_sums` are sums of pairs from existing elements.
                # `new_sums_3` are $x + s + t$.
                # This covers all triples containing x and two existing elements.
                
                # Now update mask.
                # 1. 3x (done)
                # 2. 2x + s (done)
                # 3. x + s + t for s, t in existing set.
                if len(selected) >= 2: # Selected now includes x. Old set size was len(selected)-1.
                    old_sel = np.array(selected[:-1])
                    # We need to add sums x + s + t where s, t are from old_sel.
                    # Note: s and t can be the same? Yes, "allowing repetitions".
                    # But in the set `selected`, elements are distinct.
                    # The triple is (x, s, t). s and t are chosen from the set.
                    # If s=t, it's (x, s, s).
                    # My `pair_sums` calculation `sel_arr[:, None] + sel_arr[None, :]` includes (s, s) and (t, t) and (s, t).
                    # So `new_sums_3` includes all $x+s+t$.
                    
                    pair_sums_update = old_sel[:, None] + old_sel[None, :]
                    sums_mask[x + pair_sums_update] = True
        
        return selected

    best_set = []
    
    # 1. Standard Forward
    s1 = fast_greedy()
    if len(s1) > len(best_set):
        best_set = s1
        
    # 2. Backward
    if time.time() < deadline:
        s2 = fast_greedy(order=range(n, 0, -1))
        if len(s2) > len(best_set):
            best_set = s2
            
    # 3. Randomized
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            # Random shuffle of a subset? Or random start?
            # Try skipping some small numbers
            order = list(range(1, n + 1))
            np.random.shuffle(order)
            # But greedy on random order is bad.
            # Better: Randomly select a "bias"
            # Or just run greedy but with a random "noise" in the check?
            # Simplest: Run greedy on a random permutation of the first 1000 numbers, then rest?
            
            # Let's try a "Random Start" greedy:
            # Pick a random number, add it, then greedily add neighbors?
            # Too complex.
            
            # Just try a few random shuffles of the full range?
            # 4096 log 4096 is fast.
            s_rand = fast_greedy(order=order)
            if len(s_rand) > len(best_set):
                best_set = s_rand

    return best_set
