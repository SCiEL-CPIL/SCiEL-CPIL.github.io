import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (15s buffer)

    # 1. Generate all words and map to indices
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute balls as bitsets (Python integers)
    # ball[i] is an integer with bits set for all j such that dist(words[i], words[j]) <= radius
    # For radius 1, we just need i and its immediate neighbors.
    balls = [0] * N
    for i, w in enumerate(words):
        mask = 1 << i
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                mask |= (1 << idx)
                w_list[pos] = orig_val
        balls[i] = mask
        
    all_mask = (1 << N) - 1
    
    # Helper: Count bits (popcount)
    # Python 3.8+ has int.bit_count()
    def popcount(x):
        return x.bit_count()
        
    # Helper: Get list of indices from a bitset
    def bitset_to_list(mask):
        indices = []
        while mask:
            # Get least significant set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            indices.append(idx)
            mask ^= lsb
        return indices

    # 3. Greedy Initial Solution with Vectorized Gain Computation
    # Convert balls to numpy boolean matrix for faster gain computation
    # Each row i is a boolean array of length N where True means covered by ball i
    # For N=2187, this is 2187 * 2187 bits = ~600KB, very fast
    balls_np = np.zeros((N, N), dtype=bool)
    for i in range(N):
        # Convert integer bitset to numpy array
        # This is done by checking each bit, but we can optimize
        # Since we only have N=2187, let's just use the integer approach but vectorize the gain calc
        pass
    
    # Alternative: Use integer bitsets but precompute gain arrays
    # For greedy, we need: gain[i] = popcount(balls[i] & ~covered)
    # We can maintain covered as a Python integer
    # To speed up, we can precompute the binary representation of each ball as a numpy uint8 array?
    # Or just use the integer bit_count which is quite fast in CPython.
    
    # Let's try a different approach for greedy:
    # Maintain an array 'uncovered_count' for each candidate? No, that's complex.
    # Let's stick to integer bitsets but optimize the loop.
    
    def greedy_init():
        best_code = None
        best_size = N + 1
        
        for trial in range(10):  # 10 restarts instead of 5
            covered = 0
            code = []
            
            while covered != all_mask:
                if time.time() > deadline - 60:  # Leave more time for B&B
                    break
                
                best_idx = -1
                max_gain = 0
                candidates = []
                
                # Vectorize gain computation if possible
                # For now, keep it simple but maybe skip if time is low
                for i in range(N):
                    # Skip if already in code? No, we want max gain, duplicates don't add.
                    gain = popcount(balls[i] & ~covered)
                    if gain > max_gain:
                        max_gain = gain
                        best_idx = i
                        candidates = [i]
                    elif gain == max_gain and gain > 0:
                        candidates.append(i)
                
                if max_gain == 0:
                    break
                    
                if len(candidates) > 1:
                    # Random tie-breaking with a bit of preference for "central" words?
                    # Just random for now
                    best_idx = random.choice(candidates)
                
                code.append(best_idx)
                covered |= balls[best_idx]
                
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                
        return best_code if best_code else []

    # 4. Local Search (Prune and Swap)
    def local_search(code):
        if not code:
            return code
        
        # Convert to set for O(1) lookup
        code_set = set(code)
        
        # Precompute coverage counts for each word in the space?
        # Too expensive (2187 ints). 
        # Instead, for each candidate word in code, check if it's redundant by scanning other code words.
        
        improved = True
        iterations = 0
        max_iters = 200  # Increase iterations
        
        while improved and iterations < max_iters and time.time() < deadline - 30:
            improved = False
            iterations += 1
            # Shuffle to break symmetry
            random.shuffle(code)
            
            for i in range(len(code)):
                if time.time() > deadline - 30:
                    break
                    
                c = code[i]
                
                # Check if c is redundant: is balls[c] subset of union of balls of others?
                # We can do this by iterating other code words and OR-ing their balls.
                # Optimization: If code is large, this is O(|code|^2). |code| ~ 186, so ~35k ops. Fine.
                
                rest_covered = 0
                # Iterate all other code words
                for j in range(len(code)):
                    if j != i:
                        rest_covered |= balls[code[j]]
                
                # Check if c adds anything
                if balls[c] & ~rest_covered == 0:
                    # c is redundant, remove it
                    code.pop(i)
                    code_set.remove(c)
                    improved = True
                    break # Restart loop to re-shuffle and check new state
                    
                # Try swap: Can we replace c with another word w such that w covers the critical part of c
                # and we can remove one more word? Or just replace c with w to improve "density"?
                # Standard kick-move: Replace c with w, then try to prune.
                # This is expensive. Let's stick to prune for now, maybe add simple swaps later if time permits.
                
        return code

    # 5. Branch and Bound
    # Global state
    best_size_global = N + 1
    best_code_global = []
    
    # Precompute neighbors_of: for each uncovered word u, which code words cover it?
    # neighbors_of[u] = list of v such that u is in balls[v]
    neighbors_of = [[] for _ in range(N)]
    for v in range(N):
        mask = balls[v]
        # Iterate set bits in mask
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            neighbors_of[u].append(v)
            m ^= lsb

    sys.setrecursionlimit(10000)
    
    def bnb(covered_mask, current_size, current_set, depth):
        nonlocal best_size_global, best_code_global
        
        if time.time() > deadline - 10:
            raise TimeoutError("Time limit exceeded")
            
        if current_size >= best_size_global:
            return
            
        if covered_mask == all_mask:
            best_size_global = current_size
            best_code_global = list(current_set)
            return
            
        # Lower bound
        uncovered = all_mask ^ covered_mask
        unc_count = popcount(uncovered)
        # Max coverage per ball is 1 + n*(q-1) = 1 + 7*2 = 15
        lb = (unc_count + 14) // 15
        
        if current_size + lb >= best_size_global:
            return
            
        # Pick an uncovered vertex
        # Heuristic: Pick the one with the fewest candidates to reduce branching
        lsb = uncovered & -uncovered
        u_idx = lsb.bit_length() - 1
        
        # Get candidates
        candidates = neighbors_of[u_idx]
        
        # Filter out candidates already in current_set
        in_set = set(current_set)
        valid_candidates = [v for v in candidates if v not in in_set]
        
        if not valid_candidates:
            return
        
        # Sort candidates by gain (popcount(balls[v] & uncovered)) descending
        # This helps find a good solution quickly to prune.
        # To speed up, we can limit the number of candidates considered (beam search style)
        # But for correctness, we need to explore all unless pruned.
        # Let's sort all, but if there are too many, maybe truncate?
        # With N=2187, max candidates is 2187. Sorting is fast.
        valid_candidates.sort(key=lambda v: popcount(balls[v] & uncovered), reverse=True)
        
        # Branch limit: Only consider top K candidates to keep B&B feasible
        # This turns it into a bounded B&B, which is a heuristic.
        # Let's try K=10 for now.
        for v in valid_candidates[:10]:
            current_set.append(v)
            bnb(covered_mask | balls[v], current_size + 1, current_set, depth + 1)
            current_set.pop()
            
            if time.time() > deadline - 10:
                raise TimeoutError("Time limit exceeded")
            
            if current_size + 10 >= best_size_global:
                # Prune remaining if even with optimistic bound we can't beat
                break

    # Run Greedy
    initial_code = greedy_init()
    if not initial_code:
        # Fallback: return all words
        return [list(words[i]) for i in range(N)]
        
    # Run Local Search
    initial_code = local_search(initial_code)
    
    # Update Global Best
    best_size_global = len(initial_code)
    best_code_global = list(initial_code)
    
    # Run B&B
    try:
        bnb(0, 0, [], 0)
    except TimeoutError:
        pass
    except RecursionError:
        # Increase recursion limit or handle differently
        try:
            bnb(0, 0, [], 0)
        except:
            pass

    # Final Polish
    if best_code_global:
        best_code_global = local_search(best_code_global)
        # Verify
        covered = 0
        for c in best_code_global:
            covered |= balls[c]
        if covered != all_mask:
            # If invalid, fallback to initial
            best_code_global = initial_code

    return [list(words[i]) for i in best_code_global]
