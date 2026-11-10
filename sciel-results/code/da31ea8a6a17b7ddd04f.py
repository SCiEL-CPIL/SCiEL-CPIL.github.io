import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Pre-compute pairs for fast collision checking?
    # Actually, for N=1000, sums are up to 2000.
    # We can use a boolean array `in_set` of size 1001.
    # And a count array `sum_counts` of size 2001.
    
    # Initialize with a dense random set?
    # The B2[2] density is approx 3/2 * sqrt(N) for Sidon, but for B2[2] it's higher.
    # Actually, for B2[2], the asymptotic constant is at least 3/2? No, that's for Sidon (B2[1]).
    # For B2[2], the bound is related to sqrt(N) as well, but with a larger constant.
    # Let's estimate: if |S| = k, number of sums is ~ k^2/2.
    # Each sum has <= 2 reps. Total reps ~ k^2/2.
    # Range of sums ~ 2N.
    # So k^2/2 <= 2 * 2N => k^2 <= 8N => k <= sqrt(8N) ~ 2.8 * sqrt(N).
    # For N=1000, sqrt(1000) ~ 31.6. 2.8 * 31.6 ~ 88.
    # Wait, the prompt says "asymptotic constant is at least 3/2".
    # This usually refers to the limit of |S| / sqrt(N).
    # If constant is 3/2, then |S| ~ 1.5 * sqrt(1000) ~ 47.
    # But "at least 3/2" suggests it could be higher.
    # Let's check known values.
    # For N=1000, a good B2[2] set might have size ~ 60-80?
    # Let's try to start with a greedy construction to get a decent initial set,
    # then use Simulated Annealing to improve it.
    
    # Helper functions using NumPy for speed
    # State: S (set of ints), sum_counts (array of length 2001)
    
    def compute_counts(S):
        counts = np.zeros(2001, dtype=np.int32)
        S_list = list(S)
        n = len(S_list)
        # Use numpy to compute all pair sums?
        # This is O(N^2) which is 1000^2 = 1e6. Fast enough in numpy?
        # Yes, 1e6 operations is < 1ms in numpy.
        # But doing this for every move in SA is too slow if we do it from scratch.
        # We need incremental updates.
        
        # Let's stick to incremental updates in Python but optimize the inner loop.
        # Or, since N=1000 is small, maybe we can afford to recompute counts
        # using a fast C-extension-like approach? No, pure Python is slow.
        
        # Alternative: Use a bitset?
        # Let's try to make the incremental update as fast as possible.
        
        # Actually, let's look at the "inspiration" again.
        # "leveraging NumPy vectorization to efficiently compute sum representations via convolution or boolean array operations"
        
        # If we represent S as a boolean vector v of length 1001.
        # The number of representations of sum s is the number of pairs (i,j) in S such that i+j=s.
        # This is the coefficient of x^s in (sum_{x in S} x^x)^2.
        # We can use FFT/Convolution to compute all sums at once.
        # Convolution of v with v.
        # len(v) = 1001. Convolution length ~ 2001.
        # np.convolve(v, v) is very fast.
        # Let's test if we can use this for the "evaluation" step in SA.
        # If we do a move (add/remove/swap), we update v.
        # Then we compute conv = np.convolve(v, v).
        # This gives the count of ordered pairs (i,j).
        # For i < j, count is conv[s] / 2.
        # For i = j, count is 1 if 2i=s and i in S.
        # We need unordered pairs a <= b.
        # Let R(s) be the number of unordered pairs.
        # R(s) = floor(conv[s] / 2) + (1 if s is even and s/2 in S else 0)
        # Wait, conv[s] = sum_{i+j=s} v[i]v[j].
        # If i != j, both (i,j) and (j,i) are counted. So 2 * (number of unordered pairs with i!=j).
        # If i == j, (i,i) is counted once.
        # So conv[s] = 2 * N_distinct + N_same.
        # Where N_distinct is # of pairs {i,j} with i<j, i+j=s.
        # And N_same is 1 if s=2i for some i in S, else 0.
        # The total representations a<=b is N_distinct + N_same.
        # Let T(s) = N_distinct + N_same.
        # We need T(s) <= 2 for all s.
        
        # So the condition is:
        # (conv[s] - N_same) / 2 + N_same <= 2
        # conv[s]/2 + N_same/2 <= 2
        # conv[s] + N_same <= 4
        
        # Let's verify.
        # If T(s) = 2.
        # Case 1: N_same = 0. conv[s] = 2 * 2 = 4. conv + N_same = 4. OK.
        # Case 2: N_same = 1. conv[s] = 2 * 1 + 1 = 3. conv + N_same = 4. OK.
        # Case 3: N_same = 0, T(s)=1. conv=2. conv+N_same=2 <= 4. OK.
        # Case 4: N_same = 1, T(s)=1. conv=1. conv+N_same=2 <= 4. OK.
        
        # So the validity check is:
        # max(conv + N_same_vec) <= 4
        # Where N_same_vec is a vector of 0s and 1s, 1 at s=2i if i in S.
        
        # This is extremely fast!
        # 1. Update v (boolean array)
        # 2. conv = np.convolve(v, v)
        # 3. N_same_vec: We need to construct this.
        #    N_same_vec[2*i] = 1 if v[i] else 0.
        #    This can be done by taking v[::2] (every other element) and placing them?
        #    Actually, N_same_vec is just v duplicated at even indices?
        #    Let's create an array of zeros length 2001.
        #    indices = np.arange(1, 1001) * 2
        #    N_same_vec[indices] = v[1:1001]  # since v[0] is index 0, not in 1..1000
        #    Wait, v is 0-indexed. v[i] corresponds to number i.
        #    So v[1:1001] corresponds to numbers 1..1000.
        #    Sums 2*i are at indices 2, 4, ..., 2000.
        #    So N_same_vec[2:2001:2] = v[1:1001].
        
        # 4. Check if np.max(conv + N_same_vec) <= 4.
        
    # Let's implement this SA approach.
    
    # Initial construction: Greedy
    v = np.zeros(1001, dtype=np.int8)
    S = set()
    
    # Greedy fill
    # We need to check validity after each addition.
    # Using np.convolve for every addition in greedy might be slow if we do 1000 additions.
    # 1000 * (convolve 1001x1001) ~ 1000 * 1ms = 1s. Acceptable.
    
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline - 10: break
        # Try adding x
        v[x] = 1
        conv = np.convolve(v, v)
        # Construct N_same_vec
        n_same = np.zeros(2001, dtype=np.int32)
        n_same[2:2001:2] = v[1:1001]
        if np.max(conv + n_same) <= 4:
            S.add(x)
        else:
            v[x] = 0
            
    # Now we have a valid set S.
    # Size is likely around 40-60.
    
    # Simulated Annealing
    # Objective: Maximize size.
    # Moves:
    # 1. Add a random x not in S. If valid, accept. If invalid, reject.
    # 2. Remove a random x in S. (Always valid, but reduces size. Accept with prob exp(-1/T) or similar? 
    #    Actually, we want to maximize size. So removing is bad. 
    #    We should only remove if it allows adding MORE elements later.
    #    This is a "swap" or "kick".)
    # 3. Swap: Remove x, Add y. If valid, accept.
    
    # Let's use a simple restart-based local search with random kicks.
    # Since N is small, we can afford to be a bit more aggressive.
    
    best_size = len(S)
    best_v = v.copy()
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    
    # We will run SA for the remaining time.
    # To speed up, we can pre-compute the "cost" of adding x?
    # No, the cost depends on the current set.
    
    # Optimization:
    # Instead of full convolve every time, can we update incrementally?
    # Convolution is linear.
    # If we add x, v_new = v + e_x.
    # conv_new = conv + 2 * v * e_x + e_x * e_x
    # v * e_x is just v shifted by x? No, v is the vector.
    # The contribution to conv[s] from adding x is:
    # 1 for s=2x.
    # 1 for s=x+y for all y in S.
    # So we can update conv in O(N) instead of O(N log N) or O(N^2).
    # O(N) is 1000 operations. Very fast in Python loop?
    # 1000 ops * 10000 moves = 10^7 ops. ~1s in Python.
    # So incremental update is better for SA.
    
    # Let's switch to incremental update for the SA phase.
    
    # Current state: v, S, conv (numpy array), n_same (numpy array)
    # We need to maintain conv and n_same.
    
    conv = np.convolve(v, v)
    n_same = np.zeros(2001, dtype=np.int32)
    n_same[2:2001:2] = v[1:1001]
    
    # Helper to update conv and n_same when adding/removing x
    def add_x(x):
        nonlocal conv, n_same
        # Update conv
        # conv[2x] += 1
        conv[2*x] += 1
        # conv[x+y] += 1 for all y in S
        # We need to iterate y in S
        # To make this fast, we can keep S as a list?
        # Or use numpy: indices = S_list. conv[x + S_list] += 1
        # S_list is a numpy array of elements in S.
        pass
        
    # Let's maintain S_list as a numpy array for vectorized updates.
    # But S changes.
    # Maybe just use a Python set for membership, and a list for iteration.
    # Iteration over 50-100 elements is fast.
    
    S_list = list(S)
    
    def do_add(x):
        nonlocal conv, n_same, S_list
        S.add(x)
        S_list.append(x)
        v[x] = 1
        
        # Update conv
        conv[2*x] += 1
        # For all y in S_list (including x? No, x is just added. Pairs are (x,y) for y in S_old)
        # Wait, the pair (x,x) is handled by conv[2x].
        # Pairs (x,y) for y in S_old.
        # So we iterate S_list before adding x?
        # Let's snapshot S_list before adding.
        # Actually, I appended x to S_list.
        # So S_list[-1] is x.
        # We need to update conv[x+y] for y in S_list[:-1].
        # Also, what about y=x? That's conv[2x], already done.
        
        # Let's be careful.
        # Old S. New S = Old S U {x}.
        # New pairs: (x,x) and (x,y) for y in Old S.
        # conv[2x] += 1.
        # conv[x+y] += 1 for y in Old S.
        
        # So:
        # conv[2*x] += 1
        # for y in S_list[:-1]: conv[x+y] += 1
        
        # Update n_same
        # n_same[2x] becomes 1.
        n_same[2*x] = 1
        
    def do_remove(x):
        nonlocal conv, n_same, S_list
        S.discard(x)
        S_list.remove(x)
        v[x] = 0
        
        # Remove pairs (x,x) and (x,y) for y in New S (which is Old S - {x})
        # conv[2x] -= 1
        # conv[x+y] -= 1 for y in S_list (which is now New S)
        
        conv[2*x] -= 1
        for y in S_list:
            conv[x+y] -= 1
            
        n_same[2*x] = 0
        
    # Re-initialize conv and n_same correctly based on current S
    conv = np.zeros(2001, dtype=np.int32)
    n_same = np.zeros(2001, dtype=np.int32)
    for x in S:
        # This is O(N^2) setup, but N is small.
        pass
        
    # Let's just recompute from scratch using the fast method to be safe.
    conv = np.convolve(v, v)
    n_same = np.zeros(2001, dtype=np.int32)
    n_same[2:2001:2] = v[1:1001]
    
    # Now run SA
    # We want to maximize size.
    # Move: Pick a random x not in S. Try to add.
    # If valid, accept.
    # If invalid, maybe try to remove a "blocking" element?
    
    # Better Move: Swap.
    # Pick x in S, y not in S.
    # Remove x, Add y.
    # If valid, accept.
    
    # Let's try a simple hill climbing first:
    # Try to add any element. If stuck, remove a random element and try again.
    
    current_size = len(S)
    
    while time.time() < deadline:
        # Try to add
        improved = False
        # Try a few random candidates
        for _ in range(50):
            x = random.randint(1, 1000)
            if x not in S:
                # Check if can add
                # We need to check if adding x violates constraint.
                # New conv = conv + delta
                # We can simulate the delta.
                # delta_conv[2x] = 1
                # delta_conv[x+y] = 1 for y in S
                # Check if max(conv + delta_conv + delta_n_same) <= 4
                # delta_n_same[2x] = 1 (if it was 0)
                
                # To avoid modifying global state, we can just check the specific sums.
                # The sums that change are 2x and x+y for y in S.
                # We only need to check these sums.
                # Let's collect the indices to check.
                indices_to_check = [2*x]
                for y in S:
                    indices_to_check.append(x+y)
                
                # Also, n_same[2x] changes from 0 to 1.
                # So for s=2x, the new value is conv[2x] + 1 + 1 = conv[2x] + 2.
                # For s=x+y (y!=x), new value is conv[s] + 1 + n_same[s].
                
                valid = True
                for s in indices_to_check:
                    if s == 2*x:
                        new_val = conv[s] + 2
                    else:
                        new_val = conv[s] + 1 + n_same[s]
                    if new_val > 4:
                        valid = False
                        break
                
                if valid:
                    do_add(x)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_v = v.copy()
                    improved = True
                    break
        
        if not improved:
            # Kick: remove a random element
            if S:
                x_to_remove = random.choice(list(S))
                do_remove(x_to_remove)
                current_size -= 1
                
                # Try to add something else
                for _ in range(50):
                    x = random.randint(1, 1000)
                    if x not in S:
                        # Check validity
                        indices_to_check = [2*x]
                        for y in S:
                            indices_to_check.append(x+y)
                        
                        valid = True
                        for s in indices_to_check:
                            if s == 2*x:
                                new_val = conv[s] + 2
                            else:
                                new_val = conv[s] + 1 + n_same[s]
                            if new_val > 4:
                                valid = False
                                break
                        
                        if valid:
                            do_add(x)
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_v = v.copy()
                            break
    
    # Restore best set
    # We need to return the list of integers.
    # We tracked best_v, but we modified S and conv.
    # We should reconstruct S from best_v.
    best_S = [i for i in range(1, 1001) if best_v[i] == 1]
    
    return best_S

if __name__ == "__main__":
    res = construct()
    print(len(res))
