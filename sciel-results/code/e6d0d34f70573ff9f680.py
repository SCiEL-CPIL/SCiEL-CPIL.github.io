import time
import random
import sys
import itertools

def is_valid(word, n):
    """Check if word contains all permutations of 1..n."""
    if len(word) < n:
        return False
    base = n + 1
    base_powers = [base**i for i in range(n)]
    target_set = set()
    # Generate all perms
    for p in itertools.permutations(range(1, n+1)):
        val = 0
        for i in range(n):
            val += p[i] * base_powers[i]
        target_set.add(val)
        
    seen = set()
    for i in range(len(word) - n + 1):
        val = 0
        for j in range(n):
            val += word[i+j] * base_powers[j]
        seen.add(val)
    return seen == target_set

def fallback_construct(n):
    """Guaranteed valid superpermutation (concatenation)."""
    perms = list(itertools.permutations(range(1, n+1)))
    result = list(perms[0])
    for p in perms[1:]:
        result.extend(p)
    return result

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Precompute all permutations as encoded integers
    # Encoding: base-(n+1) number where digit i is the (i+1)-th symbol
    base = n + 1
    base_powers = [base**i for i in range(n)]
    
    perms = []
    # Generate all permutations of 1..n
    def permute(arr, k):
        if k == 1:
            # Encode the permutation
            val = 0
            for i in range(n):
                val += arr[i] * base_powers[i]
            perms.append(val)
            return
        for i in range(k, 0, -1):
            arr[k-1], arr[i-1] = arr[i-1], arr[k-1]
            permute(arr, k-1)
            arr[k-1], arr[i-1] = arr[i-1], arr[k-1]
            
    initial_perm = list(range(1, n+1))
    permute(initial_perm, n)
    
    target_count = len(perms)
    target_set = set(perms)

    # 2. Greedy Construction Function
    def greedy_construct(lookahead_depth, random_seed):
        random.seed(random_seed)
        
        # Start with the first permutation
        # We need to be careful: the first window is the initial permutation.
        # We start with the string representing the first permutation.
        # Actually, it's better to start with an empty string and build, 
        # but the first n symbols must form a permutation.
        # Let's start with a random permutation.
        
        # Pick a random starting permutation
        start_p = random.choice(perms)
        
        # Decode start_p to list
        start_list = []
        temp = start_p
        for i in range(n):
            start_list.append(temp % base)
            temp //= base
        start_list.reverse()
        
        S = start_list[:]
        seen = {start_p}
        
        # If n=1, return
        if n == 1:
            return S
            
        # Greedy loop
        # We want to append symbols to S such that we cover all permutations.
        # At each step, we look at the last n-1 symbols of S.
        # We try appending each symbol c in 1..n.
        # The new window will be (last n-1 of S) + c.
        # If this window is a permutation, we check if it's new.
        # To use lookahead, we simulate appending c, then for the next step,
        # we check the best possible move from there.
        
        while len(seen) < target_count:
            last_n_minus_1 = S[-(n-1):]
            
            best_score = -1
            best_next = -1
            candidates = []
            
            # Evaluate each possible next symbol
            for c in range(1, n+1):
                # Form the new window
                new_window_list = last_n_minus_1 + [c]
                
                # Check if it's a valid permutation (all unique symbols)
                # Since last_n_minus_1 has n-1 unique symbols (from a perm), 
                # new_window is a perm iff c is not in last_n_minus_1.
                if c in last_n_minus_1:
                    continue
                    
                # Encode the new window
                new_val = 0
                for i in range(n):
                    new_val += new_window_list[i] * base_powers[i]
                    
                if new_val not in seen:
                    # This move adds 1 new permutation immediately.
                    # Now do lookahead
                    score = 1 # Immediate gain
                    
                    if lookahead_depth > 1:
                        # Simulate the next step
                        # The new state would have S + [c]
                        # The last n-1 of the new state is new_window_list[1:] + [next_c]
                        # Wait, the last n-1 of S+[c] is S[-(n-2):] + [c]? No.
                        # S = [s1, ..., sL]. New S' = [s1, ..., sL, c].
                        # Last n-1 of S' is S[-(n-2):] + [c] if n>1?
                        # Let's trace: n=3. S=[1,2]. Append 3. S'=[1,2,3]. Last 2: [2,3].
                        # General: Last n-1 of S' is S[-(n-2):] + [c] is WRONG.
                        # S has length L. S' has length L+1.
                        # Last n-1 elements of S' are S[L-(n-2):L] + [c] ?
                        # Indices: L-n+2 to L-1 from S, and L from S'.
                        # Yes.
                        
                        prev_n_minus_2 = S[-(n-2):] if n > 2 else []
                        
                        max_lookahead_gain = 0
                        for next_c in range(1, n+1):
                            if next_c in new_window_list[1:]:
                                continue
                            # Next window is new_window_list[1:] + [next_c]
                            next_window_list = new_window_list[1:] + [next_c]
                            next_val = 0
                            for i in range(n):
                                next_val += next_window_list[i] * base_powers[i]
                            if next_val not in seen and next_val != new_val:
                                max_lookahead_gain += 1
                                
                        score += max_lookahead_gain * 0.5 # Weighted lookahead
                        
                        # Add some randomness to break ties
                        score += random.random() * 0.1
                        candidates.append((score, c))
                    else:
                        score += random.random() * 0.1
                        candidates.append((score, c))
                        
            if not candidates:
                # Dead end, shouldn't happen if we started with a perm and only append valid perms
                # But if we get stuck, break
                break
                
            # Pick the best candidate
            candidates.sort(reverse=True)
            best_c = candidates[0][1]
            
            # Append best_c
            S.append(best_c)
            
            # Update seen
            new_val = 0
            new_window_list = S[-n:]
            for i in range(n):
                new_val += new_window_list[i] * base_powers[i]
            seen.add(new_val)
            
            # Time check
            if time.time() > deadline - 5:
                break

        return S

    # 3. Main Search Loop: Try multiple greedy constructions with different seeds and lookahead depths
    best_S = None
    best_len = float('inf')
    
    # Strategy:
    # - Start with lookahead 1 (fast), then 2 (slower but better).
    # - Use many random seeds.
    
    # Pre-check: if n is small, we might find optimal quickly.
    # For n=7, optimal is ~5900. Greedy usually gets ~6000-6200.
    
    iterations = 0
    max_iterations = 5000 # Will be limited by time
    
    # Phase 1: Lookahead 1, many restarts
    # We prioritize finding *any* valid solution quickly with simple greedy
    while time.time() < deadline - 20 and iterations < 100:
        seed = random.randint(0, 100000)
        S = greedy_construct(1, seed)
        if len(S) < best_len:
            best_len = len(S)
            best_S = S[:]
        iterations += 1
        
    # Phase 2: Lookahead 2, fewer restarts
    # Try to improve with better lookahead
    while time.time() < deadline - 5 and iterations < 200:
        seed = random.randint(0, 100000)
        S = greedy_construct(2, seed)
        if len(S) < best_len:
            best_len = len(S)
            best_S = S[:]
        iterations += 1
        
    # Check if best_S is valid
    if best_S is not None and is_valid(best_S, n):
        return "".join(map(str, best_S))
        
    # Fallback: return a guaranteed valid superpermutation
    # This ensures the program always returns a valid result, even if greedy fails
    fallback = fallback_construct(n)
    return "".join(map(str, fallback))

# Test locally
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 7
        
    start = time.time()
    res = construct(n)
    end = time.time()
    
    # Verify
    base = n + 1
    base_powers = [base**i for i in range(n)]
    seen = set()
    for i in range(len(res) - n + 1):
        val = 0
        for j in range(n):
            val += int(res[i+j]) * base_powers[j]
        seen.add(val)
        
    print(f"n={n}, Length={len(res)}, Time={end-start:.2f}s, PermsFound={len(seen)}/{n**0 * __import__('math').factorial(n)}")
