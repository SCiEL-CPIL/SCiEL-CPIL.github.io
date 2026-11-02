import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill
    
    def time_up():
        return (time.time() - start_time) > DEADLINE

    # Handle small cases explicitly
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Classic superperm for n=4 is length 33
        # 1234124314234132431243142132413421432143123412314234
        # Let's generate it using the recursive method to be safe
        pass
        
    if n <= 4:
        # Use the classic recursive construction for small n
        return "".join(map(str, _classic_superperm_list(n)))

    # Generate the standard superpermutation for n-1 to get a valid set of permutations
    # and an initial ordering.
    # We use the recursive construction for n-1.
    base_word = _classic_superperm_list(n - 1)
        
    # Extract permutations of size n-1 from base_word
    perm_len = n - 1
    perms_seq = []
    seen = set()
    # base_word is a list of integers
    for i in range(len(base_word) - perm_len + 1):
        window = tuple(base_word[i:i + perm_len])
        if len(set(window)) == perm_len and window not in seen:
            seen.add(window)
            perms_seq.append(window)
            
    # Ensure we have all (n-1)! permutations
    expected_count = 1
    for i in range(2, n):
        expected_count *= i
    if len(perms_seq) != expected_count:
        # Fallback: generate all permutations explicitly if extraction failed (shouldn't happen)
        perms_seq = list(permutations(range(1, n)))
        
    num_perms = len(perms_seq)
    
    # Precompute overlap matrix M
    # M[i][j] = max k such that perms_seq[i][-k:] == perms_seq[j][:k]
    # k ranges from 0 to n-1
    M = [[0] * num_perms for _ in range(num_perms)]
    
    # Optimization: Group permutations by their suffixes/prefixes
    # For each possible suffix of length k, map to list of indices
    # But n is small (7), so direct computation is fast enough for 720x720
    # 720^2 * 6 operations ~ 3M ops. Very fast.
    
    for i in range(num_perms):
        p_i = perms_seq[i]
        for j in range(num_perms):
            p_j = perms_seq[j]
            max_ov = 0
            # Check from largest possible overlap down
            # Max overlap is n-1
            for k in range(n - 1, 0, -1):
                if p_i[-k:] == p_j[:k]:
                    max_ov = k
                    break
            M[i][j] = max_ov
            
    # We want to find an ordering of indices 0..num_perms-1 that maximizes sum(M[order[k]][order[k+1]])
    # Initial order: 0, 1, ..., num_perms-1 (the order they appeared in base_word)
    # Note: The order in base_word is already a good heuristic because it's derived from a valid superpermutation.
    
    order = list(range(num_perms))
    
    def calc_score(order):
        total = 0
        for i in range(len(order) - 1):
            total += M[order[i]][order[i+1]]
        return total

    best_score = calc_score(order)
    best_order = order[:]
    
    # Local Search: 2-Opt with precomputed segment costs
    # To make 2-opt efficient, we maintain:
    # 1. The current order
    # 2. Prefix sums of edge weights for the current order
    #    pref[i] = sum_{k=0}^{i-1} M[order[k]][order[k+1]]
    #    So sum of edges from i to j-1 is pref[j] - pref[i]
    # 3. Reverse prefix sums? 
    #    When we reverse [i, j], the internal edges become M[order[k+1]][order[k]] instead of M[order[k]][order[k+1]].
    #    Let's maintain a separate structure or just recalculate the internal sum for the segment if it's small?
    #    No, 2-opt can involve large segments.
    #    
    #    Let's maintain two prefix sum arrays:
    #    fwd_pref[i] = sum_{k=0}^{i-1} M[order[k]][order[k+1]]
    #    rev_pref[i] = sum_{k=0}^{i-1} M[order[k+1]][order[k]]
    #    
    #    When we reverse [i, j]:
    #    Old internal sum = fwd_pref[j] - fwd_pref[i]
    #    New internal sum = rev_pref[j] - rev_pref[i]
    #    
    #    Boundary edges:
    #    Old: M[order[i-1]][order[i]] + M[order[j]][order[j+1]]
    #    New: M[order[i-1]][order[j]] + M[order[i]][order[j+1]]
    #    
    #    Delta = (New Internal + New Boundaries) - (Old Internal + Old Boundaries)
    #    
    #    After reversal, we need to update fwd_pref and rev_pref.
    #    This is O(N) per update if we do it naively.
    #    With N=720, O(N) per move * O(N^2) moves = O(N^3) total. Too slow.
    #    
    #    Alternative: Only use 2-opt for short segments? Or use a different heuristic.
    #    
    #    Let's stick to Swap-based local search but optimize the delta calculation.
    #    For a swap of a and b (a < b):
    #    Old: ... L, A, ..., B, R ...
    #    New: ... L, B, ..., A, R ...
    #    
    #    If b == a + 1:
    #    Old: M[L][A] + M[A][B] + M[B][R]
    #    New: M[L][B] + M[B][A] + M[A][R]
    #    Delta = M[L][B] + M[B][A] + M[A][R] - M[L][A] - M[A][B] - M[B][R]
    #    This is O(1).
    #    
    #    If b > a + 1:
    #    Old: M[L][A] + Sum_{k=a}^{b-1} M[order[k]][order[k+1]] + M[B][R]
    #    New: M[L][B] + Sum_{k=a}^{b-1} M[order[k+1]][order[k]] + M[A][R]
    #    
    #    We can precompute the forward and reverse segment sums.
    #    Let F(i, j) = Sum_{k=i}^{j-1} M[order[k]][order[k+1]]
    #    Let R(i, j) = Sum_{k=i}^{j-1} M[order[k+1]][order[k]]
    #    
    #    Delta = M[L][B] + R(a, b) + M[A][R] - M[L][A] - F(a, b) - M[B][R]
    #    
    #    If we maintain prefix sums for F and R, we can compute F(a, b) and R(a, b) in O(1).
    #    F(a, b) = fwd_pref[b] - fwd_pref[a]
    #    R(a, b) = rev_pref[b] - rev_pref[a]
    #    
    #    So Delta is O(1) to compute.
    #    
    #    However, updating fwd_pref and rev_pref after a swap is O(N).
    #    If we do many swaps, this adds up.
    #    
    #    Strategy:
    #    1. Maintain fwd_pref and rev_pref.
    #    2. Iterate through all pairs (a, b).
    #    3. Compute Delta in O(1).
    #    4. If Delta > 0, perform swap and update prefs in O(N).
    #    5. Repeat until no improvement.
    #    
    #    Complexity: O(N^3) worst case per pass? No, O(N^2) checks, O(N) updates.
    #    If we do K swaps per pass, cost is O(N^2 + K*N).
    #    With N=720, N^2 = 518k. If K is small, this is fast.
    
    current_order = best_order[:]
    current_score = best_score
    
    # Build prefix sums
    def build_prefs(order):
        fwd_pref = [0] * (len(order) + 1)
        rev_pref = [0] * (len(order) + 1)
        for i in range(len(order) - 1):
            u = order[i]
            v = order[i+1]
            fwd_pref[i+1] = fwd_pref[i] + M[u][v]
            rev_pref[i+1] = rev_pref[i] + M[v][u]
        return fwd_pref, rev_pref

    fwd_pref, rev_pref = build_prefs(current_order)
    
    improved = True
    max_passes = 100
    pass_count = 0
    
    while improved and pass_count < max_passes and not time_up():
        improved = False
        pass_count += 1
        # Randomize start to avoid bias? Or just scan all.
        # Scanning all is O(N^2).
        # To avoid getting stuck, we can randomize the order of checking.
        # But O(N^2) checks is ~500k. In Python, this might take ~0.5-1s per pass.
        # With 100 passes, that's 50-100s. Might be tight.
        # Let's limit the number of swaps per pass or use a faster check.
        
        # Actually, let's just do adjacent swaps first (O(N) per pass), then try non-adjacent if time permits.
        # Or, just do all swaps but break early if time is running out.
        
        # Let's try a random subset of swaps to speed up?
        # No, let's just do the full O(N^2) pass but with a tight time check.
        
        # To speed up the Python loop, we can use local variables for M, fwd_pref, rev_pref, current_order.
        M_local = M
        fwd_local = fwd_pref
        rev_local = rev_pref
        order_local = current_order
        N_local = num_perms
        
        for a in range(N_local):
            if time_up():
                break
            # L is order[a-1] if a>0 else -1
            L = order_local[a-1] if a > 0 else -1
            A = order_local[a]
            
            for b in range(a + 1, N_local):
                if time_up():
                    break
                B = order_local[b]
                # R is order[b+1] if b<N-1 else -1
                R = order_local[b+1] if b + 1 < N_local else -1
                
                # Calculate Delta
                # Old: M[L][A] + F(a, b) + M[B][R]
                # New: M[L][B] + R(a, b) + M[A][R]
                
                old_boundary_1 = M_local[L][A] if L != -1 else 0
                new_boundary_1 = M_local[L][B] if L != -1 else 0
                
                old_boundary_2 = M_local[B][R] if R != -1 else 0
                new_boundary_2 = M_local[A][R] if R != -1 else 0
                
                old_internal = fwd_local[b] - fwd_local[a]
                new_internal = rev_local[b] - rev_local[a]
                
                delta = (new_boundary_1 + new_internal + new_boundary_2) - (old_boundary_1 + old_internal + old_boundary_2)
                
                if delta > 0:
                    # Perform swap
                    order_local[a], order_local[b] = order_local[b], order_local[a]
                    current_score += delta
                    improved = True
                    
                    # Update prefix sums
                    # This is O(N).
                    # We can update in place.
                    # fwd_pref[k] for k <= a is unchanged.
                    # fwd_pref[k] for a < k <= b changes.
                    # fwd_pref[k] for k > b changes by delta.
                    
                    # Let's just rebuild? O(N) anyway.
                    # Or update carefully.
                    # fwd_pref[b] = fwd_pref[b] + delta
                    # for k in b+1..N: fwd_pref[k] += delta
                    # But the internal values also change structure.
                    # Rebuilding is simpler and O(N).
                    fwd_pref, rev_pref = build_prefs(order_local)
                    fwd_local = fwd_pref
                    rev_local = rev_pref
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_order = order_local[:]
                        
    # Construct the final word from best_order
    # The word is built by concatenating blocks p + [n] + p, overlapping as much as possible.
    # We have the optimal order of permutations.
    
    if not best_order:
        return ""
        
    # First block
    first_p = perms_seq[best_order[0]]
    word = list(first_p) + [n] + list(first_p)
    
    # Subsequent blocks
    for i in range(1, len(best_order)):
        p_curr = perms_seq[best_order[i]]
        block = list(p_curr) + [n] + list(p_curr)
        
        # Find max overlap between current word suffix and block prefix
        # Max possible overlap is n-1 (length of p_curr)
        # Actually, the suffix of the previous block is p_prev.
        # The prefix of the current block is p_curr.
        # Overlap is max k such that word[-k:] == block[:k].
        # Since block starts with p_curr, and word ends with p_prev,
        # the overlap is determined by p_prev and p_curr.
        # We already computed M[prev_idx][curr_idx].
        
        prev_idx = best_order[i-1]
        curr_idx = best_order[i]
        ov = M[prev_idx][curr_idx]
        
        # Append block[ov:]
        word += block[ov:]
        
    return "".join(map(str, word))

def _classic_superperm_list(n):
    """Returns the classic superpermutation for n as a list of ints."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

# Test locally
if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        print(f"n={n}, len={len(w)}")
        # Verify validity
        if n <= 5:
            perms = set(permutations(range(1, n+1)))
            found = set()
            for i in range(len(w) - n + 1):
                win = tuple(int(c) for c in w[i:i+n])
                if len(set(win)) == n:
                    found.add(win)
            if found != perms:
                print(f"Error for n={n}: Missing {len(perms - found)} perms")
            else:
                print(f"Valid for n={n}")
