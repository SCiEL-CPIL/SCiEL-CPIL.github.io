import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    max_val = 512
    target_size = 48
    
    # 1. Generate a high-quality Sidon (B2) sequence to serve as the core.
    #    A Sidon sequence has distinct pairwise sums, so no element is the mean of any pair.
    #    We use a greedy construction for the Sidon sequence.
    def generate_sidon_sequence(limit):
        seq = []
        used_sums = set()
        # Start with small numbers to maximize density
        for i in range(1, limit + 1):
            ok = True
            for j in range(len(seq)):
                s = seq[j] + i
                if s in used_sums:
                    ok = False
                    break
                # Also need to ensure i is not the mean of any two existing
                # But Sidon property (distinct sums) implies a_i != (a_j + a_k)/2 
                # because 2*a_i = a_j + a_k would imply sum a_j+a_k is even and equals 2*a_i.
                # Actually, distinct sums of pairs (a,b) with a<b implies 
                # if a_j + a_k = 2*a_i, then the sum is unique.
                # Wait, the condition is "no element is the mean of ANY subset of at least two OTHER elements".
                # For k=2, this means no a_i = (a_j + a_k)/2.
                # If all pairwise sums a_j + a_k are distinct, does that prevent a_i being a mean?
                # If a_j + a_k = 2*a_i, then the sum 2*a_i is achieved by the pair (j,k).
                # Is 2*a_i a sum of any other pair? Not necessarily relevant.
                # The condition for k=2 is simply: there do not exist distinct j,k != i such that a_j + a_k = 2*a_i.
                # A Sidon sequence ensures that if a_j + a_k = a_m + a_n, then {j,k}={m,n}.
                # This doesn't directly prevent a_j + a_k = 2*a_i unless 2*a_i is not a sum of any pair?
                # No, 2*a_i is just a number. The Sidon property is about uniqueness of sums.
                # However, it is known that Sidon sequences are "sum-free" in the sense that they don't contain solutions to x+y=z? No.
                # But if we construct the sequence such that all elements are large and spaced out, 
                # we can easily check k=2 in O(N) or O(1) with a set.
                
                # Let's stick to a robust check for the Sidon generation:
                # Ensure all pairwise sums are distinct.
            if ok:
                # Check if adding i creates a duplicate sum
                new_sums = [seq[j] + i for j in range(len(seq))]
                # Check if any new sum collides with existing sums
                collision = False
                for s in new_sums:
                    if s in used_sums:
                        collision = True
                        break
                if not collision:
                    seq.append(i)
                    used_sums.update(new_sums)
                    if len(seq) >= 30: # We want a core of ~30-35
                        break
        return seq

    core = generate_sidon_sequence(max_val)
    
    # The core is a valid set for k=2. 
    # We still need to check k>=3 for the core itself?
    # A Sidon sequence is not necessarily mean-free for k>=3.
    # Example: {1, 2, 4, 8} is not Sidon? 1+4=5, 2+? 
    # Let's verify the core is fully mean-free for all k.
    
    def is_mean_free_valid(set_list):
        """Full validation using bitset DP. Optimized for speed."""
        n = len(set_list)
        if n < 3:
            return True
        for i, a in enumerate(set_list):
            others = [x for j, x in enumerate(set_list) if j != i]
            m = len(others)
            if m < 2:
                continue
            
            # Bitset DP for subset sums of 'others'
            # dp[k] is a bitset where bit s is 1 if sum s is achievable with k elements
            dp = [0] * (m + 1)
            dp[0] = 1  # 0 elements, sum 0
            
            for x in others:
                for k in range(m, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            max_sum = sum(others)
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum:
                    break
                if dp[k] & (1 << target):
                    return False
        return True

    if not is_mean_free_valid(core):
        # Fallback: remove elements that cause violations? 
        # Or just start with a simpler, known good set.
        # For now, if the greedy Sidon fails full validity, we prune it.
        while not is_mean_free_valid(core) and len(core) > 5:
            # Remove the largest element to break sums
            core.pop()
            
    best_set = core[:]
    best_len = len(best_set)
    
    # 2. Greedy Extension
    # We will try to add elements from 1..512 that are not in the set.
    # To speed up, we maintain the DP state for the current set? 
    # That's complex because adding an element changes the DP for ALL other elements.
    # Instead, we use a fast incremental check for the NEW element, 
    # and periodically re-validate or use a very fast check for existing elements.
    
    # Actually, the most expensive part is checking if an existing element becomes a mean 
    # due to the addition of a new element.
    # Condition: a_i is mean of subset S of (Old \ {a_i}) U {new}.
    # This means sum(S) = k*a_i.
    # S can contain 'new' or not.
    # If S does not contain 'new', it was already checked.
    # If S contains 'new', then sum(S \ {new}) + new = k*a_i => sum(S') = k*a_i - new.
    # So for each existing a_i, we check if there is a subset of (Old \ {a_i}) 
    # with size k-1 and sum k*a_i - new.
    
    def try_add_element(current_set, x, deadline):
        """
        Try to add x to current_set.
        Returns True if successful, and updates current_set in place if so.
        """
        if x in current_set:
            return False
        
        n = len(current_set)
        # Check 1: Is x the mean of any subset of current_set?
        # DP for current_set
        dp = [0] * (n + 1)
        dp[0] = 1
        for v in current_set:
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << v)
        
        max_sum = sum(current_set)
        for k in range(2, n + 1):
            target = k * x
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
        
        # Check 2: Does adding x make any existing element a mean?
        # For each a in current_set:
        #   Others = current_set \ {a}
        #   We need to check if there is a subset T of Others such that:
        #   sum(T) + x = (len(T)+1) * a   (Case: x is in the mean-subset)
        #   => sum(T) = (len(T)+1)*a - x
        #   Note: The subset T must have size >= 1 if x is included? 
        #   The problem says "subset of at least two OTHER elements".
        #   If x is in the subset, the other elements are T.
        #   Total subset size is |T| + 1. This must be >= 2, so |T| >= 1.
        
        for i, a in enumerate(current_set):
            others = [v for j, v in enumerate(current_set) if j != i]
            m = len(others)
            
            # Build DP for 'others'
            dp_o = [0] * (m + 1)
            dp_o[0] = 1
            for v in others:
                for k in range(m, 0, -1):
                    if dp_o[k-1]:
                        dp_o[k] |= (dp_o[k-1] << v)
            
            max_sum_o = sum(others)
            # t is size of T. Total size is t+1.
            # We need t >= 1.
            for t in range(1, m + 1):
                total_k = t + 1
                target = total_k * a - x
                if target < 0:
                    continue
                if target > max_sum_o:
                    break
                if dp_o[t] & (1 << target):
                    return False
        
        # If we reach here, it's safe to add
        current_set.append(x)
        return True

    # 3. Search Loop
    # We perform multiple random restarts.
    # In each restart, we start with the core (or a subset) and greedily add elements.
    
    candidates = list(range(1, max_val + 1))
    
    for restart in range(100): # Enough to fill time
        if time.time() > deadline:
            break
            
        # Randomize the candidate order for diversity
        random.shuffle(candidates)
        
        # Start with a random subset of the core? Or the full core?
        # Using the full core is safer but might be suboptimal if it blocks better extensions.
        # Let's try starting with the core.
        current = core[:]
        
        # Optional: Local search to improve the core? 
        # For now, just extend.
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in current:
                continue
            
            # Fast pre-check: k=2 for x against current
            # Since core is Sidon, and we add elements, we must ensure x is not mean of pair in current.
            # 2*x = a + b.
            target_2x = 2 * x
            found_pair = False
            curr_set = set(current)
            for a in current:
                b = target_2x - a
                if b != a and b in curr_set:
                    found_pair = True
                    break
            if found_pair:
                continue
                
            # Also check if x creates a k=2 violation for existing elements?
            # If we add x, does some a become mean of b, x?
            # a = (b+x)/2 => 2a = b+x => b = 2a - x.
            # This is covered by the full check in try_add_element, 
            # but doing it here avoids the expensive DP if it fails fast.
            creates_pair_violation = False
            for a in current:
                b = 2 * a - x
                if b != a and b != x and b in curr_set:
                    creates_pair_violation = True
                    break
            if creates_pair_violation:
                continue

            # Full check
            if try_add_element(current, x, deadline):
                pass # Added
            
        if len(current) > best_len:
            best_set = current[:]
            best_len = len(best_set)
            
        # Early exit if we hit max possible? 48 is the cap, but maybe we can do better?
        # Max is 48 per contract.
        if best_len >= 48:
            break

    # 4. Final Local Search on best_set
    # Try to swap elements to increase size or fix issues? 
    # The problem is just maximize cardinality.
    # We can try to add any remaining numbers.
    
    if best_len < 48 and time.time() < deadline - 5:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, max_val + 1) if x not in best_set_set]
        random.shuffle(remaining)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in remaining:
                if time.time() > deadline - 2:
                    break
                if x in best_set_set:
                    continue
                # Fast k=2 check
                target_2x = 2 * x
                found_pair = False
                for a in best_set:
                    b = target_2x - a
                    if b != a and b in best_set_set:
                        found_pair = True
                        break
                if found_pair:
                    continue
                creates_pair_violation = False
                for a in best_set:
                    b = 2 * a - x
                    if b != a and b != x and b in best_set_set:
                        creates_pair_violation = True
                        break
                if creates_pair_violation:
                    continue

                if try_add_element(best_set, x, deadline):
                    best_set_set.add(x)
                    remaining.remove(x)
                    best_len = len(best_set)
                    improved = True
                    if best_len >= 48:
                        break

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    # Verification
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
    
    # Quick validity check
    def _valid_mean_free(selected):
        n = len(selected)
        if n < 3:
            return True
        for i, a in enumerate(selected):
            others = [x for j, x in enumerate(selected) if j != i]
            m = len(others)
            if m < 2:
                continue
            max_sum = sum(others)
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in others:
                for k in range(m, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum:
                    break
                if dp[k] & (1 << target):
                    return False
        return True
        
    assert _valid_mean_free(s), "Invalid set found!"
