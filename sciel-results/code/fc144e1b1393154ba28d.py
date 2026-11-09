import time
import random
from itertools import combinations

def is_valid_set(S):
    """
    Check if the set S is valid.
    S is a list of distinct integers.
    Condition: No element a in S is the mean of any subset T of S\{a} with |T| >= 2.
    """
    if len(S) < 3:
        return True
    
    # To speed up, we can check each a
    for a in S:
        others = [x for x in S if x != a]
        n_others = len(others)
        
        # Check all subset sizes k from 2 to n_others
        # For efficiency, we can break early if found invalid
        for k in range(2, n_others + 1):
            # Check if any combination of k elements from others sums to k*a
            target_sum = k * a
            # Optimization: If min sum of k smallest > target or max sum of k largest < target, skip
            # But sorting every time is expensive.
            # Simple check:
            for comb in combinations(others, k):
                if sum(comb) == target_sum:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_set = []
    
    # Greedy approach: Try to add numbers from 1 to 512
    # We can try multiple orderings to find a larger set
    # Ordering 1: Ascending
    # Ordering 2: Descending
    # Ordering 3: Random
    
    def greedy_attempt(order):
        current = []
        for x in order:
            if time.time() > deadline:
                return current
            # Check if adding x keeps the set valid
            # We only need to check if x is mean of subset of current,
            # and if any a in current becomes mean of subset of current+{x}
            
            # Efficient check for adding x:
            # 1. Is x the mean of a subset T of current with |T|>=2?
            #    sum(T) = |T|*x
            valid = True
            # Check 1: x is mean of subset of current
            for k in range(2, len(current) + 1):
                target = k * x
                # Check if any k-subset of current sums to target
                # Using combinations is slow for large len(current)
                # But len(current) is at most 48.
                # 48 choose 24 is huge.
                # We need a better check.
                
                # Let's use a DP or bitset approach for subset sums?
                # Or just check if the target is achievable.
                # Given the small max size (48), maybe we can just use a set of achievable sums for each k?
                # But we do this for every candidate x.
                
                # Alternative:
                # If the set is "average-free", it has strong structure.
                # Let's try a simpler heuristic:
                # If x is added, check all a in current + [x].
                
            # Let's just use the full is_valid_set on current + [x] for simplicity first,
            # and optimize if it's too slow.
            # With max 48 elements, is_valid_set might be slow if done naively.
            # Combinations of 47 elements taken 2..47 at a time is too slow.
            
            # We need a faster validity check.
            pass
            
        return current

    # The naive is_valid_set is too slow for greedy with 512 candidates.
    # We need to optimize the check "Does adding x invalidate the set?"
    
    # Optimized Check:
    # When adding x to S:
    # 1. Check if x is the mean of any subset of S.
    # 2. Check if any a in S is the mean of any subset of S U {x} that includes x.
    
    def check_addition(S, x):
        # 1. x is mean of subset T of S, |T|>=2
        # sum(T) = |T|*x
        # This is equivalent to: Is there a subset of S with sum = k*x for some k>=2?
        # We can precompute subset sums of S?
        # S changes.
        
        # Let's try a different strategy:
        # Since the max size is small (48), we can maintain the set of "forbidden" means?
        # No.
        
        # Let's use the property that if S is valid, we can check x against S.
        # For step 1:
        # Iterate k from 2 to len(S).
        # Check if there is a subset of S of size k summing to k*x.
        # This is a subset sum problem.
        # For k=2: is there a,b in S such that a+b=2x? i.e., 2x-a=b?
        # For k=3: a+b+c=3x?
        
        # Since we are building the set, we can maintain a dictionary of achievable sums for each size?
        # But S grows.
        
        # Let's try a greedy with a limited search depth or a smarter ordering.
        # Often, picking numbers that are far apart works well.
        # E.g., powers of 2: 1, 2, 4, 8... 2^8=256, 2^9=512.
        # That's only 10 numbers.
        # We need 48.
        
        # A better heuristic:
        # Start with a random set of 48 numbers and try to repair?
        # Or start with an empty set and add numbers that are "least likely" to cause conflicts.
        
        # Let's try a simple greedy with a fast check for small k.
        # Most conflicts happen with small k (2 or 3).
        # If we avoid k=2 and k=3 conflicts, we might be safe?
        # No, the task requires checking ALL k.
        
        # However, if a set is valid for k=2 and k=3, is it valid for larger k?
        # Not necessarily.
        
        # Let's assume the fast path:
        # Check k=2: Is x the midpoint of any pair in S?
        # Check k=3: Is x the mean of any triple in S?
        # And for existing a:
        # Is a the midpoint of (x, b) for some b in S? -> 2a = x+b -> b=2a-x in S?
        # Is a the mean of (x, b, c)? -> 3a = x+b+c -> b+c = 3a-x.
        
        # If we only check small k, we might accept invalid sets.
        # But maybe we can do a full check only when a candidate is accepted,
        # and use the fast check to filter out obvious bad candidates.
        
        return True # Placeholder

    # This is getting complex. Let's use a standard library approach with optimization.
    # We will use a greedy algorithm that tries to add candidates.
    # To handle the "all k" constraint efficiently, we can use the fact that
    # the number of elements is small.
    
    # Let's try a different tactic:
    # Randomized Local Search.
    # 1. Generate a random set of 48 numbers.
    # 2. Check validity.
    # 3. If invalid, remove one element involved in a conflict.
    # 4. Try to add a new random number to replace it.
    
    # Or:
    # 1. Start with empty set.
    # 2. Add numbers one by one.
    # 3. When adding x, check validity.
    #    To check validity fast:
    #    - Precompute all subset sums of the current set S?
    #    - S is small (<=48).
    #    - We can maintain a list of (sum, size) for all subsets?
    #    - Number of subsets is 2^48, too large.
    
    # We must rely on the structure.
    # Actually, for the specific problem of "no element is the mean of a subset",
    # a simple greedy often works well if we pick numbers in a specific order.
    
    # Let's try:
    # 1. Sort candidates.
    # 2. Greedily add if valid.
    # 3. Use a timeout.
    
    # To make the validity check fast enough:
    # We will check the condition:
    # For each a in S U {x}:
    #   For each k in 2..|S U {x}|-1:
    #     Does there exist a subset T of (S U {x})\{a} of size k with sum k*a?
    
    # We can optimize the subset sum check for a fixed target and size using DP?
    # Or just use combinations if the set is small?
    # If |S| is 20, combinations are manageable.
    # If |S| is 40, it's not.
    
    # However, note that if a set is "average-free", it tends to be sparse.
    # Maybe the greedy will stop growing before it becomes too large to check?
    # No, we want it to grow to 48.
    
    # Let's try a hybrid:
    # 1. Greedy add.
    # 2. If |S| > 20, use a faster approximate check? No, must be exact.
    # 3. Use bitsets for subset sums?
    #    Max sum is 48*512 = 24576.
    #    We can use a bitset of size 24577.
    #    For each element, update the bitset.
    #    But we need to know the *size* of the subset too.
    #    So we need a 2D bitset: dp[size][sum].
    #    Size up to 48, Sum up to 24576.
    #    48 * 24576 bits = 1.1 million bits = 140 KB.
    #    This is very small!
    #    We can maintain a list of bitsets, one for each size.
    #    dp[k] is a bitset where bit i is 1 if there is a subset of size k with sum i.
    
    # When adding x:
    # 1. Update dp for the new set S' = S U {x}.
    # 2. For each a in S':
    #    For each k in 2..|S'|-1:
    #       Check if dp'[k] has bit set at k*a.
    #       BUT, the subset must not contain a.
    #       So we need to check if there is a subset of S'\{a} of size k with sum k*a.
    #       We can compute this by removing a from the DP?
    #       Or, we can maintain the DP for the current set S.
    #       When checking if we can add x:
    #       - Check if x is mean of subset of S:
    #         For k=2..|S|, check if dp_S[k] has bit at k*x.
    #       - Check if any a in S is mean of subset of S U {x} containing x:
    #         For each a in S:
    #           For k=2..|S|:
    #             We need a subset T of S of size k-1 such that sum(T) + x = k*a.
    #             sum(T) = k*a - x.
    #             Check if dp_S[k-1] has bit at k*a - x.
    
    # This is efficient!
    # We maintain dp_S: list of bitsets (Python ints) for sizes 0..|S|.
    # dp_S[0] = 1 (sum 0)
    # When adding x:
    #   new_dp = dp_S.copy()
    #   for k from |S| down to 0:
    #       new_dp[k+1] |= (dp_S[k] << x)
    #   
    #   Then check validity:
    #   1. For k=2..|S|: if (new_dp[k] >> (k*x)) & 1:
    #        # Wait, new_dp includes subsets with x.
    #        # We need to check if x is mean of subset of S (old set).
    #        # So check dp_S[k] at k*x.
    #   2. For each a in S:
    #        For k=2..|S|+1:
    #           # Subset of new set of size k, mean a.
    #           # Must contain x?
    #           # If it doesn't contain x, it was already checked in S.
    #           # So we only need to check subsets that CONTAIN x.
    #           # Such a subset is {x} U T, where T is subset of S, size k-1.
    #           # Sum = x + sum(T) = k*a.
    #           # sum(T) = k*a - x.
    #           # Check if dp_S[k-1] has bit at k*a - x.
    
    # This works!

    def construct_with_bitset():
        start_time = time.time()
        deadline = start_time + 80.0
        
        selected = []
        # dp[k] is an integer bitset. Bit i is 1 if there is a subset of size k with sum i.
        # Max sum is 512 * 48 = 24576.
        # We need bits up to 24576.
        
        dp = [0] * 50  # dp[0] to dp[49]
        dp[0] = 1  # Empty set sum 0
        
        for x in range(1, 513):
            if time.time() > deadline:
                break
                
            # Check 1: Is x the mean of any subset of 'selected'?
            valid = True
            for k in range(2, len(selected) + 1):
                target_sum = k * x
                if target_sum > 24576:
                    continue
                # Check if dp[k] has bit at target_sum
                if (dp[k] >> target_sum) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check 2: Does adding x make any existing 'a' the mean of a subset containing x?
            for a in selected:
                # Subset size k in new set, containing x.
                # Corresponds to subset of size k-1 in old set.
                # k ranges from 2 to len(selected)+1
                for k in range(2, len(selected) + 2):
                    target_sum_for_T = k * a - x
                    if target_sum_for_T < 0:
                        continue
                    if target_sum_for_T > 24576:
                        continue
                    # Check if dp[k-1] has bit at target_sum_for_T
                    if (dp[k-1] >> target_sum_for_T) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Add x to selected and update dp
                selected.append(x)
                # Update dp: new_dp[k+1] |= dp[k] << x
                # We iterate backwards to avoid using x multiple times in one step
                for k in range(len(selected) - 1, -1, -1):
                    dp[k+1] |= (dp[k] << x)
                    
                if len(selected) == 48:
                    break
        
        return selected

    result = construct_with_bitset()
    return result

if __name__ == "__main__":
    s = construct()
    print(len(s))
