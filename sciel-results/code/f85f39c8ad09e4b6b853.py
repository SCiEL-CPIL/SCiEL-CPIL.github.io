"""Rewritten program: backtracking with bitset divisibility mask for subset-sum constraints."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    best = [31, 32, 34]  # fallback

    # We need a set S of distinct ints in 1..256 such that:
    # 1. All nonempty subset sums are distinct (no two subsets have same sum).
    #    Sufficient: if we build greedily with each new element > sum of all previous.
    # 2. No two distinct subset sums divide one another.
    
    # Strategy: greedy/backtracking where each new element x satisfies:
    #   - x is not in current set
    #   - x > sum(current_set)  [ensures unique subset sums via binary-like property]
    #   - For every existing subset sum s, s does not divide (s + x) and (s + x) does not divide s
    #     Actually we need: for all pairs of subset sums of the NEW set, neither divides the other.
    #     When we add x, new subset sums are {old_sums} ∪ {old_sums + x} ∪ {x}
    #     We need to check all pairs among these.
    
    # Let's use a smarter approach:
    # Build set greedily. At each step, try candidates from 1..256 in some order.
    # Maintain the set of all current subset sums.
    # When adding candidate x:
    #   new_sums = old_sums ∪ {s + x for s in old_sums} ∪ {x}
    #   Check: all elements in new_sums are distinct (guaranteed if x > sum(old_set))
    #   Check: no two distinct elements in new_sums divide one another.
    
    # For efficiency, precompute divisibility: for a given set of sums, check pairwise.
    # But the number of subset sums can be up to 2^n which is too large for n=30.
    # However, with the "x > sum(previous)" constraint, the subset sums are all distinct
    # and there are exactly 2^k - 1 of them for k elements. For k=10, that's 1023 sums.
    # Checking all pairs is O(2^(2k)) which is too slow for k > 15.
    
    # Alternative: use the fact that if x > sum(prev), then subset sums have a structure.
    # Actually, let's just do backtracking with pruning and keep k small enough.
    
    # Let me try a different approach: random restarts with greedy construction.
    # Order candidates randomly, greedily add if valid.
    
    import random
    rng = random.Random(42)
    
    def is_valid_set(sums_set):
        """Check that no two distinct non-zero values in sums_set divide one another."""
        # The problem specifies NONEMPTY subset sums. 0 corresponds to the empty set.
        # We must exclude 0 from the divisibility check.
        lst = sorted([x for x in sums_set if x != 0])
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[j] % lst[i] == 0:
                    return False
        return True
    
    def try_greedy(candidates, current_set, current_sums):
        """Greedily try to add candidates in given order. Returns (set, sums) of best found."""
        cur = list(current_set)
        cur_sums = set(current_sums)
        total_sum = sum(cur)
        
        for x in candidates:
            if x in cur:
                continue
            # Check x > total_sum for unique subset sums
            if x <= total_sum:
                continue
            # New sums
            new_sums = set(cur_sums)
            new_sums.add(x)
            for s in cur_sums:
                new_sums.add(s + x)
            # Check divisibility
            if is_valid_set(new_sums):
                cur.append(x)
                cur_sums = new_sums
                total_sum += x
        
        return cur, cur_sums
    
    # Try multiple random orderings
    base_candidates = list(range(1, 257))
    
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        shuffled = base_candidates[:]
        rng.shuffle(shuffled)
        result, _ = try_greedy(shuffled, [], {0})
        # Remove 0 from consideration - actually our sums should be nonempty subset sums
        # Let me fix: current_sums should start empty, and we track nonempty subset sums
        if len(result) > len(best):
            best = result
    
    # Also try a more systematic approach: start with small elements
    # and build up
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        # Start with a random small seed
        start_val = rng.randint(1, 50)
        cur = [start_val]
        cur_sums = {start_val}
        total_sum = start_val
        
        candidates = [x for x in range(1, 257) if x > total_sum and x != start_val]
        rng.shuffle(candidates)
        
        for x in candidates:
            if x <= total_sum:
                continue
            new_sums = set(cur_sums)
            new_sums.add(x)
            for s in cur_sums:
                new_sums.add(s + x)
            if is_valid_set(new_sums):
                cur.append(x)
                cur_sums = new_sums
                total_sum += x
                if total_sum > 256:
                    break
        
        if len(cur) > len(best):
            best = cur
    
    # Final verification of best
    if best:
        # Verify unique subset sums
        all_sums = set()
        n = len(best)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += best[i]
            if s in all_sums:
                break
            all_sums.add(s)
        else:
            # Verify no divisibility
            valid = True
            lst = sorted(all_sums)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if lst[j] % lst[i] == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
