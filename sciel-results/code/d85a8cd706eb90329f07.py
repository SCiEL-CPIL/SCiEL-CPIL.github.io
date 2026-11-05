import time
import random

_DEADLINE = 85.0

def _is_mean_of_subset(C, i):
    """
    Check if C[i] is the mean of any subset of at least 2 OTHER elements in C.
    """
    a = C[i]
    others = C[:i] + C[i+1:]
    n = len(others)
    if n < 2:
        return False
    
    zs = [y - a for y in others]
    
    # Quick check: pairs
    # If there are two elements that sum to 0, i.e., y1 - a = -(y2 - a) => y1 + y2 = 2a
    # We can check this in O(N^2) or O(N) with a set.
    seen = set()
    for z in zs:
        if -z in seen:
            return True
        seen.add(z)
        
    # If we get here, no pair sums to 0.
    # Now check subsets of size >= 3.
    # We can use DP.
    
    # Sort zs to help with bounds?
    # Let's use the DP from the previous version but optimized.
    
    # Separate pos/neg/zero
    pos = sorted([z for z in zs if z > 0])
    neg = sorted([-z for z in zs if z < 0]) # abs values, sorted asc
    zeros = sum(1 for z in zs if z == 0)
    
    if zeros >= 2:
        return True
    if zeros == 1:
        # Need sum 0 from pos/neg with size >= 1 (since 1 zero is used, need 1 more to make size >= 2? 
        # Wait, subset size >= 2.
        # If we use the zero, we need one more element z such that 0 + z = 0 => z=0.
        # But we only have 1 zero. So we can't form sum 0 with size 2 using the zero.
        # We need to form sum 0 from pos/neg with size >= 2?
        # Or sum 0 from pos/neg with size >= 1 AND use the zero?
        # If we use zero, we need sum 0 from remaining.
        # If we don't use zero, we need sum 0 from pos/neg with size >= 2.
        # So:
        # 1. Check if pos/neg can form 0 with size >= 2.
        # 2. Check if pos/neg can form 0 with size >= 1 (and we add the zero, making size >= 2).
        
        # This logic is getting complex. The DP handles it all.
        pass
        
    # DP Approach
    # We want to know if 0 is achievable with size >= 2.
    
    # Total sum bounds
    total_pos = sum(pos)
    total_neg = sum(neg)
    
    # Offset for bitset
    offset = total_neg
    max_bit = total_pos + total_neg
    
    # dp[k] is bitset
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    # Track min/max sum for each k to prune
    min_sum = [0] * (n + 1)
    max_sum = [0] * (n + 1)
    min_sum[0] = 0
    max_sum[0] = 0
    reachable = [True] + [False] * n
    
    for z in zs:
        if z == 0:
            # Zero can be added to any existing subset
            # This increases size by 1, sum by 0
            for k in range(n, 0, -1):
                if reachable[k-1]:
                    dp[k] |= dp[k-1]
                    # min/max don't change
                    if not reachable[k]:
                        min_sum[k] = min_sum[k-1]
                        max_sum[k] = max_sum[k-1]
                    else:
                        min_sum[k] = min(min_sum[k], min_sum[k-1])
                        max_sum[k] = max(max_sum[k], max_sum[k-1])
                    reachable[k] = True
            continue
            
        if z > 0:
            for k in range(n, 0, -1):
                if not reachable[k-1]:
                    continue
                shifted = dp[k-1] << z
                dp[k] |= shifted
                if reachable[k]:
                    min_sum[k] = min(min_sum[k], min_sum[k-1] + z)
                    max_sum[k] = max(max_sum[k], max_sum[k-1] + z)
                else:
                    min_sum[k] = min_sum[k-1] + z
                    max_sum[k] = max_sum[k-1] + z
                reachable[k] = True
        else:
            for k in range(n, 0, -1):
                if not reachable[k-1]:
                    continue
                shifted = dp[k-1] >> (-z)
                dp[k] |= shifted
                if reachable[k]:
                    min_sum[k] = min(min_sum[k], min_sum[k-1] + z)
                    max_sum[k] = max(max_sum[k], max_sum[k-1] + z)
                else:
                    min_sum[k] = min_sum[k-1] + z
                    max_sum[k] = max_sum[k-1] + z
                reachable[k] = True
                
    for k in range(2, n + 1):
        if reachable[k] and dp[k] & (1 << offset):
            return True
    return False

def _is_valid_incremental(selected, x):
    """
    Check if x can be added to selected without violating constraints.
    """
    # Check 1: x is mean of subset of selected (size >= 2)
    if len(selected) >= 2:
        if _is_mean_of_subset(selected + [x], len(selected)): # x is at index len(selected)
            return False

    # Check 2: any y in selected is mean of subset of selected U {x} containing x
    # This is equivalent to: is y mean of subset of (selected U {x}) \ {y} ?
    # (selected U {x}) \ {y} contains x and selected \ {y}.
    # So we check if y is mean of subset of (selected \ {y}) U {x} with size >= 2.
    # This is NOT the same as checking if y is mean of subset of selected \ {y} (size >= 1) + x.
    # It allows subsets that include x and any number of elements from selected \ {y}.
    
    # Let's re-evaluate.
    # Condition: No element $z \in S$ is mean of subset $T \subset S \setminus \{z\}$ with $|T| \ge 2$.
    # When we add $x$, the new set is $S' = S \cup \{x\}$.
    # 1. $x$ must not be mean of subset of $S$ (size $\ge 2$). (Checked above)
    # 2. For any $y \in S$, $y$ must not be mean of subset of $S' \setminus \{y\}$ (size $\ge 2$).
    #    $S' \setminus \{y\} = (S \setminus \{y\}) \cup \{x\}$.
    #    Let $A = S \setminus \{y\}$. We need to check if $y$ is mean of subset of $A \cup \{x\}$ with size $\ge 2$.
    #    This means there is a subset $T \subseteq A \cup \{x\}$, $|T| \ge 2$, such that $\sum_{t \in T} t = |T| y$.
    #    This is equivalent to: there is a subset $T' \subseteq A$ and a boolean $b \in \{0, 1\}$ (whether $x \in T$)
    #    such that $|T'| + b \ge 2$ and $\sum_{t \in T'} t + b \cdot x = (|T'| + b) y$.
    #    Rearranging: $\sum_{t \in T'} (t - y) + b \cdot (x - y) = 0$.
    #    Let $d_t = t - y$ for $t \in A$. Let $d_x = x - y$.
    #    We need $\sum_{t \in T'} d_t + b \cdot d_x = 0$ with $|T'| + b \ge 2$.
    
    # We can check this by running a DP on the values $A$ and checking if we can form $-b \cdot d_x$
    # with size $|T'|$ such that $|T'| + b \ge 2$.
    
    # Actually, it's easier to just call `_is_mean_of_subset` on the new set?
    # No, that's $O(N^2)$ calls, each $O(N \cdot \text{Sum})$.
    # But we only need to check $y$ for $y \in S$.
    # The number of $y$ is $N$.
    # So we do $N$ checks.
    # Each check is `_is_mean_of_subset` on a set of size $N$.
    # This is effectively the same cost as a full check of the new set.
    
    # Can we do better?
    # The previous code did:
    # for y in selected:
    #    target = y - x
    #    vals = [z - y for z in selected if z != y]
    #    if _can_form_sum(target, vals): return False
    
    # This checks if $y - x$ can be formed by subset of $z - y$.
    # $\sum (z - y) = y - x \implies \sum z - k y = y - x \implies \sum z = (k+1)y - x$.
    # This corresponds to the case where $x$ is in the mean subset.
    # But the mean subset could also be entirely from $S \setminus \{y\}$ (not containing $x$).
    # If it doesn't contain $x$, then it was already a valid mean subset in $S$ (without $x$), which would have been invalid already.
    # So we only need to check subsets that *contain* $x$.
    # So the previous logic is correct!
    # We only need to check if $y$ is the mean of a subset of $(S \setminus \{y\}) \cup \{x\}$ that *contains* $x$.
    # If it doesn't contain $x$, it's a subset of $S \setminus \{y\}$, which was already checked when $S$ was formed.
    
    # So we stick with the previous logic for Check 2.
    
    for y in selected:
        target = y - x
        if target == 0:
            continue
        vals = [z - y for z in selected if z != y]
        if not vals:
            continue
        if _can_form_sum(target, vals):
            return False
            
    return True

def _can_form_sum(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals with size >= 1.
    """
    if not vals:
        return False
        
    n = len(vals)
    max_sum = sum(abs(v) for v in vals)
    if abs(target) > max_sum:
        return False
        
    offset = max_sum
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for v in vals:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if v >= 0:
                shifted = dp[k-1] << v
            else:
                shifted = dp[k-1] >> (-v)
            dp[k] |= shifted
            
    bit_idx = offset + target
    for k in range(1, n + 1):
        if dp[k] & (1 << bit_idx):
            return True
    return False

def construct():
    start_time = time.time()
    best_set = []
    
    candidates = list(range(1, 513))
    
    # Orders
    orders = []
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(hi)
        if lo != hi:
            spread.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread)
    orders.append(spread[::-1])
    
    rng = random.Random(42)
    for _ in range(50):
        r = candidates[:]
        rng.shuffle(r)
        orders.append(r)
        
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        sel_set = set()
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
            if len(selected) >= 48:
                break
            if x in sel_set:
                continue
                
            # Fast pair check
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if selected[i] + selected[j] == 2 * x:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for a in selected:
                    y = 2 * a - x
                    if y in sel_set and y != a:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            if _is_valid_incremental(selected, x):
                selected.append(x)
                sel_set.add(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    if best_set:
        # Final verification
        C = best_set[:]
        for i in range(len(C)):
            if _is_mean_of_subset(C, i):
                return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
