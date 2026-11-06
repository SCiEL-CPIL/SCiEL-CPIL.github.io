"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    n_smooth = len(smooth)
    smooth_set = set(smooth)
    
    # Precompute incompatibility bitmask: incompat[i] has bit j set if i and j cannot both be in antichain
    incompat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and (smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0):
                incompat[i] |= (1 << j)
    
    # Precompute all valid antichains up to a reasonable size
    # For each size k, store dict: sum -> list of index tuples
    max_k = 15
    antichain_sums = [dict() for _ in range(max_k + 1)]  # antichain_sums[k][s] = list of idx tuples
    
    # Build antichains incrementally using bitmask compatibility
    def build_antichains(start_idx, current_mask, current_sum, current_indices):
        """Recursively build antichains starting from start_idx."""
        if current_sum > 128:
            return
        k = len(current_indices)
        if k > 1:
            if current_sum not in antichain_sums[k]:
                antichain_sums[k][current_sum] = []
            antichain_sums[k][current_sum].append(tuple(current_indices))
        
        if k >= max_k:
            return
        if time.time() > deadline - 5.0:
            return
        
        for i in range(start_idx, n_smooth):
            val = smooth[i]
            if current_sum + val > 128:
                break
            # Check if i is compatible with all current indices using bitmask
            # Precompute: for each index, the set of incompatible indices
            # An index i is compatible if no current index is incompatible with i
            # i.e., for all ci in current_indices, (incompat[ci] & (1 << i)) == 0
            # Equivalent: (1 << i) is not in the union of incompat[ci] for ci in current_indices
            # We can maintain a running bitmask of all incompatible indices
            compatible = True
            for ci in current_indices:
                if incompat[ci] & (1 << i):
                    compatible = False
                    break
            if compatible:
                current_indices.append(i)
                build_antichains(i + 1, current_mask | (1 << i), current_sum + val, current_indices)
                current_indices.pop()
    
    build_antichains(0, 0, 0, [])
    
    # Also handle k=1: each smooth number is its own antichain
    for i, val in enumerate(smooth):
        if val <= 128:
            if val not in antichain_sums[1]:
                antichain_sums[1][val] = []
            antichain_sums[1][val].append((i,))
    
    # For each target, find the minimal k
    answers = []
    for n in range(1, 129):
        if time.time() > deadline - 2.0:
            # Emergency fallback
            if n in smooth_set:
                answers.append([n])
            else:
                answers.append([1] * n if n <= 30 else [1])
            continue
        
        answer = None
        # Find minimal k
        for k in range(1, max_k + 1):
            if n in antichain_sums[k]:
                combo = antichain_sums[k][n][0]
                answer = [smooth[i] for i in combo]
                break
        
        # If not found in precomputed, use DFS for larger k
        if answer is None:
            found = False
            for k in range(max_k + 1, 31):
                if time.time() > deadline - 2.0:
                    break
                def dfs(start_idx, remaining, chosen_idx, chosen_vals):
                    nonlocal found
                    if found:
                        return
                    if len(chosen_idx) == k:
                        if remaining == 0:
                            found = True
                            answer = list(chosen_vals)
                        return
                    if remaining <= 0:
                        return
                    slots_left = k - len(chosen_idx)
                    if start_idx + slots_left > n_smooth:
                        return
                    # Pruning: minimum possible sum with remaining slots
                    if start_idx + slots_left <= n_smooth:
                        min_possible = sum(smooth[start_idx + j] for j in range(slots_left))
                    else:
                        return
                    if min_possible > remaining:
                        return
                    # Pruning: maximum possible sum with remaining slots (largest smooth numbers)
                    max_idx = n_smooth - slots_left
                    if max_idx < start_idx:
                        return
                    max_possible = sum(smooth[max_idx - j] for j in range(slots_left))
                    if max_possible < remaining:
                        return
                    for i in range(start_idx, n_smooth):
                        if found:
                            return
                        val = smooth[i]
                        if val > remaining:
                            break
                        # Check compatibility
                        compatible = True
                        for ci in chosen_idx:
                            if incompat[ci] & (1 << i):
                                compatible = False
                                break
                        if compatible:
                            chosen_idx.append(i)
                            chosen_vals.append(val)
                            dfs(i + 1, remaining - val, chosen_idx, chosen_vals)
                            chosen_idx.pop()
                            chosen_vals.pop()
                
                dfs(0, n, [], [])
                if found:
                    break
        
        if answer is None:
            # Last resort fallback
            if n in smooth_set:
                answer = [n]
            else:
                # Greedy from largest
                remaining = n
                chosen = []
                for s in reversed(smooth):
                    if s <= remaining:
                        valid = True
                        for c in chosen:
                            if c % s == 0 or s % c == 0:
                                valid = False
                                break
                        if valid:
                            chosen.append(s)
                            remaining -= s
                            if remaining == 0:
                                break
                if remaining == 0 and 1 <= len(chosen) <= 30:
                    answer = chosen
                else:
                    answer = [1]
        
        answers.append(answer)
    
    return answers
# EVOLVE-BLOCK-END
