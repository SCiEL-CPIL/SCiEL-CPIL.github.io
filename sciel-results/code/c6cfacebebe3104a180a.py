import time
import sys

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin

    # Step 1: Generate smooth numbers <= 128
    smooth = []
    smooth_set = set()
    for a in range(8):      # 2^7 = 128
        for b in range(6):  # 3^5 = 243 > 128, so 3^5 is too big if combined with anything >1, but 3^4=81, 3^5=243. 
                            # Actually 3^0..3^4 are 1,3,9,27,81. 3^5=243 > 128.
                            # So b goes 0..4.
                            # But 2^0 3^0 5^0 = 1.
                            # Max 2^a: 2^6=64, 2^7=128.
                            # Max 3^b: 3^4=81, 3^5=243.
                            # Max 5^c: 5^2=25, 5^3=125, 5^4=625.
                            # Let's just iterate and filter.
            for c in range(5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= v <= 128 and v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    smooth_idx = {v: i for i, v in enumerate(smooth)}

    # Step 2: Precompute divisibility conflicts
    # conflict[i] is a bitmask of indices j such that smooth[i] divides smooth[j] or vice versa?
    # We need "no chosen term may divide another".
    # So if i and j are in the set, smooth[i] must not divide smooth[j] AND smooth[j] must not divide smooth[i].
    # Let's precompute a matrix or bitmask for "incompatible" pairs.
    # incompatible[i] has bit j set if smooth[i] divides smooth[j] OR smooth[j] divides smooth[i].
    incompatible = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                    incompatible[i] |= (1 << j)

    # best_for_sum[s] = (count, tuple_of_indices)
    best_for_sum = {}
    targets = set(range(1, 129))
    
    # We will generate antichains of increasing size.
    # Since the max target is 128, the max size is limited.
    # We can use a recursive generator that yields antichains.
    # To ensure we find minimal sizes, we iterate by size k.
    
    # Optimization: Precompute valid combinations?
    # Let's use a backtracking search for each size k.
    
    def find_antichains_of_size(k, start_idx, current_indices, current_sum):
        """
        Recursively find antichains of size k.
        Yields (tuple_of_indices, sum).
        """
        if len(current_indices) == k:
            yield tuple(current_indices), current_sum
            return
        
        if current_sum > 128:
            return
            
        # Pruning: if not enough elements left to pick
        remaining_needed = k - len(current_indices)
        if n_smooth - start_idx < remaining_needed:
            return
            
        for i in range(start_idx, n_smooth):
            # Check if i is compatible with all in current_indices
            ok = True
            for j in current_indices:
                if incompatible[j] & (1 << i):
                    ok = False
                    break
            if not ok:
                continue
                
            current_indices.append(i)
            new_sum = current_sum + smooth[i]
            # We can prune if new_sum > 128, but we might still need to yield?
            # No, if sum > 128, it's useless for our targets.
            if new_sum <= 128:
                yield from find_antichains_of_size(k, i + 1, current_indices, new_sum)
            else:
                # Even if current sum is too high, adding more positive numbers will keep it high.
                # But wait, we are adding sorted numbers. So if new_sum > 128, any further i will be larger.
                # So we can break.
                # However, we must pop i before breaking.
                pass 
            current_indices.pop()

    # Iterate sizes
    max_size = 10 # Safety upper bound, unlikely to need more
    for k in range(1, max_size + 1):
        if time.time() - t0 > DEADLINE:
            break
            
        found_any = False
        for indices, s in find_antichains_of_size(k, 0, [], 0):
            if s in targets:
                if s not in best_for_sum:
                    best_for_sum[s] = (k, indices)
                    targets.remove(s)
                    found_any = True
                    if not targets:
                        break
        if not targets:
            break
        if not found_any:
            # If no new targets were covered by size k, maybe we need larger k?
            # But if we didn't find ANY new targets, it doesn't mean larger k won't work.
            # However, if we've exhausted all antichains of size k, we move to k+1.
            pass

    # If some targets are still missing, we need a fallback or deeper search.
    # Given the structure, it's very likely all are covered by k <= 6.
    # But to be safe, let's run a specific DFS for remaining targets.
    
    if targets:
        for target in list(targets):
            if time.time() - t0 > DEADLINE:
                break
                
            # Try sizes from 7 upwards
            found = False
            for k in range(7, 15):
                # Simple DFS for this specific target
                best = None
                selected = []
                
                def dfs(start_idx, remaining, size_left):
                    nonlocal best
                    if best is not None:
                        return
                    if size_left == 0:
                        if remaining == 0:
                            best = list(selected)
                        return
                    if remaining <= 0:
                        return
                    if n_smooth - start_idx < size_left:
                        return
                    
                    # Pruning: min possible sum
                    min_sum = 0
                    for x in range(size_left):
                        if start_idx + x >= n_smooth:
                            return
                        min_sum += smooth[start_idx + x]
                    if min_sum > remaining:
                        return
                        
                    for idx in range(start_idx, n_smooth):
                        val = smooth[idx]
                        if val > remaining:
                            break
                        ok = True
                        for s_idx in selected:
                            if incompatible[s_idx] & (1 << idx):
                                ok = False
                                break
                        if not ok:
                            continue
                        selected.append(idx)
                        dfs(idx + 1, remaining - val, size_left - 1)
                        selected.pop()
                        if best is not None:
                            return

                dfs(0, target, k)
                if best is not None:
                    best_for_sum[target] = (k, tuple(best))
                    targets.remove(target)
                    found = True
                    break
            if not targets:
                break

    # Build answers
    answers = []
    for n in range(1, 129):
        if n in best_for_sum:
            count, indices = best_for_sum[n]
            row = [smooth[i] for i in indices]
            row.sort()
            answers.append(row)
        else:
            # Fallback: should not happen, but provide a valid representation if possible
            # Find any antichain that sums to n, even if not minimal.
            # Or just return [n] if n is smooth, else [1, n-1] if valid?
            # [1, n-1] is not always an antichain (1 divides everything).
            # So we must find a valid one.
            # Since the previous step should have covered all, this is just a safety net.
            # We'll do a quick greedy or DFS.
            found = None
            for k in range(1, 15):
                # Re-use the DFS logic from above but without the 'best' optimization, just find one
                selected = []
                def dfs2(start_idx, remaining, size_left):
                    nonlocal found
                    if found is not None:
                        return
                    if size_left == 0:
                        if remaining == 0:
                            found = list(selected)
                        return
                    if remaining <= 0:
                        return
                    if n_smooth - start_idx < size_left:
                        return
                    for idx in range(start_idx, n_smooth):
                        val = smooth[idx]
                        if val > remaining:
                            break
                        ok = True
                        for s_idx in selected:
                            if incompatible[s_idx] & (1 << idx):
                                ok = False
                                break
                        if not ok:
                            continue
                        selected.append(idx)
                        dfs2(idx + 1, remaining - val, size_left - 1)
                        selected.pop()
                        if found is not None:
                            return
                dfs2(0, n, k)
                if found is not None:
                    break
            if found:
                row = [smooth[i] for i in found]
                row.sort()
                answers.append(row)
            else:
                # Absolute fallback, might be invalid but prevents crash
                answers.append([1]) 
                # Note: This will fail validation if 1 doesn't sum to n.
                # But this case should never be reached.

    return answers
