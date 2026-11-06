import time
import sys
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin for 60s limit
    max_val = 4096

    best_set = []
    best_len = 0

    def compute_sums(S):
        """Compute all unordered triple sums for set S."""
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(S[i] + S[j] + S[k])
        return sums

    def can_add(S, used_sums, x):
        """Check if x can be added to S without creating duplicate triple sums."""
        k = len(S)
        # 1. x + x + x
        s1 = 3 * x
        if s1 in used_sums:
            return False
        # 2. x + x + s
        for s in S:
            if (2 * x + s) in used_sums:
                return False
        # 3. x + s_i + s_j
        for i in range(k):
            for j in range(i, k):
                if (x + S[i] + S[j]) in used_sums:
                    return False
        return True

    def add_to_used(used_sums, S, x):
        """Add all new triple sums involving x to used_sums."""
        used_sums.add(3 * x)
        for s in S:
            used_sums.add(2 * x + s)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                used_sums.add(x + S[i] + S[j])

    def remove_from_used(used_sums, S, x):
        """Remove all triple sums involving x from used_sums."""
        used_sums.discard(3 * x)
        for s in S:
            used_sums.discard(2 * x + s)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                used_sums.discard(x + S[i] + S[j])

    def generate_candidates(S, used_sums, S_set, num_random=30):
        """Generate a diverse set of candidate numbers to try adding."""
        if not S:
            return list(range(1, max_val + 1))
        
        S_sorted = sorted(S)
        candidates = set()
        
        # Small numbers below min
        min_val = S_sorted[0]
        for x in range(1, min_val):
            candidates.add(x)
        
        # Gaps between consecutive elements
        for a, b in zip(S_sorted[:-1], S_sorted[1:]):
            gap = b - a
            if gap > 1:
                if gap < 30:
                    for x in range(a + 1, b):
                        candidates.add(x)
                else:
                    # Sample from large gaps
                    for _ in range(min(10, gap - 1)):
                        candidates.add(random.randint(a + 1, b - 1))
        
        # Numbers above max
        max_s = S_sorted[-1]
        if max_s < max_val:
            above = max_val - max_s
            if above < 50:
                for x in range(max_s + 1, max_val + 1):
                    candidates.add(x)
            else:
                for _ in range(min(30, above)):
                    candidates.add(random.randint(max_s + 1, max_val))
        
        # Random samples for diversity
        for _ in range(num_random):
            r = random.randint(1, max_val)
            candidates.add(r)
        
        # Filter out already-in-set candidates
        candidates = [x for x in candidates if x not in S_set]
        return candidates

    # Phase 1: Greedy with different strategies
    def greedy(start_offset=0, skip_prob=0.0, max_candidates=None):
        S = []
        used_sums = set()
        for x in range(1 + start_offset, max_val + 1):
            if max_candidates is not None and len(S) >= max_candidates:
                break
            if skip_prob > 0 and random.random() < skip_prob:
                continue
            if can_add(S, used_sums, x):
                S.append(x)
                add_to_used(used_sums, S[:-1], x)
        return S

    # Standard greedy
    S1 = greedy()
    best_set = S1
    best_len = len(S1)

    # Try greedy with small random skips to explore different configurations
    for trial in range(30):
        if time.time() - start_time > DEADLINE:
            break
        S_trial = greedy(skip_prob=0.02 + 0.02 * trial)
        if len(S_trial) > best_len:
            best_set = S_trial
            best_len = len(S_trial)

    # Phase 2: Improved local search with smarter candidate ordering
    def local_search(S_init, max_iterations=300):
        S = list(S_init)
        used_sums = compute_sums(S)
        current_len = len(S)
        no_improve_count = 0
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to add a new element
            improved = False
            S_set = set(S)
            candidates = generate_candidates(S, used_sums, S_set, num_random=40)
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(S, used_sums, x):
                    S.append(x)
                    add_to_used(used_sums, S[:-1], x)
                    current_len += 1
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
                
                # Try swap: remove one element, then try to add a new one
                if len(S) > 1 and no_improve_count < 5:
                    # Try removing each element (prioritize smaller ones first as they're more likely to be "waste")
                    S_sorted = sorted(S)
                    # Try removing from smallest to largest
                    for idx_to_remove in range(len(S)):
                        elem_to_remove = S[idx_to_remove]
                        
                        remove_from_used(used_sums, S, elem_to_remove)
                        S.pop(idx_to_remove)
                        
                        S_set = set(S)
                        candidates = generate_candidates(S, used_sums, S_set, num_random=50)
                        random.shuffle(candidates)
                        
                        added = False
                        for x in candidates:
                            if can_add(S, used_sums, x):
                                S.append(x)
                                add_to_used(used_sums, S[:-1], x)
                                added = True
                                break
                        
                        if added:
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            # Restore
                            S.insert(idx_to_remove, elem_to_remove)
                            add_to_used(used_sums, S, elem_to_remove)
                    
                    if not improved:
                        # Try random removals
                        for _ in range(3):
                            if time.time() - start_time > DEADLINE:
                                break
                            idx_to_remove = random.randint(0, len(S) - 1)
                            elem_to_remove = S[idx_to_remove]
                            
                            remove_from_used(used_sums, S, elem_to_remove)
                            S.pop(idx_to_remove)
                            
                            S_set = set(S)
                            candidates = generate_candidates(S, used_sums, S_set, num_random=50)
                            random.shuffle(candidates)
                            
                            added = False
                            for x in candidates:
                                if can_add(S, used_sums, x):
                                    S.append(x)
                                    add_to_used(used_sums, S[:-1], x)
                                    added = True
                                    break
                            
                            if added:
                                improved = True
                                no_improve_count = 0
                                break
                            else:
                                # Restore
                                S.insert(idx_to_remove, elem_to_remove)
                                add_to_used(used_sums, S, elem_to_remove)
                
                # If we've tried many swaps without improvement, break
                if no_improve_count > 10:
                    break
        
        return S

    # Run local search from the best set found so far
    if time.time() - start_time < DEADLINE:
        S_local = local_search(best_set, max_iterations=200)
        if len(S_local) > best_len:
            best_set = S_local
            best_len = len(S_local)

    # Phase 3: More aggressive local search with multiple restarts
    for restart in range(20):
        if time.time() - start_time > DEADLINE:
            break
        
        S_perturbed = list(best_set)
        # Remove 2-4 random elements
        num_remove = min(2 + restart // 5, len(S_perturbed) // 3)
        for _ in range(num_remove):
            if len(S_perturbed) > 1:
                idx = random.randint(0, len(S_perturbed) - 1)
                S_perturbed.pop(idx)
        
        S_local2 = local_search(S_perturbed, max_iterations=150)
        if len(S_local2) > best_len:
            best_set = S_local2
            best_len = len(S_local2)

    # Phase 4: Try greedy from different starting points
    for offset in [0, 3, 7, 13, 21, 34, 55, 89]:
        if time.time() - start_time > DEADLINE:
            break
        S_off = greedy(start_offset=offset)
        if len(S_off) > best_len:
            best_set = S_off
            best_len = len(S_off)
        
        if time.time() - start_time < DEADLINE - 5:
            S_off_local = local_search(S_off, max_iterations=100)
            if len(S_off_local) > best_len:
                best_set = S_off_local
                best_len = len(S_off_local)

    # Phase 5: Final polish - try to insert elements one by one
    if time.time() - start_time < DEADLINE - 3:
        S_polish = list(best_set)
        used_sums = compute_sums(S_polish)
        S_set = set(S_polish)
        
        # Try all numbers in order, inserting if possible
        for x in range(1, max_val + 1):
            if time.time() - start_time > DEADLINE - 2:
                break
            if x in S_set:
                continue
            if can_add(S_polish, used_sums, x):
                S_polish.append(x)
                add_to_used(used_sums, S_polish[:-1], x)
                S_set.add(x)
        
        if len(S_polish) > best_len:
            best_set = S_polish
            best_len = len(S_polish)

    # Phase 6: Systematic gap-filling
    if time.time() - start_time < DEADLINE - 2:
        S_gap = list(best_set)
        used_sums = compute_sums(S_gap)
        S_set = set(S_gap)
        
        # Sort and try to fill gaps systematically
        for _ in range(3):  # Multiple passes
            S_sorted = sorted(S_gap)
            S_set = set(S_gap)
            changed = False
            
            # Try filling gaps between consecutive elements
            for i in range(len(S_sorted) - 1):
                if time.time() - start_time > DEADLINE - 1:
                    break
                a, b = S_sorted[i], S_sorted[i + 1]
                for x in range(a + 1, b):
                    if x in S_set:
                        continue
                    if can_add(S_gap, used_sums, x):
                        S_gap.append(x)
                        add_to_used(used_sums, S_gap[:-1], x)
                        S_set.add(x)
                        changed = True
                        S_sorted = sorted(S_gap)
                        break
            
            # Try adding small numbers
            for x in range(1, S_sorted[0] if S_sorted else 1):
                if time.time() - start_time > DEADLINE - 1:
                    break
                if x in S_set:
                    continue
                if can_add(S_gap, used_sums, x):
                    S_gap.append(x)
                    add_to_used(used_sums, S_gap[:-1], x)
                    S_set.add(x)
                    changed = True
                    S_sorted = sorted(S_gap)
            
            # Try adding large numbers
            if S_sorted and S_sorted[-1] < max_val:
                for x in range(S_sorted[-1] + 1, max_val + 1):
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    if x in S_set:
                        continue
                    if can_add(S_gap, used_sums, x):
                        S_gap.append(x)
                        add_to_used(used_sums, S_gap[:-1], x)
                        S_set.add(x)
                        changed = True
            
            if not changed:
                break
        
        if len(S_gap) > best_len:
            best_set = S_gap
            best_len = len(S_gap)

    # Final verification
    if time.time() - start_time < DEADLINE - 2:
        S_final = best_set
        n = len(S_final)
        sum_counts = {}
        valid = True
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = S_final[i] + S_final[j] + S_final[k]
                    if s in sum_counts:
                        valid = False
                        break
                    sum_counts[s] = 1
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Invalid - fall back to greedy
            S_greedy_final = greedy()
            best_set = S_greedy_final
            best_len = len(S_greedy_final)

    return best_set
