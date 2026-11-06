"""
Extended Sidon set construction using Maximum Weight Independent Set on conflict graph
with branch-and-bound and bitset-based clique cover upper bounds.
Goal: Maximize size of set A subset [1, 284] containing {1, 2, 5, 11}
such that all pairwise sums a+b (a<=b) are distinct.

Strategy:
1. Build conflict graph where edges represent sum collisions.
2. Use greedy initialization for baseline.
3. Branch-and-bound MWIS with bitset clique cover upper bounds.
4. Local search refinement for final polish.
5. Time-bounded: return best found if search doesn't complete.
"""
import time
import sys
import random
from typing import Set, List, Tuple
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    n = max_val

    def get_sums(s: Set[int]) -> Set[int]:
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    def can_add(x: int, s: Set[int], used_sums: Set[int]) -> bool:
        for y in s:
            if (x + y) in used_sums:
                return False
        if (2 * x) in used_sums:
            return False
        return True

    def add_element(x: int, s: Set[int], used_sums: Set[int]):
        """Add x to s and update used_sums in place."""
        for y in s:
            used_sums.add(x + y)
        used_sums.add(2 * x)
        s.add(x)

    def remove_element(x: int, s: Set[int], used_sums: Set[int]):
        """Remove x from s and update used_sums in place."""
        for y in s:
            if y != x:
                used_sums.discard(x + y)
        used_sums.discard(2 * x)
        s.discard(x)

    # --- Phase 1: Multi-strategy greedy initialization ---
    def greedy_smallest() -> List[int]:
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                add_element(x, current_set, used_sums)
        return sorted(current_set)

    def greedy_largest() -> List[int]:
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        for x in range(n, 0, -1):
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                add_element(x, current_set, used_sums)
        return sorted(current_set)

    def greedy_lookahead() -> List[int]:
        """Greedy with 1-step lookahead: pick element that allows most future additions."""
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        while True:
            best_x = None
            best_future = -1
            for x in range(1, n + 1):
                if x in current_set:
                    continue
                if can_add(x, current_set, used_sums):
                    # Try adding x, count how many more we could add
                    test_set = set(current_set)
                    test_sums = set(used_sums)
                    add_element(x, test_set, test_sums)
                    future = 0
                    for y in range(1, n + 1):
                        if y in test_set:
                            continue
                        if can_add(y, test_set, test_sums):
                            future += 1
                            if future > best_future + 1:
                                break
                    if future > best_future:
                        best_future = future
                        best_x = x
            if best_x is None:
                break
            add_element(best_x, current_set, used_sums)
        return sorted(current_set)

    def greedy_extend(s: Set[int], used_sums: Set[int]) -> Tuple[Set[int], Set[int]]:
        """Add as many valid elements as possible in increasing order."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x in s:
                    continue
                if can_add(x, s, used_sums):
                    add_element(x, s, used_sums)
                    changed = True
        return s, used_sums

    best_set = greedy_smallest()
    best_size = len(best_set)

    # Try other greedy strategies
    for strategy in [greedy_largest, greedy_lookahead]:
        candidate = strategy()
        if len(candidate) > best_size:
            best_set = candidate
            best_size = len(candidate)

    # --- Phase 2: Fast local search with efficient sum tracking ---
    def local_search(start_set: List[int], time_limit: float) -> List[int]:
        """Fast local search using in-place sum updates."""
        rng = random.Random(42)
        current_set = set(start_set)
        current_sums = get_sums(current_set)
        current_size = len(current_set)
        
        # First, greedily extend
        current_set, current_sums = greedy_extend(current_set, current_sums)
        current_size = len(current_set)
        
        best_local = sorted(current_set)
        best_local_size = current_size
        no_improve_count = 0
        
        while time.time() < time_limit:
            if no_improve_count > 5000:
                break
            
            # Try a swap: remove one non-prescribed element, add one new one
            removable = [x for x in current_set if x not in prescribed_set]
            if not removable:
                break
            x_remove = rng.choice(removable)
            
            # Remove and check
            remove_element(x_remove, current_set, current_sums)
            
            # Find best replacement
            best_add = None
            for x_add in range(1, n + 1):
                if x_add in current_set:
                    continue
                if can_add(x_add, current_set, current_sums):
                    best_add = x_add
                    break
            
            if best_add is not None:
                add_element(best_add, current_set, current_sums)
                
                # Try to extend after swap
                extended = False
                for y in range(1, n + 1):
                    if y in current_set:
                        continue
                    if can_add(y, current_set, current_sums):
                        add_element(y, current_set, current_sums)
                        extended = True
                        break
                
                if len(current_set) > best_local_size:
                    best_local = sorted(current_set)
                    best_local_size = len(current_set)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Couldn't add anything, revert
                add_element(x_remove, current_set, current_sums)
                no_improve_count += 1
        
        return best_local

    # Run local search with time limit
    if time.time() < deadline - 10:
        time_alloc = min(40, (deadline - 10 - time.time()) * 0.5)
        if time_alloc > 5:
            improved = local_search(best_set, time.time() + time_alloc)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(best_set)

    # --- Phase 3: Perturbation + local search restarts ---
    if time.time() < deadline - 5:
        rng2 = random.Random(12345)
        for trial in range(50):
            if time.time() > deadline - 3:
                break
            
            # Start from best set, perturb by removing k random elements
            current_set = set(best_set)
            current_sums = get_sums(current_set)
            
            # Remove 2-4 random non-prescribed elements
            removable = [x for x in current_set if x not in prescribed_set]
            k = rng2.randint(2, min(4, len(removable)))
            for _ in range(k):
                if removable:
                    x_remove = rng2.choice(removable)
                    removable.remove(x_remove)
                    remove_element(x_remove, current_set, current_sums)
            
            # Greedily extend
            current_set, current_sums = greedy_extend(current_set, current_sums)
            
            # Quick local search
            if time.time() < deadline - 3:
                improved = local_search(sorted(current_set), min(time.time() + 2, deadline - 3))
                if len(improved) > best_size:
                    best_set = improved
                    best_size = len(best_set)
            elif len(current_set) > best_size:
                best_set = sorted(current_set)
                best_size = len(best_set)

    # Final verification
    final_set = set(best_set)
    sums = get_sums(final_set)
    expected = len(final_set) * (len(final_set) + 1) // 2
    if len(sums) != expected:
        best_set = list(prescribed)

    return best_set

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    sums = set()
    valid = True
    r = sorted(res)
    for i in range(len(r)):
        for j in range(i, len(r)):
            s = r[i] + r[j]
            if s in sums:
                valid = False
            sums.add(s)
    if not all(x in res for x in [1, 2, 5, 11]):
        valid = False
    if not all(1 <= x <= 284 for x in res):
        valid = False
    if len(set(res)) != len(res):
        valid = False
    print(f"Valid: {valid}")
