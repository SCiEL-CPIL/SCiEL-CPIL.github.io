import random
import time
import sys

def construct():
    # Deadline guard: 150s limit, aim for 110s work + safety margin
    start_time = time.time()
    time_limit = 110.0
    
    # Helper to check if adding x to set S is valid
    def try_add(x, S, counts):
        for y in S:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        s = 2 * x
        if counts.get(s, 0) >= 2:
            return False
        return True

    def get_new_counts(x, S, counts):
        new_counts = dict(counts)
        for y in S:
            s = x + y
            new_counts[s] = new_counts.get(s, 0) + 1
        s = 2 * x
        new_counts[s] = new_counts.get(s, 0) + 1
        return new_counts

    def remove_from_counts(y, S, counts):
        """Remove element y from counts (S still contains y)"""
        new_counts = dict(counts)
        for z in S:
            if z == y:
                continue
            s = y + z
            new_counts[s] = new_counts.get(s, 0) - 1
            if new_counts[s] <= 0:
                del new_counts[s]
        s = 2 * y
        new_counts[s] = new_counts.get(s, 0) - 1
        if new_counts[s] <= 0:
            del new_counts[s]
        return new_counts

    def greedy_build(order):
        """Build a B2[2] set using a given ordering of candidates"""
        S = set()
        counts = {}
        for x in order:
            if try_add(x, S, counts):
                S.add(x)
                counts = get_new_counts(x, S, counts)
        return S, counts

    # Initial greedy construction with multiple strategies
    best_set = set()
    best_counts = {}
    best_size = 0
    
    # Strategy 1: Standard ascending greedy
    S, counts = greedy_build(range(1, 1001))
    if len(S) > best_size:
        best_set = S
        best_counts = counts
        best_size = len(S)

    # Strategy 2: Descending greedy
    S, counts = greedy_build(range(1000, 0, -1))
    if len(S) > best_size:
        best_set = S
        best_counts = counts
        best_size = len(S)

    # Strategy 3: Random permutations
    for _ in range(15):
        if time.time() - start_time > time_limit * 0.15:
            break
        perm = list(range(1, 1001))
        random.shuffle(perm)
        S, counts = greedy_build(perm)
        if len(S) > best_size:
            best_set = S
            best_counts = counts
            best_size = len(S)

    # Local Search with conflict-aware swaps
    current_set = set(best_set)
    current_counts = dict(best_counts)
    current_size = len(current_set)
    candidates = list(set(range(1, 1001)) - current_set)
    
    def get_addable(S, counts, candidates_list):
        addable = []
        for x in candidates_list:
            if try_add(x, S, counts):
                addable.append(x)
        return addable

    def get_conflicting_sums(S, counts):
        """Get sums that are at capacity (count == 2)"""
        return [s for s, c in counts.items() if c == 2]

    def find_blocking_elements(x, S, counts):
        """Find elements in S that block x from being added"""
        blockers = []
        for y in S:
            s = x + y
            if counts.get(s, 0) >= 2:
                blockers.append(y)
        s = 2 * x
        if counts.get(s, 0) >= 2:
            # 2*x is at capacity; find who contributes to it
            for y in S:
                if y + y == s:  # only y=x but x not in S
                    pass
                # Actually 2*x can only be formed by x+x, but x not in S
                # So this shouldn't happen if x not in S
                break
        return blockers

    # Main local search loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        # Try to add any available element
        addable = get_addable(current_set, current_counts, candidates)
        
        if addable:
            # Pick the one that opens most future possibilities
            # Simple heuristic: pick randomly from addable
            x = random.choice(addable)
            current_set.add(x)
            current_counts = get_new_counts(x, current_set, current_counts)
            current_size += 1
            candidates.remove(x)
            no_improve_count = 0
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
                best_counts = dict(current_counts)
        else:
            # No direct addition possible. Try targeted swaps.
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a more aggressive restart: remove 2 random elements and rebuild
                if len(current_set) >= 2 and candidates:
                    # Remove two random elements
                    remove1 = random.choice(list(current_set))
                    current_set.remove(remove1)
                    current_counts = remove_from_counts(remove1, current_set | {remove1}, current_counts)
                    
                    if current_set:
                        remove2 = random.choice(list(current_set))
                        current_set.remove(remove2)
                        current_counts = remove_from_counts(remove2, current_set | {remove2}, current_counts)
                    
                    candidates.extend([remove1, remove2])
                    candidates = list(set(candidates))
                    current_size = len(current_set)
                    no_improve_count = 0
                    max_no_improve = 30  # Reset threshold
                else:
                    break
            else:
                # Try a targeted swap
                if candidates and current_set:
                    # Pick a random candidate and find what blocks it
                    x = random.choice(candidates)
                    blockers = find_blocking_elements(x, current_set, current_counts)
                    
                    if blockers:
                        # Try removing one of the blockers
                        y = random.choice(blockers)
                        
                        # Remove y
                        temp_counts = remove_from_counts(y, current_set, current_counts)
                        temp_set = set(current_set)
                        temp_set.remove(y)
                        
                        # Check if x can now be added
                        if try_add(x, temp_set, temp_counts):
                            # Accept the swap
                            temp_set.add(x)
                            final_counts = get_new_counts(x, temp_set, temp_counts)
                            
                            current_set = temp_set
                            current_counts = final_counts
                            candidates.remove(x)
                            candidates.append(y)
                            candidates = list(set(candidates))
                            # Size stays same, but we may have opened up new additions
                        else:
                            # Swap didn't help, try another blocker or candidate
                            pass
                    else:
                        # x is not blocked by specific elements (shouldn't happen if not addable)
                        # Try a random swap as fallback
                        y = random.choice(list(current_set))
                        temp_counts = remove_from_counts(y, current_set, current_counts)
                        temp_set = set(current_set)
                        temp_set.remove(y)
                        
                        if try_add(x, temp_set, temp_counts):
                            temp_set.add(x)
                            final_counts = get_new_counts(x, temp_set, temp_counts)
                            current_set = temp_set
                            current_counts = final_counts
                            candidates.remove(x)
                            candidates.append(y)
                            candidates = list(set(candidates))
                else:
                    break

    # Final polish: try to add any remaining candidates
    addable = get_addable(current_set, current_counts, candidates)
    for x in addable:
        if time.time() - start_time > time_limit - 5:
            break
        current_set.add(x)
        current_counts = get_new_counts(x, current_set, current_counts)
        current_size += 1
        candidates.remove(x)
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
            best_counts = dict(current_counts)

    return sorted(best_set)

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
