import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin from 90s limit

    def generate_sums(S):
        """Generate all unordered triple sums for set S."""
        sums = set()
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                for k in range(j, n):
                    c = S[k]
                    sums.add(a + b + c)
        return sums

    def can_add(x, S, sums):
        """Check if x can be added to S without creating duplicate triple sums."""
        # 1. x, x, x
        if 3 * x in sums:
            return False
        
        # 2. x, x, y
        for y in S:
            if 2 * x + y in sums:
                return False
        
        # 3. x, y, z
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i, n):
                z = S[j]
                if x + y + z in sums:
                    return False
        return True

    def add_element(x, S, sums):
        """Add x to S and update sums."""
        S.append(x)
        # Add 3x
        sums.add(3 * x)
        # Add 2x + y
        for y in S[:-1]:
            sums.add(2 * x + y)
        # Add x + y + z
        n_old = len(S) - 1
        for i in range(n_old):
            y = S[i]
            for j in range(i, n_old):
                z = S[j]
                sums.add(x + y + z)

    def greedy_insert(order, S_init, sums_init):
        """Try to insert elements from order into S_init."""
        S = list(S_init)
        sums = set(sums_init)
        S_set = set(S)
        
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            
            if can_add(x, S, sums):
                add_element(x, S, sums)
                S_set.add(x)
                
        return S

    def local_improve(S, sums):
        """Try to improve S by swapping out elements for better ones."""
        S = list(S)
        sums = set(sums)
        S_set = set(S)
        improved = True
        max_swaps = 20
        
        while improved and max_swaps > 0:
            improved = False
            max_swaps -= 1
            
            # Try removing each element and finding a replacement
            for i in range(len(S)):
                if time.time() > deadline:
                    return S, sums
                x_remove = S[i]
                
                # Temporarily remove
                S_new = S[:i] + S[i+1:]
                sums_new = set(sums)
                
                # Recompute sums without x_remove
                sums_new = generate_sums(S_new)
                
                # Try to find a better replacement
                best_replacement = None
                for candidate in range(1, 4097):
                    if candidate in S_set:
                        continue
                    if candidate == x_remove:
                        continue
                    
                    # Check if candidate can be added
                    if can_add(candidate, S_new, sums_new):
                        # Score: prefer candidates that create more "space"
                        S_test = S_new + [candidate]
                        S_test.sort()
                        # Simple heuristic: prefer larger gaps
                        gap_score = 0
                        for j in range(len(S_test)):
                            if j > 0:
                                gap_score += S_test[j] - S_test[j-1]
                        if best_replacement is None or gap_score > best_replacement[1]:
                            best_replacement = (candidate, gap_score)
                
                if best_replacement and best_replacement[0] not in S_set:
                    candidate = best_replacement[0]
                    S_new.append(candidate)
                    S_new.sort()
                    sums_final = generate_sums(S_new)
                    S = S_new
                    sums = sums_final
                    S_set = set(S)
                    improved = True
                    break
        
        return S, sums

    # 1. Create Base-4 Skeleton
    skeleton = [4**i for i in range(7) if 4**i <= 4096]
    # skeleton = [1, 4, 16, 64, 256, 1024, 4096]
    skeleton_sums = generate_sums(skeleton)
    
    best_set = skeleton
    best_size = len(skeleton)

    # 2. Try Ascending Insertion
    candidates_asc = [x for x in range(1, 4097) if x not in skeleton]
    if time.time() < deadline:
        res = greedy_insert(candidates_asc, skeleton, skeleton_sums)
        if len(res) > best_size:
            best_set = res
            best_size = len(res)

    # 3. Try Descending Insertion
    candidates_desc = sorted(candidates_asc, reverse=True)
    if time.time() < deadline:
        res = greedy_insert(candidates_desc, skeleton, skeleton_sums)
        if len(res) > best_size:
            best_set = res
            best_size = len(res)

    # 4. Try Gap-Based Ordering: prioritize values that maximize minimum distance to existing elements
    def gap_based_order(S_current):
        """Generate candidates ordered by how well they fit in gaps."""
        S_sorted = sorted(S_current)
        candidates = []
        for x in range(1, 4097):
            if x in S_current:
                continue
            # Compute minimum distance to any element in S_current
            min_dist = min(abs(x - y) for y in S_sorted) if S_sorted else 0
            candidates.append((min_dist, x))
        # Sort by minimum distance (descending) - prefer values far from existing
        candidates.sort(key=lambda t: -t[0])
        return [x for _, x in candidates]

    if time.time() < deadline:
        gap_order = gap_based_order(skeleton)
        res = greedy_insert(gap_order, skeleton, skeleton_sums)
        if len(res) > best_size:
            best_set = res
            best_size = len(res)

    # 5. Try Randomized Insertion with multiple strategies
    random.seed(42)
    for i in range(30):
        if time.time() > deadline - 5.0:
            break
        
        if i % 3 == 0:
            candidates_rand = candidates_asc[:]
            random.shuffle(candidates_rand)
        elif i % 3 == 1:
            candidates_rand = gap_based_order(skeleton)
            random.shuffle(candidates_rand[:len(candidates_rand)//2])
        else:
            # Mix of ascending and random
            candidates_rand = candidates_asc[:]
            random.shuffle(candidates_rand)
        
        res = greedy_insert(candidates_rand, skeleton, skeleton_sums)
        if len(res) > best_size:
            best_set = res
            best_size = len(res)

    # 6. Local improvement on best set found
    if time.time() < deadline - 10.0 and best_size > len(skeleton):
        best_sums = generate_sums(best_set)
        improved_set, improved_sums = local_improve(best_set, best_sums)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)

    # 7. Final polish: try to add more elements in a smart order
    if time.time() < deadline - 5.0:
        current_sums = generate_sums(best_set)
        current_set = set(best_set)
        
        # Try adding elements that maximize spacing
        remaining = [x for x in range(1, 4097) if x not in current_set]
        remaining.sort(key=lambda x: -min(abs(x - y) for y in best_set))
        
        for x in remaining:
            if time.time() > deadline - 2.0:
                break
            if can_add(x, best_set, current_sums):
                add_element(x, best_set, current_sums)
                current_set.add(x)
                best_size = len(best_set)

    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
