import time
import random

def construct():
    """
    Construct a 4-term-progression-free subset of [1, 512].
    Uses bitset-based local search with improved swap strategies and polish pass.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    n = 512
    
    # Precompute conflicts: for each element, store masks of other 3 elements in each 4-AP containing it
    conflicts = [[] for _ in range(n)]
    
    # Generate all 4-APs and build conflict lists
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            t1 = a
            t2 = a + d
            t3 = a + 2 * d
            t4 = a + 3 * d
            
            m1 = 1 << (t1 - 1)
            m2 = 1 << (t2 - 1)
            m3 = 1 << (t3 - 1)
            m4 = 1 << (t4 - 1)
            
            conflicts[t1 - 1].append(m2 | m3 | m4)
            conflicts[t2 - 1].append(m1 | m3 | m4)
            conflicts[t3 - 1].append(m1 | m2 | m4)
            conflicts[t4 - 1].append(m1 | m2 | m3)

    def can_add(S, x_idx):
        """Check if element at index x_idx can be added to bitset S without forming a 4-AP."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def count_bits(S):
        """Count set bits in S."""
        return bin(S).count("1")

    def greedy_init(seed):
        """Greedy initialization with random ordering."""
        random.seed(seed)
        S = 0
        candidates = list(range(n))
        random.shuffle(candidates)
        for idx in candidates:
            if can_add(S, idx):
                S |= (1 << idx)
        return S

    def try_add_all(S):
        """Try to add all possible elements to S. Returns (new_S, added_count)."""
        added = 0
        # Build list of candidates not in S once
        for idx in range(n):
            if (S & (1 << idx)) == 0:
                if can_add(S, idx):
                    S |= (1 << idx)
                    added += 1
        return S, added

    def local_search(S, time_limit):
        """
        Perform local search on bitset S with improved swap strategy.
        """
        start_ls = time.time()
        best_S = S
        best_size = count_bits(S)
        
        while time.time() - start_ls < time_limit:
            # Phase 1: Try to add all possible elements
            S, added = try_add_all(S)
            if added > 0:
                current_size = count_bits(S)
                if current_size > best_size:
                    best_size = current_size
                    best_S = S
                continue  # Loop back to try more additions
            
            # Phase 2: Systematic swap - try removing elements and see if we can add more
            current_size = count_bits(S)
            improved = False
            
            # Try removing elements in order of their "cost" (number of conflicts)
            elements_in_S = [i for i in range(n) if (S & (1 << i)) != 0]
            
            # Sort by number of conflicts (fewer conflicts = cheaper to remove)
            elements_in_S.sort(key=lambda i: len(conflicts[i]))
            
            for remove_idx in elements_in_S:
                if time.time() - start_ls >= time_limit:
                    break
                
                # Try removing this element
                S_temp = S & ~(1 << remove_idx)
                
                # Try to add all possible elements
                S_new, added = try_add_all(S_temp)
                
                new_size = count_bits(S_new)
                
                # Accept if we improve
                if new_size > current_size:
                    S = S_new
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S
                    improved = True
                    break  # Restart the loop with the new S
            
            if not improved:
                # Stuck in local minimum - try multi-element removals
                # First try removing 2 elements
                for _ in range(5):
                    if time.time() - start_ls >= time_limit:
                        break
                    
                    # Pick 2 random distinct elements from S
                    temp = S
                    elems = []
                    while temp and len(elems) < 2:
                        if temp & 1:
                            elems.append(0)
                        temp >>= 1
                        # Need to track actual indices
                    # Better approach: collect indices
                    elems = [i for i in range(n) if (S & (1 << i)) != 0]
                    if len(elems) < 2:
                        break
                    r1, r2 = random.sample(elems, 2)
                    
                    S_temp = S & ~(1 << r1) & ~(1 << r2)
                    S_new, added = try_add_all(S_temp)
                    new_size = count_bits(S_new)
                    
                    if new_size > current_size:
                        S = S_new
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_S = S
                        improved = True
                        break
            
            if not improved:
                # Try removing 3 elements
                for _ in range(5):
                    if time.time() - start_ls >= time_limit:
                        break
                    
                    elems = [i for i in range(n) if (S & (1 << i)) != 0]
                    if len(elems) < 3:
                        break
                    r1, r2, r3 = random.sample(elems, 3)
                    
                    S_temp = S & ~(1 << r1) & ~(1 << r2) & ~(1 << r3)
                    S_new, added = try_add_all(S_temp)
                    new_size = count_bits(S_new)
                    
                    if new_size > current_size:
                        S = S_new
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_S = S
                        improved = True
                        break
            
            if not improved:
                # Final fallback: remove 1 random element
                for _ in range(3):
                    if time.time() - start_ls >= time_limit:
                        break
                    
                    elems = [i for i in range(n) if (S & (1 << i)) != 0]
                    if not elems:
                        break
                    r1 = random.choice(elems)
                    
                    S_temp = S & ~(1 << r1)
                    S_new, added = try_add_all(S_temp)
                    new_size = count_bits(S_new)
                    
                    if new_size > current_size:
                        S = S_new
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_S = S
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_S, best_size

    def polish_pass(S, time_limit):
        """
        High-precision polish pass: try all single-element swaps, double-element swaps,
        and triple-element swaps.
        """
        start_polish = time.time()
        best_S = S
        best_size = count_bits(S)
        
        while time.time() - start_polish < time_limit:
            improved = False
            current_size = count_bits(best_S)
            
            # Try all single removals
            for remove_idx in range(n):
                if time.time() - start_polish >= time_limit:
                    break
                
                if (best_S & (1 << remove_idx)) == 0:
                    continue
                
                S_temp = best_S & ~(1 << remove_idx)
                S_new, added = try_add_all(S_temp)
                new_size = count_bits(S_new)
                
                if new_size > current_size:
                    best_S = S_new
                    current_size = new_size
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try all double removals
            elements_in_best = [i for i in range(n) if (best_S & (1 << i)) != 0]
            
            for i1 in range(len(elements_in_best)):
                if time.time() - start_polish >= time_limit:
                    break
                
                for i2 in range(i1 + 1, len(elements_in_best)):
                    if time.time() - start_polish >= time_limit:
                        break
                    
                    r1 = elements_in_best[i1]
                    r2 = elements_in_best[i2]
                    
                    S_temp = best_S & ~(1 << r1) & ~(1 << r2)
                    S_new, added = try_add_all(S_temp)
                    new_size = count_bits(S_new)
                    
                    if new_size > current_size:
                        best_S = S_new
                        current_size = new_size
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try triple removals on a subset of elements (limit to keep runtime manageable)
                # Focus on elements with fewer conflicts first
                elements_sorted = sorted(elements_in_best, key=lambda i: len(conflicts[i]))
                # Only try triples among the first 30 lowest-conflict elements
                triple_candidates = elements_sorted[:min(30, len(elements_sorted))]
                
                for i1 in range(len(triple_candidates)):
                    if time.time() - start_polish >= time_limit:
                        break
                    
                    for i2 in range(i1 + 1, len(triple_candidates)):
                        if time.time() - start_polish >= time_limit:
                            break
                        
                        for i3 in range(i2 + 1, len(triple_candidates)):
                            if time.time() - start_polish >= time_limit:
                                break
                            
                            r1 = triple_candidates[i1]
                            r2 = triple_candidates[i2]
                            r3 = triple_candidates[i3]
                            
                            S_temp = best_S & ~(1 << r1) & ~(1 << r2) & ~(1 << r3)
                            S_new, added = try_add_all(S_temp)
                            new_size = count_bits(S_new)
                            
                            if new_size > current_size:
                                best_S = S_new
                                current_size = new_size
                                improved = True
                                break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_S, best_size

    best_S = 0
    best_size = 0
    
    # Phase 1: Multiple restarts with local search
    num_restarts = 300  # Cap, but will break on time
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 6.0:  # Leave 6s for polish
            break
            
        seed = 42 + i
        current_S = greedy_init(seed)
        current_size = count_bits(current_S)
        
        if current_size > best_size:
            best_S = current_S
            best_size = current_size
        
        # Allocate time for local search
        remaining = deadline - 6.0 - (time.time() - start_time)
        if remaining <= 0:
            break
        
        # Give more time to local search for better candidates
        ls_time = min(0.08, remaining / max(1, num_restarts - i))
        
        searched_S, searched_size = local_search(current_S, ls_time)
        
        if searched_size > best_size:
            best_S = searched_S
            best_size = searched_size
    
    # Phase 2: Polish pass on the best solution
    if time.time() - start_time < deadline - 1.0:
        polish_time = min(5.0, deadline - 1.0 - (time.time() - start_time))
        best_S, best_size = polish_pass(best_S, polish_time)
    
    # Final verification
    def is_4ap_free_bitset(S):
        for a in range(n):
            if (S & (1 << a)) == 0:
                continue
            for d in range(1, (n - a) // 3 + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if e >= n:
                    break
                if (S & (1 << a)) and (S & (1 << b)) and (S & (1 << c)) and (S & (1 << e)):
                    return False
        return True

    if not is_4ap_free_bitset(best_S):
        # Fallback to base-3 construction
        result = set()
        for x in range(1, n + 1):
            temp = x
            valid = True
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                result.add(x)
        return sorted(result)

    # Convert bitset to sorted list of 1-based integers
    result = []
    for i in range(n):
        if (best_S & (1 << i)) != 0:
            result.append(i + 1)
            
    return sorted(result)

if __name__ == "__main__":
    s = construct()
    print(f"Size: {len(s)}")
