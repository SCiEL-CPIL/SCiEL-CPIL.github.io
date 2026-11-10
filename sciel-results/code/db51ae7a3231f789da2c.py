import time
import random

def construct():
    """
    Construct a 4-term-progression-free subset of [1, 512].
    Uses bitset-based local search to maximize cardinality.
    """
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s limit
    
    n = 512
    # Map 1-based indices to 0-based bits
    # Bit i represents number i+1
    
    # Precompute conflicts
    # conflicts[x] is a list of bitmasks. 
    # If x is added, and any mask M in conflicts[x] is fully contained in S, 
    # then a 4-AP is formed.
    # We want to check: does S contain the other 3 elements of any 4-AP with x?
    
    conflicts = [[] for _ in range(n)]
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d
    # For each AP, for each term x in the AP, add the mask of the other 3 terms to conflicts[x]
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            t1 = a
            t2 = a + d
            t3 = a + 2 * d
            t4 = a + 3 * d
            
            # Convert to 0-based indices for bit operations
            # Bit position for value v is v-1
            m1 = 1 << (t1 - 1)
            m2 = 1 << (t2 - 1)
            m3 = 1 << (t3 - 1)
            m4 = 1 << (t4 - 1)
            
            # For t1, others are t2, t3, t4
            conflicts[t1 - 1].append(m2 | m3 | m4)
            # For t2, others are t1, t3, t4
            conflicts[t2 - 1].append(m1 | m3 | m4)
            # For t3, others are t1, t2, t4
            conflicts[t3 - 1].append(m1 | m2 | m4)
            # For t4, others are t1, t2, t3
            conflicts[t4 - 1].append(m1 | m2 | m3)

    def can_add(S, x_idx):
        """
        Check if element at index x_idx (0-based) can be added to bitset S.
        """
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def greedy_init(seed):
        random.seed(seed)
        S = 0
        candidates = list(range(n))
        random.shuffle(candidates)
        for idx in candidates:
            if can_add(S, idx):
                S |= (1 << idx)
        return S

    def local_search(S, time_limit):
        """
        Perform local search on bitset S.
        """
        start_ls = time.time()
        improved = True
        # Keep track of best size
        best_S = S
        best_size = bin(S).count("1")
        
        # We will do a mix of "Add" and "Swap"
        # To make it faster, we can precompute a list of indices not in S?
        # That's dynamic. 
        
        while time.time() - start_ls < time_limit:
            # Phase 1: Add
            changed = False
            # Iterate through all indices
            # Optimization: Only check indices not in S
            # We can iterate 0..n-1 and check if bit is 0
            for idx in range(n):
                if (S & (1 << idx)) == 0:
                    if can_add(S, idx):
                        S |= (1 << idx)
                        changed = True
                        if bin(S).count("1") > best_size:
                            best_size = bin(S).count("1")
                            best_S = S
            
            if not changed:
                # Phase 2: Swap
                # Try removing a random element and refilling
                # To be efficient, we try a few random removals
                # We need a list of elements in S
                # Extracting list from bitset is O(n)
                # Let's just pick a random index and check if it's in S
                # Or maintain a list? Maintaining a list is error prone with swaps.
                # n=512 is small enough to extract list if needed, but let's avoid if possible.
                
                # Strategy: Pick 10 random indices. If one is in S, try swapping it out.
                swaps_tried = 0
                while swaps_tried < 10 and time.time() - start_ls < time_limit:
                    random_idx = random.randint(0, n - 1)
                    if (S & (1 << random_idx)) != 0:
                        # Try removing random_idx
                        S_temp = S & ~(1 << random_idx)
                        
                        # Now try to add anything
                        # We can just run the add phase on S_temp
                        # But we only accept if size increases
                        added = 0
                        for idx in range(n):
                            if (S_temp & (1 << idx)) == 0:
                                if can_add(S_temp, idx):
                                    S_temp |= (1 << idx)
                                    added += 1
                                    # We can break early if we know we can't beat current size?
                                    # Current size of S is K. S_temp starts at K-1.
                                    # If we add 2, we win.
                                    # If we add 1, we tie. We might still accept tie to escape local min?
                                    # Usually strict improvement is safer, but tie-breaking can help.
                                    # Let's require strict improvement for now to avoid oscillation.
                                    if added >= 2:
                                        break
                        
                        if added >= 1:
                            # If we added at least 1, the new size is (K-1)+added.
                            # If added > 1, size increased.
                            # If added == 1, size is same.
                            # Let's accept if size > best_size OR (size == best_size and random) 
                            # Actually, accepting ties can lead to cycles.
                            # Let's only accept if new_size > current_size (K)
                            # i.e., added > 1
                            if added > 1:
                                S = S_temp
                                changed = True
                                if bin(S).count("1") > best_size:
                                    best_size = bin(S).count("1")
                                    best_S = S
                    swaps_tried += 1
            
            if not changed:
                # Stuck in local minimum. 
                # Try a more aggressive random perturbation?
                # Or just break and rely on multiple restarts.
                # For now, break to save time for other restarts.
                break
        
        return best_S, best_size

    best_S = 0
    best_size = 0
    
    # Run multiple restarts
    # We estimate time per restart.
    # Greedy is fast. Local search is slower.
    
    num_restarts = 1000 # Cap, but will break on time
    for i in range(num_restarts):
        if time.time() - start_time > deadline:
            break
            
        seed = 42 + i
        current_S = greedy_init(seed)
        current_size = bin(current_S).count("1")
        
        if current_size > best_size:
            best_S = current_S
            best_size = current_size
            
        # Determine time for local search
        # We want to finish all restarts by deadline.
        # Let's allocate 50ms per restart on average?
        # 512 elements, local search might take ~10-50ms depending on depth.
        # Let's give it 20ms.
        ls_time = 0.02
        # If we have lots of time left, give more to the current one?
        # No, better to have many diverse restarts.
        
        searched_S, searched_size = local_search(current_S, ls_time)
        
        if searched_size > best_size:
            best_S = searched_S
            best_size = searched_size
            
        # If we found a very large set, we might want to spend more time polishing it
        # But for now, uniform is safer.

    # Final verification and conversion to list
    def is_4ap_free_bitset(S):
        # Verify no 4-AP
        # This is just a sanity check, can_add already ensures this during construction
        # But good to be sure.
        for a in range(n):
            if (S & (1 << a)) == 0:
                continue
            for d in range(1, (n - a) // 3 + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if e >= n:
                    break
                # Check if a, b, c, e are all in S
                if (S & (1 << a)) and (S & (1 << b)) and (S & (1 << c)) and (S & (1 << e)):
                    return False
        return True

    if not is_4ap_free_bitset(best_S):
        # Fallback to base-3 if something went wrong (shouldn't happen)
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
