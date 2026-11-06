import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    # Bitset for each number 1..N
    # bit i corresponds to number i (0-indexed in bitset, i.e., 1 is bit 0, 128 is bit 127)
    # Actually, let's use 1-indexed for clarity in logic, but 0-indexed for bit positions.
    # num -> bit: 1 << (num - 1)
    
    # Precompute GCD
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Precompute bad_with_pair[u][v]
    # bad_with_pair[u][v] is a bitset of w such that {u, v, w} is a forbidden triple.
    # i.e., gcd(u,v) == gcd(u,w) == gcd(v,w)
    
    bad_with_pair = [[0]*(N+1) for _ in range(N+1)]
    
    for u in range(1, N+1):
        for v in range(u+1, N+1):
            guv = gcd_table[u][v]
            mask = 0
            for w in range(1, N+1):
                if w == u or w == v:
                    continue
                if gcd_table[u][w] == guv and gcd_table[v][w] == guv:
                    mask |= (1 << (w - 1))
            bad_with_pair[u][v] = mask
            bad_with_pair[v][u] = mask
            
    def can_add(v, S_bitset):
        """
        Check if v can be added to set S (represented as bitset).
        v is 1-indexed.
        S_bitset is an integer.
        """
        # We need to check if there exist u, w in S such that {u, v, w} is forbidden.
        # This is equivalent to: (Union of bad_with_pair[u][v] for all u in S) intersects S.
        # Note: bad_with_pair[u][v] contains w.
        # So if w is in S, and w is in bad_with_pair[u][v], then {u, v, w} is bad.
        
        union_mask = 0
        # Iterate through bits set in S_bitset
        temp = S_bitset
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            u_idx = lsb.bit_length() - 1 # 0-indexed
            u = u_idx + 1 # 1-indexed
            
            union_mask |= bad_with_pair[u][v]
            
            temp ^= lsb
            
        if union_mask & S_bitset:
            return False
        return True

    def greedy(order):
        S_bitset = 0
        S_list = []
        for v in order:
            if can_add(v, S_bitset):
                S_bitset |= (1 << (v - 1))
                S_list.append(v)
        return S_list, S_bitset

    def local_search(S_list, S_bitset, time_limit):
        """
        Local search to improve the set.
        """
        best_S_list = list(S_list)
        best_S_bitset = S_bitset
        best_size = len(best_S_list)
        
        # Precompute list of all vertices
        all_vertices = list(range(1, N+1))
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add a single vertex
            for v in all_vertices:
                if (1 << (v - 1)) & best_S_bitset:
                    continue
                if can_add(v, best_S_bitset):
                    best_S_bitset |= (1 << (v - 1))
                    best_S_list.append(v)
                    best_size += 1
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try to swap: remove 1, add 2
            # We need to find u in S such that removing it allows adding v and w.
            # This is O(N^3) roughly if done naively.
            # Let's try a simpler swap: remove u, add v. (Size doesn't change, but might enable future adds)
            # Actually, let's just do "remove 1, add 1" to escape local minima, 
            # and "remove 1, add 2" to increase size.
            
            # Try remove 1, add 2
            found_swap = False
            for i, u in enumerate(best_S_list):
                # Remove u
                temp_bitset = best_S_bitset & ~(1 << (u - 1))
                temp_list = best_S_list[:i] + best_S_list[i+1:]
                
                added = 0
                new_bits = 0
                new_list = list(temp_list)
                
                # Try to add two vertices
                candidates = [v for v in all_vertices if not ((1 << (v - 1)) & temp_bitset)]
                
                for v in candidates:
                    if added == 2: break
                    if can_add(v, temp_bitset):
                        temp_bitset |= (1 << (v - 1))
                        new_list.append(v)
                        added += 1
                        
                if added > 1:
                    best_S_bitset = temp_bitset
                    best_S_list = new_list
                    best_size += added - 1
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                continue
                
            # 3. Try remove 2, add 3? Or just random restarts if stuck.
            # Let's try a random perturbation: remove k random elements, then greedy add.
            if best_size > 5:
                # Remove 1 random element
                idx = random.randint(0, len(best_S_list) - 1)
                u = best_S_list[idx]
                temp_bitset = best_S_bitset & ~(1 << (u - 1))
                temp_list = best_S_list[:idx] + best_S_list[idx+1:]
                
                # Greedy add from remaining
                candidates = [v for v in all_vertices if not ((1 << (v - 1)) & temp_bitset)]
                random.shuffle(candidates)
                
                for v in candidates:
                    if can_add(v, temp_bitset):
                        temp_bitset |= (1 << (v - 1))
                        temp_list.append(v)
                        
                if len(temp_list) > best_size:
                    best_S_bitset = temp_bitset
                    best_S_list = temp_list
                    best_size = len(temp_list)
                    improved = True
                    continue
            
            if not improved:
                # Stuck, break or try harder?
                # Given time limit, break and let outer loop restart
                break
                
        return best_S_list, best_S_bitset

    best_result = []
    best_bitset = 0
    
    # Multiple random restarts
    rng = random.Random(42)
    
    # 1. Ascending
    res_list, res_bit = greedy(range(1, N+1))
    if len(res_list) > len(best_result):
        best_result = res_list
        best_bitset = res_bit
        
    # 2. Descending
    if time.time() < deadline - 10:
        res_list, res_bit = greedy(range(N, 0, -1))
        if len(res_list) > len(best_result):
            best_result = res_list
            best_bitset = res_bit
            
    # 3. Random restarts
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        res_list, res_bit = greedy(order)
        if len(res_list) > len(best_result):
            best_result = res_list
            best_bitset = res_bit
            
    # Local search on the best found so far
    if time.time() < deadline - 2:
        best_result, best_bitset = local_search(best_result, best_bitset, deadline - 1)
        
    return best_result if best_result else [1]
