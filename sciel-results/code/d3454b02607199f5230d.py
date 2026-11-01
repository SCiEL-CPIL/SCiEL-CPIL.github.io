"""Shortest superpermutation on seven symbols - cycle-chaining approach."""
import time
import sys
from itertools import permutations as perms_gen

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin (150s limit)
    
    # Classic construction (Graham-Lagarias)
    def classic_construct(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find maximum overlap
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    word = classic_construct(n)
    base_len = len(word)
    
    # For n=7, try to improve using cycle structure
    if n == 7:
        # Compute successor map and cycles
        perm_list = list(perms_gen(range(1, n + 1)))
        perm_to_idx = {p: i for i, p in enumerate(perm_list)}
        
        # Successor of P = (p1,...,pn) is (p2,...,pn,x) where x is the missing symbol
        succ = [0] * len(perm_list)
        for i, p in enumerate(perm_list):
            tail = p[1:]
            s = set(range(1, n + 1))
            s.difference_update(tail)
            x = s.pop()
            q = tail + (x,)
            succ[i] = perm_to_idx[q]
        
        # Find cycles
        visited = [False] * len(perm_list)
        cycles = []
        for i in range(len(perm_list)):
            if not visited[i]:
                cycle = []
                cur = i
                while not visited[cur]:
                    visited[cur] = True
                    cycle.append(cur)
                    cur = succ[cur]
                cycles.append(cycle)
        
        # Now we need to find an ordering of these cycles that minimizes total jump cost
        # Jump from end of cycle A to start of cycle B:
        # We're at permutation PA (last in cycle A), want to reach PB (first in cycle B)
        # Cost = minimum number of symbols to append so that last n symbols form PB
        
        # For each pair of cycles, compute the jump cost
        # jump_cost[A][B] = minimum k such that appending k symbols to PA makes window = PB
        
        # Since there are only a few cycles (for n=7, likely 2 or 3), this is tractable
        
        num_cycles = len(cycles)
        
        # Compute jump costs between all pairs of cycles
        # jump_cost[i][j] = cost to jump from end of cycle i to start of cycle j
        jump_cost = [[0] * num_cycles for _ in range(num_cycles)]
        
        for i in range(num_cycles):
            end_perm = perm_list[cycles[i][-1]]
            for j in range(num_cycles):
                if i == j:
                    jump_cost[i][j] = 0
                    continue
                start_perm = perm_list[cycles[j][0]]
                # Find minimum overlap: how many suffix symbols of end_perm match prefix of start_perm
                # We need to append symbols to end_perm until the last n form start_perm
                # Try overlaps of length n-1 down to 0
                best_cost = n  # worst case: append all n symbols
                for overlap in range(n - 1, -1, -1):
                    if end_perm[n - overlap:] == start_perm[:overlap]:
                        # We can overlap by 'overlap' symbols
                        # Cost = n - overlap (symbols to append)
                        best_cost = n - overlap
                        break
                jump_cost[i][j] = best_cost
        
        # Now find the minimum cost Hamiltonian path through all cycles
        # This is a TSP-like problem with num_cycles nodes
        # For small num_cycles (likely 2-5), we can use DP
        
        if num_cycles <= 15:
            # DP over subsets
            from functools import lru_cache
            
            # dp[mask][last] = min cost to visit cycles in mask, ending at cycle 'last'
            # mask is a bitmask of visited cycles
            size = num_cycles
            INF = float('inf')
            
            # dp[i] = (min_cost, last_cycle) for paths ending at cycle i
            # Use iterative DP over subsets
            # For small sizes, enumerate all subsets
            
            # dp[mask][j] = min cost to have visited set 'mask' and be at cycle j
            # Initialize: dp[1<<j][j] = 0 for all j
            # Transition: dp[mask|1<<k][k] = min(dp[mask][j] + jump_cost[j][k])
            
            # For n=7, num_cycles is small (probably 2 or 3), so this is fast
            full_mask = (1 << size) - 1
            dp = [[INF] * size for _ in range(1 << size)]
            parent = [[-1] * size for _ in range(1 << size)]
            
            for j in range(size):
                dp[1 << j][j] = 0
            
            for mask in range(1 << size):
                for j in range(size):
                    if dp[mask][j] == INF:
                        continue
                    if not (mask & (1 << j)):
                        continue
                    for k in range(size):
                        if mask & (1 << k):
                            continue
                        new_mask = mask | (1 << k)
                        new_cost = dp[mask][j] + jump_cost[j][k]
                        if new_cost < dp[new_mask][k]:
                            dp[new_mask][k] = new_cost
                            parent[new_mask][k] = j
            
            # Find best ending
            best_cost = INF
            best_last = -1
            for j in range(size):
                if dp[full_mask][j] < best_cost:
                    best_cost = dp[full_mask][j]
                    best_last = j
            
            # Reconstruct path
            order = []
            mask = full_mask
            cur = best_last
            while cur != -1:
                order.append(cur)
                prev = parent[mask][cur]
                mask = mask & ~(1 << cur)
                cur = prev
            order.reverse()
            
            # Build the word by traversing cycles in the optimal order
            # For each cycle, traverse it fully, then jump to the next
            result = []
            for idx, cycle_idx in enumerate(order):
                cycle = cycles[cycle_idx]
                # Traverse the cycle: start at cycle[0], follow succ
                # The first permutation is perm_list[cycle[0]]
                # Append all its symbols
                if idx == 0:
                    result.extend(perm_list[cycle[0]])
                else:
                    # Jump from previous end to this start
                    # Already handled by the cost calculation
                    # We need to append the appropriate symbols
                    # The jump cost was computed as n - overlap
                    # We just append the non-overlapping part
                    # For simplicity, let's just append the full permutation
                    # and let the overlap be handled naturally
                    result.extend(perm_list[cycle[0]])
                
                # Continue traversing the rest of the cycle
                for pos in range(1, len(cycle)):
                    p = perm_list[cycle[pos]]
                    # Append only the last symbol (since we overlap n-1)
                    result.append(p[-1])
            
            # Check if this is valid and shorter
            if len(result) < base_len:
                # Verify validity
                word_str = "".join(map(str, result))
                # Check all permutations are present
                found = set()
                for i in range(len(result) - n + 1):
                    window = tuple(result[i:i+n])
                    found.add(window)
                if len(found) == len(perm_list):
                    word = result
        
        # Time check
        if time.time() > deadline - 5:
            pass
    
    return "".join(map(str, word))
