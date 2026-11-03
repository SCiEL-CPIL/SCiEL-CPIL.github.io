"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    if n <= 1:
        return "1"
    
    # For small n, we can use the classic construction or even brute force/DP
    # But TSP heuristic works well for n up to 7 or 8
    if n <= 4:
        return _classic_construct(n)
    
    # Generate all permutations
    from itertools import permutations
    perms = list(permutations(range(1, n+1)))
    N = len(perms)
    
    # Map permutation to index for fast lookup
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap costs
    # cost[i][j] = number of chars to append to perms[i] to reach perms[j]
    # This is n - max k such that perms[i][k:] == perms[j][:n-k]
    # Actually, it's simpler: find the longest suffix of perms[i] that is a prefix of perms[j]
    # Let L be the length of that suffix. The cost is n - L.
    # We can precompute this for all pairs? N=5040, N^2 = 25M. Might be slow in Python if not optimized.
    # Let's try to compute on the fly or use a smarter representation.
    
    # Alternative: Represent permutations as integers or use string matching?
    # For n=7, strings are okay.
    
    # Let's precompute a dictionary of transitions for each permutation
    # next_perms[i] = list of (j, cost)
    # Actually, we just need to find the next best permutation.
    
    # Heuristic: Nearest Neighbor with random restarts
    # We want to minimize the sum of costs.
    
    def get_cost(p1, p2):
        # Find longest suffix of p1 that is prefix of p2
        # p1 and p2 are tuples of length n
        max_k = n
        # Check k from n down to 1
        # If p1[-k:] == p2[:k], then we can append p2[k:]
        # Cost is n - k
        for k in range(n, 0, -1):
            if p1[n-k:] == p2[:k]:
                return n - k
        # If no overlap (k=0 is not possible for distinct perms? 
        # Actually, if p1 and p2 share no suffix-prefix, cost is n.
        # But since they are permutations of 1..n, they must share at least one element?
        # No, e.g., 1234 and 4321. Suffix 4, Prefix 4. k=1. Cost 3.
        # Is it possible k=0?
        # p1 ends with x, p2 starts with y. If x != y for all possible overlaps?
        # Wait, p1 is a permutation. p2 is a permutation.
        # If p1 = (1,2,3) and p2 = (3,1,2). Suffix 3, Prefix 3. k=1.
        # If p1 = (1,2,3) and p2 = (2,3,1). Suffix 3, Prefix 2. No. Suffix 23, Prefix 23. k=2.
        # It is always possible to have k >= 1?
        # Let last = p1[-1]. Let first = p2[0].
        # If last == first, k >= 1.
        # If last != first, can we have k=2? p1[-2:] == p2[:2].
        # If not, k=1 fails.
        # So k can be 0?
        # If k=0, we append all n chars.
        # Example: n=3. p1=(1,2,3), p2=(3,2,1). 
        # Suffix 3, Prefix 3. k=1. Cost 2.
        # Example: p1=(1,2,3), p2=(2,1,3).
        # Suffix 3, Prefix 2. No.
        # Suffix 23, Prefix 21. No.
        # Suffix 123, Prefix 213. No.
        # So k=0? Cost 3.
        # Yes, k can be 0.
        return n

    # To speed up, we can precompute for each permutation p, the cost to all other permutations?
    # That's N^2 operations. 5040^2 ~ 25M. In Python, this might take a few seconds.
    # Let's try to do it.
    
    costs = [[0]*N for _ in range(N)]
    # This memory is 5040*5040*8 bytes ~ 200MB. Might be too much or slow.
    # Let's compute on the fly but cache it?
    # Or just compute nearest neighbors for each node?
    
    # Let's try a simpler approach: 
    # Start with a random permutation.
    # Repeatedly pick the nearest unvisited neighbor.
    # This is O(N^2) per run.
    # With 1000 restarts, 25M * 1000 = 25B ops. Too slow.
    
    # We need a better heuristic or precomputation.
    
    # Precompute "next" candidates?
    # For each permutation p, and each possible next char c, 
    # what permutation does it lead to?
    # If we append c to p, we get a sequence of length n+1.
    # The last n elements form a new permutation q.
    # So from p, we can go to q by appending 1 char. Cost 1.
    # If we don't append 1 char, we append more.
    
    # Actually, the graph is:
    # Node: Permutation
    # Edge p -> q exists if q can be formed by appending some chars to p.
    # The minimum cost is 1 if the last n-1 of p are the first n-1 of q? No.
    # If we append 1 char c to p, the new window is p[1:] + (c,).
    # So q = (p[1], p[2], ..., p[n-1], c).
    # This is a valid permutation if c is not in p[1:].
    # Since p is a permutation, p[1:] contains n-1 distinct elements.
    # The only element not in p[1:] is p[0].
    # So c MUST be p[0].
    # Thus, from any permutation p, there is exactly ONE transition with cost 1:
    # q = (p[1], ..., p[n-1], p[0]).
    # This is the "left rotation" or shift.
    
    # If we take cost 1 steps, we just cycle through the n rotations of the same set of relative orders?
    # No, p = (1,2,3). q = (2,3,1).
    # From q, next is (3,1,2).
    # From (3,1,2), next is (1,2,3).
    # So cost 1 edges form cycles of length n.
    # To visit all N permutations, we cannot just use cost 1 edges.
    # We must take "jumps" of cost > 1.
    
    # A jump of cost k means we append k characters.
    # This effectively changes the permutation more drastically.
    
    # This looks like a TSP on a graph where:
    # - There are N nodes.
    # - There are N "cheap" edges (cost 1) forming n disjoint cycles.
    # - There are many "expensive" edges.
    
    # We can contract the cost-1 cycles into "super-nodes"?
    # Each cycle contains n permutations.
    # If we are at any node in a cycle, we can reach any other node in the same cycle with cost n-1 (by going around) or less?
    # Actually, to go from p to p' in the same cycle, the cost is the distance along the cycle.
    
    # Let's define the "cycle" of a permutation p as the set {p, rot(p), rot^2(p), ...}.
    # There are N/n cycles.
    # Let's index the cycles.
    
    # This structure suggests that we should visit the cycles in some order, and within each cycle, visit all nodes.
    # Since we must visit ALL permutations, we must visit all nodes in all cycles.
    
    # Strategy:
    # 1. Identify all cycles.
    # 2. Find a path that visits all cycles, entering and exiting them at optimal points.
    # 3. Within each cycle, traverse all nodes.
    
    # Let's identify the cycles.
    visited_cycles = set()
    cycles = []
    for i in range(N):
        if i not in visited_cycles:
            # Find the cycle containing i
            cycle = []
            cur = i
            while cur not in visited_cycles:
                visited_cycles.add(cur)
                cycle.append(cur)
                # Next node in cycle: p -> (p[1],...,p[n-1],p[0])
                p = perms[cur]
                next_p = p[1:] + p[:1]
                cur = perm_to_idx[next_p]
            cycles.append(cycle)
            
    C = len(cycles) # Number of cycles = N/n
    
    # Now we have C super-nodes (cycles).
    # We need to visit all C cycles.
    # Cost to enter a cycle and visit all its nodes and exit?
    # If we enter cycle c at node u and exit at node v, having visited all nodes in c:
    # The cost is the length of the path visiting all nodes in c, starting at u and ending at v.
    # Since the nodes in c form a directed cycle, the path must traverse the entire cycle.
    # So the cost is (size of cycle - 1) if we start and end at specific points?
    # Actually, if we enter at u, we must visit all nodes.
    # The "natural" way is to follow the cycle.
    # If we enter at u, we can go u -> next(u) -> ... -> prev(u) -> u (back to start).
    # But we need to EXIT to the next cycle.
    # So we enter at u, traverse the cycle, and exit from some node v to the next cycle.
    # The cost within the cycle is (n-1) if we traverse all edges?
    # No, if we enter u, we are at u. We need to visit n-1 other nodes.
    # The cost to visit them is n-1 (using cost 1 edges).
    # Then we are at some node v. We pay the cost to jump to the next cycle.
    
    # So the problem reduces to TSP on the C cycles!
    # Node: Cycle
    # Edge cost from Cycle A to Cycle B:
    # min over u in A, v in B of (cost to jump from u to v)
    # Note: The cost to jump from u to v is get_cost(perms[u], perms[v]).
    
    # Let's precompute the minimum jump cost between any two cycles.
    # C = 5040 / 7 = 720.
    # 720^2 = 518,400 pairs.
    # For each pair, we need min over 7x7 = 49 combinations of u,v.
    # Total ops: 518,400 * 49 ~ 25M. This is feasible.
    
    # Let's do this.
    
    # cycle_id[i] = index of the cycle containing permutation i
    cycle_id = [0]*N
    for cid, c in enumerate(cycles):
        for idx in c:
            cycle_id[idx] = cid
            
    # Precompute min_jump[cid1][cid2]
    min_jump = [[n]*C for _ in range(C)] # Max cost is n
    
    for i in range(C):
        for j in range(C):
            if i == j: continue
            min_c = n
            for u in cycles[i]:
                pu = perms[u]
                for v in cycles[j]:
                    pv = perms[v]
                    c_cost = get_cost(pu, pv)
                    if c_cost < min_c:
                        min_c = c_cost
            min_jump[i][j] = min_c
            
    # Now solve TSP on C nodes with cost matrix min_jump.
    # C = 720.
    # Use Nearest Neighbor with random restarts.
    
    best_length = n * (N + 1) # Upper bound
    best_path_cycles = None
    
    # Time check
    if time.time() - start_time > deadline - 5:
         # Fallback to classic
         return _classic_construct(n)
         
    max_restarts = 500
    restarts_done = 0
    while restarts_done < max_restarts and time.time() - start_time < deadline - 2:
        # Random start
        start_c = random.randint(0, C-1)
        unvisited = set(range(C))
        unvisited.remove(start_c)
        path = [start_c]
        
        cur = start_c
        while unvisited:
            # Find nearest neighbor
            best_n = -1
            best_cost = float('inf')
            for nxt in unvisited:
                c_cost = min_jump[cur][nxt]
                if c_cost < best_cost:
                    best_cost = c_cost
                    best_n = nxt
            path.append(best_n)
            unvisited.remove(best_n)
            cur = best_n
            
        # Calculate total cost
        # Total length = n (first perm) + sum of costs between cycles + sum of internal costs?
        # Wait, the cost between cycles is the jump cost.
        # The internal cost of visiting a cycle is n-1 (to visit all n nodes).
        # But we also need to account for the start.
        # Let's trace:
        # Start with perm p_start in cycle c0.
        # Length so far: n.
        # Visit rest of c0: cost n-1.
        # Jump to c1: cost min_jump[c0][c1].
        # Visit c1: cost n-1.
        # ...
        # Jump to c_{C-1}: cost min_jump[c_{C-2}][c_{C-1}].
        # Visit c_{C-1}: cost n-1.
        
        # Total Length = n + (C-1) * (n-1) + sum_{i=0}^{C-2} min_jump[path[i]][path[i+1]]
        
        total_internal = (C - 1) * (n - 1)
        total_jump = 0
        for k in range(len(path) - 1):
            total_jump += min_jump[path[k]][path[k+1]]
            
        current_length = n + total_internal + total_jump
        
        if current_length < best_length:
            best_length = current_length
            best_path_cycles = path[:]
            
        restarts_done += 1
        
    if best_path_cycles is None:
        return _classic_construct(n)
        
    # Reconstruct the actual word
    # We need to pick the specific entry/exit points in the cycles to achieve the min_jump.
    # And we need to order the nodes within the cycle.
    
    # This reconstruction is tricky because min_jump was just the min.
    # We need to find the specific u in c_i and v in c_{i+1} that achieves the cost.
    
    word = []
    
    # Start with first cycle
    c0 = best_path_cycles[0]
    # Pick any node in c0 to start? 
    # To minimize the first jump, we should pick the u in c0 that minimizes jump to c1.
    # But we also need to visit all nodes in c0.
    # If we enter at u, we must traverse the cycle.
    # The exit point is determined by where we end up after traversing the cycle.
    # If we enter at u, and traverse n-1 steps, we end at prev(u).
    # So the exit point is fixed by the entry point!
    # Entry u -> Exit v = prev(u) in the cycle.
    
    # So for each transition c_i -> c_{i+1}, the cost depends on entry u_i and exit v_i = prev(u_i).
    # We want to minimize sum of costs.
    # This is a shortest path problem on the cycles, where the state is the entry node.
    
    # Let's refine the TSP to include the entry node.
    # State: (cycle_id, entry_node)
    # But C * n = 720 * 7 = 5040 states.
    # This is just the original TSP on permutations!
    # But we constrained the path within a cycle to be the full cycle.
    # Is it always optimal to visit a cycle fully before leaving?
    # Yes, because leaving and re-entering a cycle would incur extra jump costs.
    # Jumps are expensive (cost >= 1, usually > 1).
    # Traversing within a cycle costs 1 per step.
    # So it's better to stay in the cycle and finish it.
    
    # So we just need to find the optimal entry node for each cycle.
    # Let entry[c] be the node in cycle c where we enter.
    # Exit[c] is the node in cycle c where we leave, which is prev(entry[c]).
    # Cost of transition c -> c' is get_cost(perms[exit[c]], perms[entry[c']]).
    
    # We want to minimize sum of these costs.
    # This is still a TSP-like problem but with 720 nodes and 7 choices each.
    # We can use the path found by the coarse TSP (best_path_cycles) and then optimize the entry nodes locally.
    
    # For the sequence of cycles best_path_cycles:
    # We can try all 7^C combinations? No.
    # We can do dynamic programming or local search on the entry nodes.
    
    # Let's just use the entry node that minimizes the outgoing jump for each cycle, 
    # assuming the incoming jump is fixed? 
    # Actually, the entry node affects BOTH the incoming and outgoing jump costs.
    
    # Let's try a simple local search:
    # Initialize entry[c] = 0 (first node in cycle c).
    # Repeat:
    #   For each cycle c in the path, try changing entry[c] to all 7 nodes in the cycle.
    #   Calculate the change in total cost (incoming + outgoing).
    #   Keep the best.
    
    entries = [0] * C # index within the cycle list
    
    # Calculate total cost for a given set of entries
    def calc_total_cost(entries_list):
        total = 0
        for k in range(len(best_path_cycles) - 1):
            c1 = best_path_cycles[k]
            c2 = best_path_cycles[k+1]
            u = cycles[c1][entries[c1]]
            v = cycles[c2][entries[c2]]
            # Exit from c1 is prev(u)
            # Find index of u in cycles[c1]
            idx_u = entries[c1]
            idx_v = entries[c2]
            # prev of idx_u is (idx_u - 1) % n
            exit_u_idx = (idx_u - 1) % n
            exit_u = cycles[c1][exit_u_idx]
            entry_v = v
            
            total += get_cost(perms[exit_u], perms[entry_v])
        return total
        
    current_cost = calc_total_cost(entries)
    
    # Local search
    improved = True
    while improved and time.time() - start_time < deadline - 1:
        improved = False
        for k in range(len(best_path_cycles)):
            c = best_path_cycles[k]
            old_entry = entries[c]
            for new_e in range(n):
                if new_e == old_entry: continue
                entries[c] = new_e
                new_cost = calc_total_cost(entries)
                if new_cost < current_cost:
                    current_cost = new_cost
                    old_entry = new_e
                    improved = True
                    break # Restart loop? Or just continue
                else:
                    entries[c] = old_entry
                    
    # Reconstruct word
    # Start with first cycle
    c0 = best_path_cycles[0]
    start_node = cycles[c0][entries[c0]]
    word = list(perms[start_node])
    
    for k in range(len(best_path_cycles)):
        c = best_path_cycles[k]
        # Visit all nodes in cycle c starting from entries[c]
        # We are currently at the exit of the previous cycle, which jumped to entries[c].
        # So we are at entries[c].
        # We need to visit the other n-1 nodes.
        # Order: entries[c], next(entries[c]), ..., prev(entries[c])
        
        if k == 0:
            # We already added the start node.
            # Add the rest
            for step in range(1, n):
                idx = (entries[c] + step) % n
                node = cycles[c][idx]
                p = perms[node]
                # Append the new char
                # The new char is p[-1]
                word.append(p[-1])
        else:
            # We just jumped to entries[c].
            # The jump cost was get_cost(prev_exit, entries[c]).
            # This implies we appended (n - overlap) chars.
            # We need to actually append those chars.
            # The chars to append are the suffix of entries[c] that is not in prev_exit.
            # Actually, we can just simulate the jump.
            # We know the cost. We know the destination.
            # We can just append the necessary chars to reach entries[c].
            
            # Let's find the overlap between current word end and perms[entries[c]]
            # Current word ends with ... 
            # We are at the end of the previous cycle's traversal.
            # The last permutation in the word is the exit node of the previous cycle.
            # Let's call it p_exit.
            # We want to reach p_entry = perms[entries[c]].
            # We append chars until the last n chars match p_entry.
            
            p_entry = perms[cycles[c][entries[c]]]
            # Find longest suffix of word that is prefix of p_entry
            # We can just try appending chars from p_entry
            # Actually, we know the cost k.
            # We can just take the last n-1 chars of word, and find how many match p_entry.
            
            # Simpler: 
            # The jump was calculated as get_cost(p_exit, p_entry).
            # This is n - L, where L is the overlap.
            # So we need to append p_entry[L:]
            
            # How to find L?
            # p_exit is the last permutation in the word.
            # p_exit = tuple(word[-n:])
            # We need max L such that p_exit[n-L:] == p_entry[:L]
            
            p_exit = tuple(word[-n:])
            L = 0
            for l in range(n, 0, -1):
                if p_exit[n-l:] == p_entry[:l]:
                    L = l
                    break
            # Append p_entry[L:]
            for ch in p_entry[L:]:
                word.append(ch)
                
            # Now we are at p_entry.
            # Visit the rest of the cycle
            for step in range(1, n):
                idx = (entries[c] + step) % n
                node = cycles[c][idx]
                p = perms[node]
                word.append(p[-1])
                
    return "".join(map(str, word))

def _classic_construct(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
