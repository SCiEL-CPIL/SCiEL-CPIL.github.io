"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill
    
    n = 256
    # Build conflict graph: edge (a,b) if (a*b) % (a+b) == 0
    # Vertices are 0..255 representing numbers 1..256
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute neighbor lists as sorted lists for speed
    adj_lists = [list(adj[i]) for i in range(n)]
    
    # Greedy MIS heuristic for initial lower bound
    def greedy_mis(adj):
        """Greedy: repeatedly pick vertex with fewest neighbors, include it, remove it and neighbors."""
        remaining = set(range(n))
        result = set()
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            result.add(best_v)
            remaining.discard(best_v)
            remaining -= adj[best_v]
        return result
    
    # Initial greedy solution
    initial_set = greedy_mis(adj)
    best_size = len(initial_set)
    best_solution = set(initial_set)
    
    # Bitset-accelerated B&B with greedy coloring bound
    # Use Python integers as bitsets for speed
    
    # Precompute adjacency as bitsets
    adj_bits = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)
    
    # Greedy coloring bound using bitsets
    def color_bound(remaining_bits):
        """Return upper bound on MIS size using greedy coloring with bitsets."""
        if remaining_bits == 0:
            return 0
        # Greedy coloring: assign colors greedily
        # Use a simple ordering: process vertices in order of degree in remaining
        # For speed, just process in natural order but skip isolated vertices
        color_of = {}
        max_color = 0
        rem = remaining_bits
        
        # Get list of vertices in remaining
        vlist = []
        temp = rem
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            vlist.append(v)
            temp ^= lsb
        
        # Sort by degree in remaining (descending) for better coloring
        vlist.sort(key=lambda v: -((adj_bits[v] & rem).bit_count()))
        
        for v in vlist:
            used = set()
            # Find colored neighbors of v in remaining
            nb = adj_bits[v] & rem
            while nb:
                lsb = nb & (-nb)
                u = lsb.bit_length() - 1
                if u in color_of:
                    used.add(color_of[u])
                nb ^= lsb
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
        return max_color + 1
    
    # Greedy MIS on a bitset for initial lower bound improvement
    def greedy_on_bits(remaining_bits):
        """Greedy MIS using bitsets. Returns (size, solution_bits)."""
        if remaining_bits == 0:
            return 0, 0
        rem = remaining_bits
        res = 0
        while rem:
            # Pick min degree vertex
            best_v = -1
            best_deg = n + 1
            temp = rem
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (adj_bits[v] & rem).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            res |= (1 << best_v)
            rem &= ~(1 << best_v)
            rem &= ~adj_bits[best_v]
        return res.bit_count(), res
    
    # B&B search using bitsets
    # State: (remaining_bits, included_bits, size)
    # We track best solution globally
    
    best_size_global = best_size
    best_sol_bits_global = 0
    # Convert initial solution to bits
    for v in initial_set:
        best_sol_bits_global |= (1 << v)
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Count for time monitoring
    node_count = [0]
    
    def bnb_search(remaining_bits, included_bits, size):
        nonlocal best_size_global, best_sol_bits_global
        nonlocal start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        node_count[0] += 1
        
        if remaining_bits == 0:
            if size > best_size_global:
                best_size_global = size
                best_sol_bits_global = included_bits
            return
        
        # Upper bound via greedy coloring
        ub = size + color_bound(remaining_bits)
        if ub <= best_size_global:
            return
        
        # Reduction: if any vertex in remaining has no neighbors in remaining,
        # it must be in the MIS
        changed = True
        while changed:
            changed = False
            rem = remaining_bits
            temp = rem
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                if (adj_bits[v] & rem) == 0:
                    # v must be included
                    included_bits |= lsb
                    size += 1
                    rem &= ~lsb
                    rem &= ~adj_bits[v]
                    changed = True
                temp ^= lsb
            remaining_bits = rem
        
        if remaining_bits == 0:
            if size > best_size_global:
                best_size_global = size
                best_sol_bits_global = included_bits
            return
        
        # Re-check upper bound after reduction
        ub = size + color_bound(remaining_bits)
        if ub <= best_size_global:
            return
        
        # Pick vertex with max degree in remaining
        best_v = -1
        best_deg = -1
        temp = remaining_bits
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = (adj_bits[v] & remaining_bits).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        v_bit = 1 << best_v
        
        # Branch 1: include v
        new_remaining_1 = remaining_bits & ~v_bit & ~adj_bits[best_v]
        bnb_search(new_remaining_1, included_bits | v_bit, size + 1)
        
        # Branch 2: exclude v
        new_remaining_2 = remaining_bits & ~v_bit
        bnb_search(new_remaining_2, included_bits, size)
    
    # Start B&B
    bnb_search((1 << n) - 1, 0, 0)
    
    # Convert best solution back to set
    current = set()
    bits = best_sol_bits_global
    while bits:
        lsb = bits & (-bits)
        v = lsb.bit_length() - 1
        current.add(v)
        bits ^= lsb
    
    # Local search polish: try to add vertices that don't conflict
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        for v in range(n):
            if v not in current:
                # Check if v conflicts with any vertex in current
                conflict = False
                for u in adj[v]:
                    if u in current:
                        conflict = True
                        break
                if not conflict:
                    current.add(v)
                    improved = True
    
    # Try swap improvements: remove one vertex, add all possible
    for _ in range(20):
        if (time.time() - start_time) > DEADLINE:
            break
        best_swap = None
        best_gain = 0
        for v in list(current):
            trial = set(current)
            trial.discard(v)
            added = 0
            for u in range(n):
                if u not in trial:
                    conflict = False
                    for w in adj[u]:
                        if w in trial:
                            conflict = True
                            break
                    if not conflict:
                        trial.add(u)
                        added += 1
            gain = added - 1
            if gain > best_gain:
                best_gain = gain
                best_swap = trial
        
        if best_swap:
            current = best_swap
        else:
            break
    
    result = sorted(v + 1 for v in current)
    return result

# EVOLVE-BLOCK-END
