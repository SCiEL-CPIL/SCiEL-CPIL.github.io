import time
from math import gcd
import sys

def construct():
    start = time.time()
    deadline = start + 85.0

    def elapsed():
        return time.time() - start

    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflict masks for all pairs
    conflicts = [0] * N
    
    def inc(a, b):
        g = gcd(a, b)
        u = a // g
        v = b // g
        return (u + v) % g == 0

    for i in range(N):
        a = numbers[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = numbers[j]
            if inc(a, b):
                mask |= (1 << j)
        conflicts[i] = mask

    # 1. Identify odds and evens
    odds_indices = [i for i in range(N) if numbers[i] % 2 == 1]
    evens_indices = [i for i in range(N) if numbers[i] % 2 == 0]

    # 2. Filter evens that are compatible with ALL odds
    odds_ok_indices = []
    for e in evens_indices:
        ok = True
        for o in odds_indices:
            if (conflicts[e] >> o) & 1:
                ok = False
                break
        if ok:
            odds_ok_indices.append(e)

    # 3. Solve MIS on the subgraph induced by odds_ok_indices
    n_ev = len(odds_ok_indices)
    
    if n_ev == 0:
        return [numbers[i] for i in odds_indices]

    # Build local conflict masks
    local_conflicts = [0] * n_ev
    for i in range(n_ev):
        global_i = odds_ok_indices[i]
        mask = 0
        for j in range(n_ev):
            if i == j:
                continue
            global_j = odds_ok_indices[j]
            if (conflicts[global_i] >> global_j) & 1:
                mask |= (1 << j)
        local_conflicts[i] = mask

    # Efficient Tomita's algorithm for Maximum Independent Set
    sys.setrecursionlimit(10000)

    def popcount(x):
        return bin(x).count('1')

    # Greedy coloring to get upper bound on MIS size in subgraph induced by P
    def greedy_color_bound(P, adj):
        if P == 0:
            return 0, []
        
        color_of = {}
        num_colors = 0
        
        # Get list of vertices in P
        verts = []
        temp = P
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            verts.append(v)
            temp &= temp - 1
        
        # Greedy coloring: assign smallest available color
        for v in verts:
            forbidden = set()
            neighbors = adj[v] & P
            temp_n = neighbors
            while temp_n:
                u = (temp_n & (-temp_n)).bit_length() - 1
                if u in color_of:
                    forbidden.add(color_of[u])
                temp_n &= temp_n - 1
            
            c = 1
            while c in forbidden:
                c += 1
            color_of[v] = c
            if c > num_colors:
                num_colors = c
        
        # Sort vertices by color
        ordered = sorted(color_of.items(), key=lambda x: x[1])
        return num_colors, ordered

    best_size = [0]
    best_set = [0]  # bitmask of selected vertices

    def solve(P, X):
        if elapsed() > deadline - 2.0:
            return

        if P == 0:
            sz = popcount(X)
            if sz > best_size[0]:
                best_size[0] = sz
                best_set[0] = X
            return

        # Get upper bound via greedy coloring on P
        bound, ordered = greedy_color_bound(P, local_conflicts)
        
        if popcount(X) + bound <= best_size[0]:
            return

        # Process vertices in the order given by coloring
        for v, c in ordered:
            if (P >> v) & 1 == 0:
                continue
            
            # Include v
            new_X = X | (1 << v)
            new_P = P & ~(1 << v) & ~local_conflicts[v]
            
            solve(new_P, new_X)
            
            # Exclude v
            P = P & ~(1 << v)
            
            # Update bound
            bound, ordered = greedy_color_bound(P, local_conflicts)
            if popcount(X) + bound <= best_size[0]:
                break

    solve((1 << n_ev) - 1, 0)

    # Map back to global indices
    final_indices = set(odds_indices)
    temp = best_set[0]
    while temp:
        idx = (temp & (-temp)).bit_length() - 1
        final_indices.add(odds_ok_indices[idx])
        temp &= temp - 1

    # Convert to values
    final_values = [numbers[i] for i in final_indices]

    # Try to add more evens that weren't in odds_ok_indices
    current_set = set(final_values)
    for e in evens_indices:
        ev = numbers[e]
        if ev in current_set:
            continue
        ok = True
        for cv in current_set:
            if (ev * cv) % (ev + cv) == 0:
                ok = False
                break
        if ok:
            current_set.add(ev)
    
    final_values = list(current_set)
    
    return final_values

if __name__ == "__main__":
    res = construct()
    print(len(res))
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a*b) % (a+b) == 0:
                valid = False
    print(valid)
