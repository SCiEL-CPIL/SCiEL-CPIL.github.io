import time
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        lcm_table[i][i] = i
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    num_bad = len(bad_triples)
    
    # Build conflict graph: edge (x,y) exists if x,y appear together in some bad triple
    conflict = [[False] * n for _ in range(n)]
    for (a, b, c) in bad_triples:
        ia, ib, ic = a - 1, b - 1, c - 1
        conflict[ia][ib] = conflict[ib][ia] = True
        conflict[ia][ic] = conflict[ic][ia] = True
        conflict[ib][ic] = conflict[ic][ib] = True
    
    # Precompute adjacency lists
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and conflict[i][j]:
                adj[i].append(j)
    
    # Also track which triples each pair is in
    pair_triples = defaultdict(list)
    for t, (a, b, c) in enumerate(bad_triples):
        ia, ib, ic = a - 1, b - 1, c - 1
        pair_triples[(ia, ib)].append(t)
        pair_triples[(ib, ia)].append(t)
        pair_triples[(ia, ic)].append(t)
        pair_triples[(ic, ia)].append(t)
        pair_triples[(ib, ic)].append(t)
        pair_triples[(ic, ib)].append(t)
    
    # For each element, list of bad triples it participates in
    elem_triples = [[] for _ in range(n)]
    for t, (a, b, c) in enumerate(bad_triples):
        elem_triples[a-1].append(t)
        elem_triples[b-1].append(t)
        elem_triples[c-1].append(t)
    
    triple_elems = [(a-1, b-1, c-1) for (a, b, c) in bad_triples]
    
    def count_violations(selected):
        """Count the number of bad triples that are fully selected."""
        cnt = 0
        for t in range(num_bad):
            a, b, c = triple_elems[t]
            if selected[a] and selected[b] and selected[c]:
                cnt += 1
        return cnt
    
    def add_violations(selected, x):
        """If we add element x, how many new violations are created?"""
        cnt = 0
        for t in elem_triples[x]:
            a, b, c = triple_elems[t]
            others = [v for v in (a, b, c) if v != x]
            if selected[others[0]] and selected[others[1]]:
                cnt += 1
        return cnt
    
    # Phase 1: Greedy construction with multiple orderings
    best_size = 0
    best_sel = [False] * n
    
    # Ordering 1: by degree (fewest constraints first)
    degree = [len(adj[i]) for i in range(n)]
    order1 = sorted(range(n), key=lambda i: (degree[i], i))
    
    sel = [False] * n
    for x in order1:
        if add_violations(sel, x) == 0:
            sel[x] = True
    sz = sum(sel)
    if sz > best_size:
        best_size = sz
        best_sel = sel[:]
    
    # Ordering 2: reverse degree
    order2 = sorted(range(n), key=lambda i: (-degree[i], i))
    sel = [False] * n
    for x in order2:
        if add_violations(sel, x) == 0:
            sel[x] = True
    sz = sum(sel)
    if sz > best_size:
        best_size = sz
        best_sel = sel[:]
    
    # Ordering 3: natural order
    sel = [False] * n
    for x in range(n):
        if add_violations(sel, x) == 0:
            sel[x] = True
    sz = sum(sel)
    if sz > best_size:
        best_size = sz
        best_sel = sel[:]
    
    # Ordering 4: reverse natural
    sel = [False] * n
    for x in range(n-1, -1, -1):
        if add_violations(sel, x) == 0:
            sel[x] = True
    sz = sum(sel)
    if sz > best_size:
        best_size = sz
        best_sel = sel[:]
    
    # Greedy addition pass on best
    changed = True
    while changed and time.time() < deadline - 40:
        changed = False
        for x in range(n):
            if not best_sel[x] and add_violations(best_sel, x) == 0:
                best_sel[x] = True
                changed = True
    best_size = sum(best_sel)
    
    # Phase 2: Exact branch-and-bound for maximum triangle-free subset
    # We model this as: find max subset S such that no bad triple is fully in S
    # This is equivalent to: for each bad triple (a,b,c), at most 2 of {a,b,c} in S
    
    # Use branch and bound with the following approach:
    # - Process elements in a good order
    # - At each step, decide to include or exclude the current element
    # - Bound: current_size + remaining_elements (optimistic)
    # - Prune: if current_size + remaining <= best_size, backtrack
    
    # Better: use the conflict structure. If we include element x, we can't include
    # any pair (y,z) where (x,y,z) is a bad triple AND both y,z are already included.
    # Actually, the constraint is: no three elements form a bad triple.
    
    # Let's use a different exact approach: 
    # The problem is to find the maximum subset with no "forbidden triple".
    # This is a 3-uniform hypergraph independent set problem.
    
    # For exact solution, use branch and bound with strong pruning:
    # - Maintain a set of "active" elements (not yet decided)
    # - Maintain current selection
    # - Bound: |current| + |active| - (minimum removals needed to make active triangle-free)
    #   The last term is hard to compute, so use |current| + |active| as upper bound
    
    # Actually, let's use a smarter approach. Since the hypergraph is sparse,
    # we can use the following:
    # - Start with a good initial solution (from greedy)
    # - Use local search with 1-flip and 2-flip moves
    # - Add random restarts
    
    # Phase 3: Intensive local search
    import random
    random.seed(12345)
    
    def local_search(sel, max_iters=200):
        """Perform local search with 1-remove/k-add and 2-remove/k-add moves."""
        sz = sum(sel)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 10:
            improved = False
            iters += 1
            
            # Try 1-remove, k-add
            for r in range(n):
                if not sel[r]:
                    continue
                sel[r] = False
                added = []
                for x in range(n):
                    if not sel[x] and add_violations(sel, x) == 0:
                        sel[x] = True
                        added.append(x)
                net = len(added) - 1
                if net > 0:
                    sz += net
                    improved = True
                else:
                    for x in added:
                        sel[x] = False
                    sel[r] = True
            
            # Try 2-remove, k-add
            for r1 in range(n):
                if not sel[r1]:
                    continue
                for r2 in range(r1 + 1, n):
                    if not sel[r2]:
                        continue
                    sel[r1] = False
                    sel[r2] = False
                    added = []
                    for x in range(n):
                        if not sel[x] and add_violations(sel, x) == 0:
                            sel[x] = True
                            added.append(x)
                    net = len(added) - 2
                    if net > 0:
                        sz += net
                        improved = True
                    else:
                        for x in added:
                            sel[x] = False
                        sel[r1] = True
                        sel[r2] = True
        
        return sz
    
    # Apply local search to best
    best_size = local_search(best_sel, max_iters=100)
    
    # Phase 4: Random restarts
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random ordering
        order = list(range(n))
        random.shuffle(order)
        
        sel = [False] * n
        for x in order:
            if add_violations(sel, x) == 0:
                sel[x] = True
        
        # Greedy addition
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if not sel[x] and add_violations(sel, x) == 0:
                    sel[x] = True
                    changed = True
        
        sz = sum(sel)
        if sz > best_size:
            best_size = sz
            best_sel = sel[:]
        
        # Local search
        sz = local_search(sel, max_iters=50)
        if sz > best_size:
            best_size = sz
            best_sel = sel[:]
    
    # Phase 5: Final polish
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for x in range(n):
            if not best_sel[x] and add_violations(best_sel, x) == 0:
                best_sel[x] = True
                changed = True
    
    # Verify the solution is valid
    # Check no bad triple is fully selected
    for t in range(num_bad):
        a, b, c = triple_elems[t]
        if best_sel[a] and best_sel[b] and best_sel[c]:
            # Remove one element to fix
            # Remove the one with highest degree
            degs = [len(adj[a]), len(adj[b]), len(adj[c])]
            max_idx = degs.index(max(degs))
            best_sel[[a, b, c][max_idx]] = False
            break
    
    # Final verification
    for t in range(num_bad):
        a, b, c = triple_elems[t]
        if best_sel[a] and best_sel[b] and best_sel[c]:
            best_sel[c] = False
    
    result = [x + 1 for x in range(n) if best_sel[x]]
    if not result:
        result = [1]
    
    return result
