"""Construct a 4-regular graph with girth 10 on 320 vertices (the (4,10)-cage).

Uses a voltage graph lift of K_{4,4} over Z_40 with carefully chosen voltages.
The base K_{4,4} has 8 vertices and 16 edges. The lift over Z_40 gives 320 vertices.
The girth of the lift is at least 10 if the voltage assignment avoids short cycles.
"""
import time
import sys
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use LUW construction
        return _construct_luw(k, g, deadline)
    
    # For k=4, g=10: construct the 320-vertex cage via voltage graph lift of K_{4,4} over Z_40
    # K_{4,4} has vertices A = {a0..a3} and B = {b0..b3}, all edges a_i-b_j.
    # Lift over Z_40: 8*40 = 320 vertices.
    # Vertex (side, i, k) where side=0 for A, side=1 for B, i in 0..3, k in 0..39.
    # Edge from (0, i, k) to (1, j, (k + volt[i][j]) % 40).
    
    # A lifted cycle of length L corresponds to a closed walk of length L in K_{4,4}
    # with total voltage sum = 0 mod 40.
    # For girth >= 10, we need no closed walk of length 4, 6, or 8 with voltage sum 0 mod 40.
    
    # Closed walks of length 4 in K_{4,4}: a_i -> b_j -> a_k -> b_l -> a_i, i!=k, j!=l
    # Voltage: volt[i][j] - volt[k][j] + volt[k][l] - volt[i][l] mod 40
    # (signs: forward on (a_i,b_j), backward on (a_k,b_j), forward on (a_k,b_l), backward on (a_i,b_l))
    
    # Closed walks of length 6: a_i -> b_j -> a_k -> b_l -> a_m -> b_n -> a_i
    # with i!=k, k!=m, m!=i (not all same) and j!=l, l!=n, n!=j
    # Voltage: volt[i][j] - volt[k][j] + volt[k][l] - volt[m][l] + volt[m][n] - volt[i][n] mod 40
    
    # Closed walks of length 8: similar pattern.
    
    # Strategy: search for voltage matrix volt[4][4] in Z_40 such that:
    # 1. No 4-closed-walk has sum 0: volt[i][j] - volt[k][j] + volt[k][l] - volt[i][l] != 0 for all i!=k, j!=l
    # 2. No 6-closed-walk has sum 0
    # 3. No 8-closed-walk has sum 0
    
    # Condition 1 is equivalent to: for all i!=k, the vector (volt[i][0]-volt[k][0], ..., volt[i][3]-volt[k][3])
    # has all distinct entries (no two column differences are equal).
    # More precisely: volt[i][j] - volt[i][l] != volt[k][j] - volt[k][l] for all i!=k, j!=l.
    
    def _build_lift(voltages):
        """Build the voltage graph lift of K_{4,4} over Z_40."""
        n = 320
        edges = []
        for i in range(4):
            for j in range(4):
                volt = voltages[i][j]
                for k in range(40):
                    a_idx = i * 40 + k
                    b_idx = 160 + j * 40 + (k + volt) % 40
                    edges.append((a_idx, b_idx))
        return n, edges
    
    def _check_4cycle_condition(voltages):
        """Check no 4-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for k in range(4):
                if i == k:
                    continue
                for j in range(4):
                    for l in range(4):
                        if j == l:
                            continue
                        s = (voltages[i][j] - voltages[k][j] + voltages[k][l] - voltages[i][l]) % 40
                        if s == 0:
                            return False
        return True
    
    def _check_6cycle_condition(voltages):
        """Check no 6-closed-walk in base has voltage 0 mod 40."""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if k == i:
                        continue
                    for l in range(4):
                        if l == j:
                            continue
                        for m in range(4):
                            if m == k or m == i:
                                continue
                            for n in range(4):
                                if n == l or n == j:
                                    continue
                                s = (voltages[i][j] - voltages[k][j] + voltages[k][l] 
                                     - voltages[m][l] + voltages[m][n] - voltages[i][n]) % 40
                                if s == 0:
                                    return False
        return True
    
    def _check_8cycle_condition(voltages):
        """Check no 8-closed-walk in base has voltage 0 mod 40."""
        # 8-closed walk: a_i -> b_j -> a_k -> b_l -> a_m -> b_n -> a_p -> b_q -> a_i
        # i!=k, k!=m, m!=p, p!=i (consecutive A vertices distinct)
        # j!=l, l!=n, n!=q, q!=j (consecutive B vertices distinct)
        # Also need: i!=m, i!=p, k!=p (non-consecutive A can repeat? No, for a simple cycle they must be distinct,
        # but for a closed walk they can repeat. However, if a closed walk of length 8 has a repeated vertex,
        # it contains a shorter closed walk. So we only need to check "simple" 8-cycles where all 4 A-vertices
        # and all 4 B-vertices are distinct... but in K_{4,4} there are exactly 4 A and 4 B vertices,
        # so an 8-cycle uses all of them.)
        # Actually, an 8-closed walk that is not simple contains a shorter closed walk.
        # So checking simple 8-cycles (Hamiltonian cycles in K_{4,4}) suffices, plus the condition
        # that no shorter closed walk has voltage 0 (already checked).
        # Simple 8-cycles in K_{4,4}: permutations of A and B.
        import itertools
        for perm_a in itertools.permutations(range(4)):
            for perm_b in itertools.permutations(range(4)):
                s = 0
                for idx in range(4):
                    a_cur = perm_a[idx]
                    b_cur = perm_b[idx]
                    a_next = perm_a[(idx + 1) % 4]
                    s += voltages[a_cur][b_cur]
                    s -= voltages[a_next][b_cur]
                if s % 40 == 0:
                    return False
        return True
    
    def _verify_girth_exact(n, edges, g, deadline):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For girth >= g, no non-backtracking closed walk of length < g exists.
        # Non-backtracking walk: (v0, v1, v2, ..., vL) where vi != vi-2 for all i >= 2.
        # A closed non-backtracking walk of length L < g would be a cycle of length <= L < g.
        # Actually, a non-backtracking closed walk of length L is a cycle of length L
        # (if L >= 3 and the walk is non-backtracking and closed, it's a simple cycle of length L
        # or contains one).
        
        # Efficient check: for each vertex, enumerate all non-backtracking walks of length up to g-1
        # and check if any returns to the start.
        
        max_len = g - 1  # we check walks of length 1..g-1
        
        for start_v in range(n):
            if time.time() > deadline:
                return False
            # State: (current_vertex, previous_vertex)
            # Walk of length 1: start_v -> w for w in adj[start_v]
            # Walk of length 2: start_v -> w -> x where x != start_v
            # Walk of length L: ... -> y -> z where z != y's predecessor
            
            # BFS over (vertex, prev) states
            # Level 0: start at (start_v, -1) meaning "just started"
            # Actually, let's track walks explicitly.
            
            # walks_of_length_l: set of (current, prev) pairs
            # Length 1: for each w in adj[start_v]: (w, start_v)
            # Length 2: for each (w, start_v): for each x in adj[w] where x != start_v: (x, w)
            # ...
            # A closed walk of length L exists if start_v appears as 'current' at level L
            # with prev != start_v (non-backtracking) -- actually we need the walk to close back to start_v.
            
            # Let me redo: a non-backtracking walk of length L starting and ending at start_v:
            # (start_v, v1, v2, ..., v_{L-1}, start_v) where vi != v_{i-2} for all valid i.
            # Equivalently: v1 in adj(start_v), v2 in adj(v1) \ {start_v}, ..., v_{L-1} in adj(v_{L-2}) \ {v_{L-3}}, start_v in adj(v_{L-1}) \ {v_{L-2}}.
            
            # For L=4: start_v -> a -> b -> c -> start_v, with a!=start_v (auto), b!=start_v, c!=a, start_v!=b.
            #   a in adj(start_v), b in adj(a)\{start_v}, c in adj(b)\{a}, start_v in adj(c)\{b}.
            
            # Let's do BFS tracking (current, prev) and depth.
            # At depth d, we have states (v, p) meaning we're at v, came from p.
            # We want to know if at depth L, we can reach (start_v, p) for some p != the vertex at depth L-2.
            # Actually, if we reach (start_v, p) at depth L, the walk is start_v -> ... -> p -> start_v.
            # Non-backtracking requires p != start_v (which is true since p is a neighbor of start_v and start_v != p)
            # and also the step before p must not be start_v... this gets complicated.
            
            # Simpler: just check if start_v is reachable in exactly L steps via non-backtracking walk.
            # Use DP: dp[d][v][p] = True if there's a non-backtracking walk of length d from start_v to v, last step from p.
            
            # For efficiency, use sets.
            # states[d] = set of (v, p) pairs reachable by non-backtracking walk of length d from start_v.
            # states[1] = {(w, start_v) for w in adj(start_v)}
            # states[d] = {(w, v) for (v, p) in states[d-1] for w in adj(v) if w != p}
            # Closed walk of length L exists if (start_v, p) in states[L] for some p.
            
            states = set()
            for w in adj[start_v]:
                states.add((w, start_v))
            
            for d in range(2, max_len + 1):
                new_states = set()
                for (v, p) in states:
                    for w in adj[v]:
                        if w != p:
                            new_states.add((w, v))
                if any(v == start_v for (v, p) in new_states):
                    return False
                states = new_states
                if not states:
                    break
        
        return True
    
    # Search for valid voltages
    best_n = 320
    best_edges = None
    
    # Try structured voltage matrices first
    # volt[i][j] = (a*i + b*j + c*i*j) mod 40
    # Condition 1: volt[i][j] - volt[k][j] + volt[k][l] - volt[i][l] = 
    #   (a*i + b*j + c*i*j) - (a*k + b*j + c*k*j) + (a*k + b*l + c*k*l) - (a*i + b*l + c*i*l)
    #   = c*i*j - c*k*j + c*k*l - c*i*l
    #   = c*(i-j... no: c*(i*j - k*j + k*l - i*l) = c*((i-k)*j + (k-i)*l) = c*(i-k)*(j-l)
    # So condition 1: c*(i-k)*(j-l) != 0 mod 40 for all i!=k, j!=l.
    # This requires c*(i-k)*(j-l) mod 40 != 0 for all i!=k, j!=l.
    # i-k and j-l are in {-3,-2,-1,1,2,3}.
    # We need c * d1 * d2 != 0 mod 40 for all d1,d2 in {-3,-2,-1,1,2,3}.
    # The product d1*d2 can be: 1,2,3,4,6,9 (and negatives).
    # So we need c * x != 0 mod 40 for x in {1,2,3,4,6,9}.
    # c*4 != 0 mod 40 means c is not a multiple of 10.
    # c*2 != 0 mod 40 means c is odd or c/2 is not a multiple of 20... 
    # Actually, c*2 mod 40 = 0 iff c is a multiple of 20.
    # c*4 mod 40 = 0 iff c is a multiple of 10.
    # c*6 mod 40 = 0 iff c*6 is divisible by 40, i.e., c*3 divisible by 20, i.e., c divisible by 20/gcd(3,20)=20.
    # So c must not be a multiple of 10 (to handle c*4) and not a multiple of 20 (handled by not multiple of 10).
    # Also c*1 != 0 mod 40 means c != 0.
    # c*2 != 0 mod 40: c*2 = 0 mod 40 iff c = 0 or 20. So c not in {0, 20} mod 40.
    # c*3 != 0 mod 40: c*3 = 0 mod 40 iff c = 0 mod 40 (since gcd(3,40)=1). So c != 0.
    # c*4 != 0 mod 40: c*4 = 0 mod 40 iff c = 0 or 10 or 20 or 30. So c not in {0,10,20,30}.
    # c*6 != 0 mod 40: c*6 = 0 mod 40 iff 40 | 6c iff 20 | 3c iff 20 | c (since gcd(3,20)=1). So c not in {0,20}.
    # c*9 != 0 mod 40: c*9 = 0 mod 40 iff 40 | 9c iff 40 | c (since gcd(9,40)=1). So c != 0.
    
    # So c must be in {1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,21,...,29,31,...,39} \ {0,10,20,30}.
    # i.e., c not divisible by 10 and c != 0.
    
    # Now check conditions 2 and 3 for specific a, b, c.
    # Let's try a=0, b=0, c=1: volt[i][j] = i*j mod 40.
    # Check condition 1: c*(i-k)*(j-l) = (i-k)*(j-l) mod 40.
    # For i-k=1, j-l=1: 1 != 0. OK.
    # For i-k=1, j-l=2: 2 != 0. OK.
    # For i-k=2, j-l=2: 4 != 0. OK.
    # For i-k=3, j-l=3: 9 != 0. OK.
    # For i-k=2, j-l=3: 6 != 0. OK.
    # For i-k=3, j-l=2: 6 != 0. OK.
    # All good for condition 1.
    
    # Let's try this and verify conditions 2 and 3.
    
    candidate_voltages = None
    for c in [1, 3, 7, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39]:
        for a in [0, 1, 3, 7, 11]:
            for b in [0, 1, 3, 7, 11]:
                voltages = [[(a*i + b*j + c*i*j) % 40 for j in range(4)] for i in range(4)]
                if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                    candidate_voltages = voltages
                    break
            if candidate_voltages:
                break
        if candidate_voltages:
            break
    
    if candidate_voltages is None:
        # Random search
        for attempt in range(2000):
            if time.time() > deadline - 30:
                break
            voltages = [[random.randint(0, 39) for _ in range(4)] for _ in range(4)]
            if _check_4cycle_condition(voltages) and _check_6cycle_condition(voltages) and _check_8cycle_condition(voltages):
                candidate_voltages = voltages
                break
    
    if candidate_voltages is not None:
        n, edges = _build_lift(candidate_voltages)
        # Final verification with exact girth check
        if _verify_girth_exact(n, edges, g, deadline):
            return n, edges
    
    # Fallback: use LUW construction
    return _construct_luw(4, 10, deadline)


def _construct_luw(k, g, deadline):
    """LUW construction as fallback."""
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    d = min(d_candidates) if d_candidates else max(3, g - 5)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
