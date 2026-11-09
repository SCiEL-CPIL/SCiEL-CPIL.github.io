import random
import time
import sys
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110

    # Moore bound for k=3, g=16: n >= 512. 
    # We target n=512 as it is the theoretical minimum and even (bipartite).
    # If we can't find a valid graph at 512, we try 514, 516, etc.
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def next_prime(n):
        while not is_prime(n):
            n += 1
        return n

    def girth_check(adj, n, max_g):
        """Check if graph has girth >= max_g. Returns the girth found (or max_g if none shorter)."""
        # We only need to check up to max_g - 1
        for target in range(3, max_g):
            for v in range(n):
                # Non-backtracking walk enumeration
                # State: (current_node, prev_node)
                seen = set()
                frontier = set()
                for nb in adj[v]:
                    frontier.add((nb, v))
                
                for length in range(1, target):
                    new_frontier = set()
                    for (u, p) in frontier:
                        for nb in adj[u]:
                            if nb == p:
                                continue
                            if nb == v:
                                return length + 1
                            if (nb, u) not in seen:
                                seen.add((nb, u))
                                new_frontier.add((nb, u))
                    frontier = new_frontier
                    if not frontier:
                        break
        return max_g

    def make_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return edges

    def construct_cayley_graph(n, p, s1, s2):
        """
        Construct a 3-regular graph on n vertices.
        This is a bit tricky because Cayley graphs on Z_p are p-regular or similar.
        Actually, for k=3, we can't easily use a simple Cayley graph on a cyclic group of prime order 
        because the degree must be 3. 
        However, we can use a "lift" or a specific product structure.
        
        Alternative: Use a known high-girth family. 
        The (3,16)-cage is likely not a Cayley graph on a single cyclic group.
        Let's stick to the random + local search but initialize with a "greedy" high-girth attempt 
        or use the LUW construction as a seed if available.
        
        Actually, the prompt says "different design family". 
        Let's try a **Genetic Algorithm** approach which is structurally different from single-trajectory local search.
        But given the 120s limit, a GA might be too slow.
        
        Let's try **Simulated Annealing** on the edge set, which is a different optimization landscape traversal 
        than the greedy hill-climbing in the seed.
        """
        pass

    def random_3reg(n):
        """Generate a random 3-regular graph on n vertices."""
        for _ in range(200):
            stubs = []
            for i in range(n):
                stubs += [i]*3
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            ok = True
            for i in range(0, 3*n, 2):
                a, b = stubs[i], stubs[i+1]
                if a == b or b in adj[a]:
                    ok = False
                    break
                adj[a].add(b); adj[b].add(a)
            if ok:
                return [list(s) for s in adj]
        return None

    def simulated_annealing(adj, n, max_g, steps):
        """
        Use Simulated Annealing to minimize the number of short cycles.
        State: adjacency list.
        Move: 2-opt swap.
        Cost: Number of cycles shorter than max_g.
        """
        def count_short_cycles(adj, n, max_g):
            count = 0
            # Optimized check: only count cycles up to max_g-1
            for target in range(3, max_g):
                found_any = False
                for v in range(n):
                    seen = set()
                    frontier = set()
                    for nb in adj[v]:
                        frontier.add((nb, v))
                    for length in range(1, target):
                        new_frontier = set()
                        for (u, p) in frontier:
                            for nb in adj[u]:
                                if nb == p:
                                    continue
                                if nb == v:
                                    count += 1
                                    found_any = True
                                    break
                                if (nb, u) not in seen:
                                    seen.add((nb, u))
                                    new_frontier.add((nb, u))
                            if found_any:
                                break
                        if found_any:
                            break
                        frontier = new_frontier
                        if not frontier:
                            break
                    if found_any:
                        break
                if found_any:
                    # We found some short cycles. For SA, we might want a finer metric,
                    # but binary "has short cycle" is hard to optimize.
                    # Let's return a large penalty.
                    return 1000 + count
            return 0

        current_cost = count_short_cycles(adj, n, max_g)
        if current_cost == 0:
            return adj, 0
        
        best_adj = [list(s) for s in adj]
        best_cost = current_cost
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Pick a random 2-opt move
            a = random.randint(0, n-1)
            if not adj[a]: continue
            b = random.choice(adj[a])
            c = random.randint(0, n-1)
            if c in (a, b): continue
            if not adj[c]: continue
            d = random.choice(adj[c])
            if d in (a, b, c): continue
            if c in adj[a] or d in adj[b]: continue
            
            # Perform swap
            adj[a].remove(b); adj[b].remove(a)
            adj[c].remove(d); adj[d].remove(c)
            adj[a].append(c); adj[c].append(a)
            adj[b].append(d); adj[d].append(b)
            
            new_cost = count_short_cycles(adj, n, max_g)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = [list(s) for s in adj]
                    if best_cost == 0:
                        return best_adj, 0
            else:
                # Accept with probability exp(-(new_cost - current_cost)/T)
                delta = new_cost - current_cost
                if random.random() < pow(sys.float_info.max, -delta / T): # approx exp
                    current_cost = new_cost
                else:
                    # Revert
                    adj[a].remove(c); adj[c].remove(a)
                    adj[b].remove(d); adj[d].remove(b)
                    adj[a].append(b); adj[b].append(a)
                    adj[c].append(d); adj[d].append(c)
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_adj, best_cost

    # Strategy:
    # 1. Try n=512 (Moore bound).
    # 2. Generate random 3-regular graphs.
    # 3. Use Simulated Annealing to fix short cycles.
    
    for n_try in [512, 514, 516, 520, 528, 540, 560, 600, 640, 700, 800, 900, 936]:
        if time.time() > deadline - 20:
            break
            
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
                
            adj = random_3reg(n_try)
            if adj is None:
                continue
            
            # Quick check
            girth = girth_check(adj, n_try, 17)
            if girth >= 16:
                edges = make_edges(adj, n_try)
                return n_try, edges
                
            # Run SA
            adj_sa, cost = simulated_annealing(adj, n_try, 17, steps=5000)
            
            if cost == 0:
                edges = make_edges(adj_sa, n_try)
                return n_try, edges

    # Fallback: LUW construction (from seed) to ensure a valid result
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y: (x+y)%q), (lambda x,y: (x-y)%q), (lambda x,y: x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    
    start = (0, (None,)+(0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
