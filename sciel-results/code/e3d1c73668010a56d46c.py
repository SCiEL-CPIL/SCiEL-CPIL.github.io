"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 18
FULL_MASK = (1 << N) - 1

def _is_k4_free(adj_bits):
    """Check if graph is K4-free using bitmasks."""
    for a in range(N):
        na = adj_bits[a]
        for b in range(a + 1, N):
            if not (na >> b) & 1:
                continue
            common_ab = na & adj_bits[b]
            for c in range(b + 1, N):
                if not (common_ab >> c) & 1:
                    continue
                common_abc = common_ab & adj_bits[c]
                mask_above_c = FULL_MASK ^ ((1 << (c + 1)) - 1)
                if common_abc & mask_above_c:
                    return False
    return True

def _largest_triangle_free(adj_bits):
    """Find largest triangle-free induced subgraph size using bitmask enumeration."""
    # Precompute triangle masks: for each pair (i,j) that is an edge,
    # the set of vertices k > j that form a triangle with i,j
    # For a subset S to be triangle-free: for every edge (i,j) in S with i<j,
    # no k in S with j<k should be adjacent to both i and j.
    
    # Strategy: enumerate subsets by size descending, use bitmask operations
    # For each subset, check if it contains any triangle
    
    # Precompute for each vertex v: adj[v] bitmask
    # A subset S is triangle-free iff for all triples (i,j,k) in S with i<j<k,
    # not all three edges exist.
    
    # Efficient check: for a subset mask S, it's triangle-free iff
    # for every i in S, the subgraph induced by S on N(i) ∩ S has no edges
    # i.e., for every i in S: (adj[i] & S) induces no edges within itself
    # i.e., for every i in S: for every j in (adj[i] & S) with j > i:
    #    adj[j] & S & adj[i] & ((1<<(N))-1) ^ ((1<<(j+1))-1) == 0
    
    # Actually simpler: iterate subsets of size s from N down, check triangle-free
    # C(18,10) = 43758, C(18,11) = 31824, C(18,12) = 18564, C(18,13)=8568, C(18,14)=3060, C(18,15)=816, C(18,16)=153, C(18,17)=18, C(18,18)=1
    # Total from 10 to 18: ~107k subsets. Each check is O(s^2) or so.
    # That's feasible for a few thousand evaluations.
    
    # Optimize: use the fact that if a set of size s is triangle-free, 
    # we can build up. But for exact answer, let's just check.
    
    # Faster triangle-free check using bitmasks:
    # A set S (as bitmask) is triangle-free iff:
    # for all i in S: let Ni = adj[i] & S. Then Ni must be an independent set.
    # Ni is independent iff for all j in Ni: (adj[j] & Ni) == 0 (no edges within Ni)
    # Wait, that's not quite right. Ni is the neighbors of i within S.
    # For S to be triangle-free, Ni must be an independent set (no edges among neighbors of i in S).
    
    def is_triangle_free(mask):
        # For each vertex i in mask, check that adj[i] & mask is an independent set
        m = mask
        while m:
            low_bit = m & (-m)
            i = low_bit.bit_length() - 1
            m ^= low_bit
            ni = adj_bits[i] & mask
            # Check if ni is an independent set: no two vertices in ni are adjacent
            # ni is independent iff for all j in ni: (adj[j] & ni) has no bits other than possibly j's self (but no self-loops)
            # Actually: ni is independent iff for all j in ni: adj[j] & ni == 0 (since no self loops, adj[j] & ni contains only neighbors of j that are also in ni)
            # Wait: adj[j] & ni gives neighbors of j that are in ni. For ni to be independent, this should be empty for all j in ni.
            # But adj[j] & ni includes j only if j is adjacent to itself, which it's not.
            # So: ni is independent iff for all j in ni: (adj[j] & ni) == 0
            # Equivalent: the induced subgraph on ni has no edges.
            # We can check: for each j in ni, adj[j] & ni should be 0.
            # But this is O(|ni|^2) in the worst case.
            # Alternative: precompute edge masks. An independent set check:
            # ni is independent iff for all j in ni: adj[j] & ni == 0
            # Or: the number of edges in the induced subgraph on ni is 0.
            # edges_in_ni = sum over j in ni of popcount(adj[j] & ni & (mask of vertices > j in ni))
            # Simpler: just iterate
            nn = ni
            while nn:
                low_bit2 = nn & (-nn)
                j = low_bit2.bit_length() - 1
                nn ^= low_bit2
                if adj_bits[j] & ni:
                    return False
        return True
    
    # Check sizes from N down to 1
    for size in range(N, 0, -1):
        found = False
        # Enumerate all subsets of given size
        for subset in combinations(range(N), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_triangle_free(mask):
                found = True
                break
        if found:
            return size
    return 0

def _bits_to_edges(adj_bits):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(12345)
    
    # === Phase 1: Greedy construction (structural change from seed) ===
    # Start with empty graph, add edges greedily to maximize triangle-breaking
    # while maintaining K4-free property
    
    def greedy_construct():
        adj = [0] * N
        # Order edges by a heuristic: prefer edges that break the most triangles
        # Start with random edge ordering, try to add each if K4-free
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng2 = random.Random(rng.random())
        rng2.shuffle(all_edges)
        
        for (u, v) in all_edges:
            # Try adding edge (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free(adj):
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return adj
    
    best_adj = greedy_construct()
    best_score = _largest_triangle_free(best_adj)
    
    if time.time() > deadline:
        return _bits_to_edges(best_adj)
    
    # === Phase 2: Simulated annealing with bitmask operations ===
    # Moves: add an edge (if K4-free), remove an edge, or swap
    
    def evaluate(adj_bits):
        return _largest_triangle_free(adj_bits)
    
    current_adj = best_adj[:]
    current_score = best_score
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edge_set = set(all_edges)
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling = 0.995
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try adding a random non-edge
            non_edges = [(u, v) for (u, v) in all_edges if not (current_adj[u] >> v) & 1]
            if not non_edges:
                continue
            u, v = rng.choice(non_edges)
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if not _is_k4_free(new_adj):
                continue
            new_score = evaluate(new_adj)
            delta = new_score - current_score  # We want to minimize
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        elif move_type < 0.8:
            # Try removing a random edge
            edges_present = [(u, v) for (u, v) in all_edges if (current_adj[u] >> v) & 1]
            if not edges_present:
                continue
            u, v = rng.choice(edges_present)
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_score = evaluate(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        else:
            # Try a 2-opt swap: remove one edge, add another
            edges_present = [(u, v) for (u, v) in all_edges if (current_adj[u] >> v) & 1]
            non_edges = [(u, v) for (u, v) in all_edges if not (current_adj[u] >> v) & 1]
            if not edges_present or not non_edges:
                continue
            u1, v1 = rng.choice(edges_present)
            u2, v2 = rng.choice(non_edges)
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
            new_adj = current_adj[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            if not _is_k4_free(new_adj):
                continue
            new_score = evaluate(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        
        # Cool down
        T = max(T_min, T * cooling)
        
        if best_score <= 9:
            break
    
    # === Phase 3: Local search refinement on best found ===
    if time.time() < deadline - 10:
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            
            adj = best_adj[:]
            # Try all single-edge additions
            improved = False
            for (u, v) in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                new_adj = adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                if not _is_k4_free(new_adj):
                    continue
                new_score = evaluate(new_adj)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = new_adj[:]
                    improved = True
                    break
            
            # Try all single-edge removals
            if not improved:
                for (u, v) in all_edges:
                    if not (adj[u] >> v) & 1:
                        continue
                    new_adj = adj[:]
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                    new_score = evaluate(new_adj)
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = new_adj[:]
                        improved = True
                        break
            
            if not improved and iteration > 100:
                break
    
    return _bits_to_edges(best_adj)
# EVOLVE-BLOCK-END
