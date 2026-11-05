import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55  # Safety margin before 60s
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build adjacency/conflict sets
    # adj[i] contains j (0-indexed) such that (i+1) + (j+1) is a perfect square
    # Note: We treat the graph as undirected for the purpose of finding an Independent Set.
    # If i+j is a square, j+i is a square.
    adj = [set() for _ in range(N)]
    
    # Identify vertices with self-loops (2*v is a square)
    # These vertices cannot be in the solution set.
    forbidden = set()
    
    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden.add(i)
        
        # Find neighbors
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                adj[i].add(j)
    
    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if i not in forbidden]
    
    # Precompute degrees for degree-based ordering
    degrees = [len(adj[i]) for i in range(N)]
    
    # Modular structure: group by residue mod 8 to exploit square-sum structure
    # Squares mod 8 are {0, 1, 4}. If a+b is square, then a+b mod 8 in {0,1,4}.
    # This creates structure we can exploit in initialization.
    residue_groups = {r: [] for r in range(8)}
    for i in candidates:
        residue_groups[(i + 1) % 8].append(i)
    
    best_size = 0
    best_set = frozenset()
    
    def greedy_solve(random_seed=None, use_degree_order=False):
        if random_seed is not None:
            random.seed(random_seed)
            
        if use_degree_order:
            # Sort by degree (lower degree first - greedy heuristic)
            order = sorted(candidates, key=lambda x: (degrees[x], random.random()))
        else:
            # Randomize order of candidates
            order = candidates[:]
            random.shuffle(order)
        
        current_set = set()
        
        for v in order:
            if not (adj[v] & current_set):
                current_set.add(v)
                
        return current_set

    def modular_greedy(random_seed=None):
        """
        Exploit modular structure: process residues in an order that
        minimizes cross-residue conflicts.
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Try different residue processing orders
        # Squares mod 8: {0, 1, 4}
        # If a mod 8 = r1 and b mod 8 = r2, then (r1+r2) mod 8 should not be in {0,1,4}
        # for them to potentially coexist (necessary condition, not sufficient)
        
        # Compute conflict matrix between residue classes
        residue_conflicts = [[0]*8 for _ in range(8)]
        for r1 in range(8):
            for r2 in range(8):
                if (r1 + r2) % 8 in {0, 1, 4}:
                    residue_conflicts[r1][r2] = 1
        
        current_set = set()
        
        # Try multiple residue orderings
        for _ in range(5):
            # Randomize residue order
            residues = list(range(8))
            random.shuffle(residues)
            
            trial_set = set()
            for r in residues:
                # Process vertices in this residue class
                group = residue_groups[r][:]
                random.shuffle(group)
                for v in group:
                    if not (adj[v] & trial_set):
                        trial_set.add(v)
            
            if len(trial_set) > len(current_set):
                current_set = trial_set
        
        return current_set

    def local_improve(s):
        """Try to add vertices that don't conflict."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if v in s:
                    continue
                if not (adj[v] & s):
                    s.add(v)
                    improved = True
        return s

    def two_swap_search(s):
        """
        2-swap local search: try swapping one vertex out and two in,
        or removing one and adding multiple.
        """
        s = set(s)
        s = local_improve(s)
        
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 2):
            improved = False
            
            # Try: remove v from s, then try to add multiple vertices
            for v in list(s):
                s.remove(v)
                s_new = local_improve(s)
                if len(s_new) > len(s) + 1:  # Net gain
                    s = s_new
                    improved = True
                    break
                else:
                    s.add(v)
                    s = local_improve(s)
            
            # Try: 2-swap - remove v1, v2, add w1, w2
            if not improved and (time.time() - start_time < DEADLINE - 3):
                s_list = list(s)
                random.shuffle(s_list)
                # Sample a few pairs to try
                for _ in range(min(50, len(s_list) * (len(s_list) - 1) // 2)):
                    v1 = random.choice(s_list)
                    v2 = random.choice([x for x in s_list if x != v1])
                    if v1 == v2:
                        continue
                    
                    s.remove(v1)
                    s.remove(v2)
                    s_new = local_improve(s)
                    
                    if len(s_new) > len(s) + 2:  # Net gain of at least 0 (removed 2, gained 2+)
                        s = s_new
                        improved = True
                        break
                    else:
                        s.add(v1)
                        s.add(v2)
                        s = local_improve(s)
        
        return s

    def local_search(s):
        """
        Combined local search: fill, then 2-swap.
        """
        s = set(s)
        s = local_improve(s)
        s = two_swap_search(s)
        return s

    # Iterated Local Search / Random Restart Greedy
    while time.time() - start_time < DEADLINE:
        # Mix different initialization strategies
        if random.random() < 0.3:
            s = modular_greedy()
        elif random.random() < 0.5:
            s = greedy_solve(use_degree_order=True)
        else:
            s = greedy_solve()
        
        s = local_improve(s)
        
        # Do deeper local search if we have time
        if time.time() - start_time < DEADLINE - 5:
            s = local_search(s)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = frozenset(s)
            
    # Convert 0-based indices back to 1-based values
    return sorted([v + 1 for v in best_set])

# EVOLVE-BLOCK-END
