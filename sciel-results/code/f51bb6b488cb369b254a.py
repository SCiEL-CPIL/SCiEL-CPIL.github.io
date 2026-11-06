"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    # Precompute neighbors for all vertices in Q_N
    V = 1 << N
    neighbors = [[] for _ in range(V)]
    for v in range(V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    # Precompute adjacency matrix for fast bit_count check? 
    # For SAT, we need to encode constraints.
    # Variables: x_v for v in 0..V-1 indicating if v is in the snake.
    # We want to maximize sum(x_v).
    # Constraints for induced path:
    # 1. If x_u and x_v are both 1 and u~v (adjacent), they must be consecutive in the path.
    #    This is tricky to encode directly in SAT without ordering variables.
    #    Alternative: Use the fact that an induced path in a hypercube is a sequence.
    #    Let's try a different formulation:
    #    A set of vertices S is an induced path iff:
    #    a) The subgraph induced by S is a path.
    #    b) There are no chords (edges between non-consecutive vertices in S).
    
    # Since N=9, V=512. A SAT solver on 512 variables might be slow if not well-constrained.
    # However, we can use a heuristic: start with a known good snake (from greedy) and try to extend it or find a better one via local search on the SAT model?
    # Actually, implementing a full SAT solver for maximization (MaxSAT) is complex.
    
    # Let's pivot to a different structural change: 
    # The current program uses greedy extension + local search (detours).
    # The "SAT" idea in the prompt was one option. 
    # Another option: **Simulated Annealing** on the permutation of vertices? No, induced path is not a permutation.
    # How about **Genetic Algorithm** on the set of vertices?
    # Or **Beam Search** with a better heuristic?
    
    # The prompt specifically suggested: "Model the snake as a Hamiltonian path in the directed line graph... and use a SAT solver".
    # Let's implement a simplified DPLL-based search for the longest path, but given the 120s limit and Python speed, 
    # a pure Python SAT solver on 512 variables with complex constraints might timeout.
    
    # Let's try a **Stochastic Local Search (SLK)** with a different move set: 
    # Instead of just detours, we allow **reversals** (reversing a segment) and **relocations** (moving a vertex).
    # But induced path constraints are strict.
    
    # Let's stick to the **SAT-inspired** approach but simplified:
    # We will use a **backtracking search with strong pruning** (not just greedy, but branching with lookahead).
    # The current program does greedy. We will do **DFS with branching** on the first few steps, then greedy, 
    # or **Iterative Deepening**.
    
    # Actually, the most robust structural change for this specific problem in Python with 120s is 
    # **Improved Local Search with more aggressive moves**:
    # 1. **Segment Reversal**: Reverse a subpath. This preserves induced path property?
    #    If P is an induced path, is P_reversed an induced path? Yes.
    #    Does reversing a segment preserve global induced path property?
    #    Let P = A + B + C. Reverse B to get B'. 
    #    Connections: A_last - B'_first (was A_last - B_last). 
    #    B'_last - C_first (was B_first - C_first).
    #    We need to ensure no new adjacencies are created between B' and A/C.
    #    Since B was induced, B' is induced.
    #    Adjacencies between B' and A: 
    #    Originally, B_first was adjacent to A_last. B_last was not adjacent to any A except maybe A_last? No, A_last is only adjacent to B_first in the path.
    #    Wait, in an induced path, non-consecutive vertices are NOT adjacent.
    #    So A_last is only adjacent to B_first in the path? No, A_last is adjacent to B_first in the path.
    #    Are there other adjacencies? No, by definition of induced path.
    #    So if we reverse B, B'_first is B_last. 
    #    Is B_last adjacent to A_last? 
    #    In the original path, B_last was non-consecutive to A_last (unless B has length 1).
    #    So B_last is NOT adjacent to A_last.
    #    Therefore, reversing B breaks the connection A_last - B_first if B_last != B_first.
    #    So simple reversal of an internal segment does NOT preserve the path connectivity.
    
    # So we are stuck with **Detours** (replacing a segment with a longer path between the same endpoints) 
    # or **Extensions** (adding to the end).
    
    # The current program does detours. 
    # Let's try **Multi-Start with Elite Crossover**:
    # Combine parts of two good snakes? Hard to ensure validity.
    
    # Let's try **Iterative Deepening A* (IDA*)** for the path construction?
    # Or **Beam Search** with a wider beam and better heuristic.
    
    # Heuristic: For a vertex v, score = (number of unused neighbors) - (penalty for being adjacent to used non-consecutive vertices).
    
    # Let's implement **Beam Search** instead of pure Greedy.
    # Beam width B.
    # State: (path, used_set).
    # At each step, expand all states in beam, keep top B.
    
    # And then **Local Search** on the best beam result.
    
    def beam_search(beam_width=50, max_depth=None):
        # Start with beam of all single vertices? No, start with 0.
        # Or start with a few random starts?
        # To be efficient, start with 0.
        beam = [( [0], {0} )]
        
        for depth in range(1, max_depth or 200):
            if time.time() > deadline:
                break
            next_beam = []
            for path, used in beam:
                cur = path[-1]
                # Generate candidates
                candidates = []
                for b in range(N):
                    v = cur ^ (1 << b)
                    if v in used:
                        continue
                    # Check validity
                    valid = True
                    for i in range(len(path) - 1):
                        if (v ^ path[i]).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        # Heuristic: count future options
                        # Score = number of valid extensions from v
                        # This is expensive. Use simple heuristic: 
                        # Score = - (number of used neighbors of v) ? 
                        # Or just random?
                        # Let's use a simple score: 0. 
                        # But we need to diversify.
                        # Let's keep all valid candidates, then prune.
                        candidates.append(v)
                        
            # This loop structure is wrong. We need to expand each state in the beam.
            pass # Re-implementation below

    # Correct Beam Search Implementation
    def beam_search_impl(beam_width=100):
        beam = [( [0], {0} )]
        best_path = [0]
        
        while True:
            if time.time() > deadline:
                break
            if not beam:
                break
            
            next_beam = []
            for path, used in beam:
                cur = path[-1]
                for b in range(N):
                    v = cur ^ (1 << b)
                    if v in used:
                        continue
                    # Check validity against path[:-1]
                    valid = True
                    for i in range(len(path) - 1):
                        if (v ^ path[i]).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        new_path = path + [v]
                        new_used = used | {v}
                        # Heuristic: Prefer paths that have more "open" ends?
                        # Or just keep all.
                        next_beam.append((new_path, new_used))
            
            if not next_beam:
                break
            
            # Sort by length (all same length in beam), then by some diversity metric?
            # Since all paths in beam have same length, we can just sample or pick top by a score.
            # Score: Sum of degrees of unused neighbors?
            def score(path, used):
                cur = path[-1]
                score_val = 0
                for b in range(N):
                    nb = cur ^ (1 << b)
                    if nb not in used:
                        # Check if nb is a valid extension
                        valid = True
                        for i in range(len(path) - 1):
                            if (nb ^ path[i]).bit_count() == 1:
                                valid = False
                                break
                        if valid:
                            score_val += 1
                return score_val
            
            # To save time, only score a subset if beam is large?
            # Or just random selection if too slow.
            # Let's try to sort by score.
            # But scoring is O(L) for each candidate.
            
            # Optimization: 
            # If next_beam is too large, sample.
            if len(next_beam) > beam_width * 2:
                # Sample
                next_beam = random.sample(next_beam, beam_width * 2)
                
            # Sort by score (descending)
            # To avoid full sort cost, we can just pick top K.
            # Let's just random sample if beam_width is small enough.
            # Actually, for beam search, we usually keep the top K.
            
            # Let's use a simple heuristic: 
            # Score = - (number of used neighbors of cur) 
            # This is O(1) if we maintain used_neighbors count?
            # Too complex.
            
            # Let's just random sample from next_beam to get diversity.
            # This turns it into a random walk with beam width.
            if len(next_beam) > beam_width:
                beam = random.sample(next_beam, beam_width)
            else:
                beam = next_beam
            
            # Track best
            for p, u in beam:
                if len(p) > len(best_path):
                    best_path = p
                    
        return best_path

    def local_search(path, max_iterations=5000):
        best_path = list(path)
        best_len = len(best_path)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            if len(best_path) < 2:
                break
                
            # Pick a random segment to detour
            segment_len = random.randint(1, min(4, len(best_path)-1))
            i = random.randint(0, len(best_path) - segment_len - 1)
            j = i + segment_len
            
            start_v = best_path[i]
            end_v = best_path[j]
            
            forbidden = set(best_path[:i]) | set(best_path[j:])
            
            # DFS for detour
            def dfs_detour(cur, path_so_far, depth_limit):
                if len(path_so_far) > segment_len:
                    return path_so_far
                if len(path_so_far) >= depth_limit:
                    return None
                if cur == end_v:
                    return None
                
                for b in range(N):
                    nxt = cur ^ (1 << b)
                    if nxt in forbidden:
                        continue
                    if nxt in path_so_far:
                        continue
                    
                    is_adj_to_forbidden = False
                    for v in forbidden:
                        if (nxt ^ v).bit_count() == 1:
                            is_adj_to_forbidden = True
                            break
                    if is_adj_to_forbidden:
                        continue
                    
                    res = dfs_detour(nxt, path_so_far + [nxt], depth_limit)
                    if res:
                        return res
                return None

            depth_limit = segment_len + 4
            found_path = dfs_detour(start_v, [start_v], depth_limit)
            
            if found_path and len(found_path) > segment_len + 1:
                new_path = best_path[:i] + found_path + best_path[j:]
                if len(new_path) > best_len:
                    best_path = new_path
                    best_len = len(new_path)
        
        return best_path

    best_path = []
    
    # Phase 1: Beam Search
    # Run multiple beam searches with different widths
    for bw in [10, 50, 100]:
        if time.time() > deadline:
            break
        p = beam_search_impl(bw)
        if len(p) > len(best_path):
            best_path = p
            
    # Phase 2: Local Search on best
    if time.time() < deadline:
        improved = local_search(best_path)
        if len(improved) > len(best_path):
            best_path = improved
            
    # Phase 3: More local search if time
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            # Random restart local search
            p = beam_search_impl(10)
            improved = local_search(p, max_iterations=500)
            if len(improved) > len(best_path):
                best_path = improved

    return best_path
# EVOLVE-BLOCK-END
