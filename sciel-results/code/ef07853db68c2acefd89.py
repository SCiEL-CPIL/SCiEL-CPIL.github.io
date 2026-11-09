import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 16
    
    # Helper: Check if adding a triple violates linearity
    def is_linear(triples, new_triple):
        a, b, c = new_triple
        for t in triples:
            common = len(set([a, b, c]) & set(t))
            if common > 1:
                return False
        return True

    # Helper: Calculate independence number (approximate or exact for small N)
    # Since N=16, we can use a branch and bound or simple pruning for exact check
    # But for speed during SA, we might need a fast heuristic or exact but optimized.
    # Given N=16, exact check via bitmasks is feasible.
    
    # Precompute masks for triples
    def get_mask(triple):
        m = 0
        for p in triple:
            m |= (1 << p)
        return m

    # Exact independence number calculation for a set of triples
    # We want to find the max size S such that no triple is subset of S.
    # Equivalently, min vertex cover in the 3-uniform hypergraph.
    # For N=16, we can iterate over all subsets? 2^16 = 65536. That's fast.
    # Let's precompute the "bad" subsets? No, we check if a subset contains a triple.
    
    def calc_independence(triples):
        if not triples:
            return N
        
        # Create a set of forbidden masks (triples)
        # A subset S is independent if no triple is a subset of S.
        # We want max |S| such that for all t in triples, t is not subset of S.
        
        # Iterate from large sizes down? Or just check all?
        # 2^16 is 65536. Checking each subset against all triples might be slow if many triples.
        # Let's optimize: 
        # Represent triples as list of masks.
        # For a subset mask S, it is independent if for all tm in triple_masks: (S & tm) != tm.
        
        triple_masks = [get_mask(t) for t in triples]
        
        # We can prune: if a triple is subset of another, the larger one is redundant for "contains triple"?
        # No, if S contains a triple, it's not independent.
        
        # Let's just iterate all masks from 1 to 2^N-1?
        # Actually, we can be smarter.
        # Max independent set size.
        # We can use a recursive search with pruning, but for 16 points, brute force with bit ops is likely fast enough in Python if optimized?
        # 65536 iterations * (num_triples checks). If num_triples is ~100, that's 6.5M ops. Might be too slow in pure Python loop per SA step.
        
        # Alternative: Use a heuristic upper bound for SA, and only calculate exact for best candidates?
        # Or optimize the exact calculation.
        
        # Let's try a faster exact method:
        # Precompute for each point, which triples contain it?
        # Actually, let's just use a simple greedy heuristic for the SA cost function to make it fast,
        # and calculate the exact score at the end.
        # But we need a good cost function for SA.
        
        # Better: Use the number of "uncovered" pairs? No.
        # Let's stick to a fast approximation for the SA loop:
        # Independence number is at least the max number of points in any "component" if we view it as a graph? No.
        
        # Let's try to speed up the exact calculation.
        # We can precompute the list of triples as integers.
        # For each mask S, check if any triple is subset.
        # This is O(2^N * T).
        
        # Maybe we can limit the number of triples?
        # A linear system on 16 points can have at most C(16,2)/3 = 40 triples?
        # Actually, max size of LSTS(16) is not trivial but bounded.
        # If we keep the number of triples small, the check is fast.
        
        best_size = 0
        # Iterate subsets in descending order of popcount?
        # No, just iterate all.
        # To speed up:
        # If a triple is {a,b,c}, then any S containing a,b,c is invalid.
        
        # Let's use a different approach for the cost function in SA:
        # Minimize the number of "collisions" or something?
        # No, the objective is the independence number.
        
        # Let's use a randomized greedy for the independence number estimate during SA,
        # and exact check for the final result and occasional verification.
        
        # Actually, for N=16, we can precompute the "forbidden" property more efficiently?
        # Let's just use a simple heuristic:
        # Greedy removal of points to hit all triples. The number of points removed is a vertex cover.
        # N - (min vertex cover) = max independent set.
        # Greedy vertex cover:
        def greedy_vertex_cover(triples):
            points_to_remove = set()
            remaining_triples = list(triples)
            while remaining_triples:
                # Pick a triple with points most frequently appearing in remaining triples
                counts = {}
                for t in remaining_triples:
                    for p in t:
                        if p not in points_to_remove:
                            counts[p] = counts.get(p, 0) + 1
                if not counts:
                    break
                # Pick point with max count
                best_p = max(counts, key=counts.get)
                points_to_remove.add(best_p)
                # Remove all triples containing best_p
                remaining_triples = [t for t in remaining_triples if best_p not in t]
            return len(points_to_remove)
        
        # This is an approximation. The true independence number is <= N - greedy_vc.
        # It's a lower bound on the VC, so upper bound on IS?
        # Wait, VC covers all edges. IS is disjoint from VC? No.
        # IS is a set of vertices such that no edge is fully inside it.
        # VC is a set of vertices such that every edge has at least one vertex in it.
        # If V is the set of all vertices, and VC is a vertex cover, then V \ VC is an independent set.
        # So |V \ VC| is an independent set size.
        # Min VC gives Max IS.
        # Greedy gives a VC, so N - GreedyVC is a valid IS size (lower bound on max IS).
        # We want to MINIMIZE the max IS. So we want to MINIMIZE the size of the largest independent set.
        # So we want to MAXIMIZE the min vertex cover?
        # No, we want to find a system where the min vertex cover is as LARGE as possible,
        # which makes the max independent set as SMALL as possible.
        
        # So the cost function for SA should be: Minimize (N - GreedyVC) ?
        # No, GreedyVC is an approximation of MinVC.
        # If we use GreedyVC as a proxy for MinVC, then N - GreedyVC is a proxy for MaxIS.
        # We want to minimize MaxIS. So we minimize (N - GreedyVC) which is equivalent to maximizing GreedyVC.
        
        # Let's use this as the fast cost function.
        
        def fast_cost(triples):
            if not triples:
                return N
            vc_size = greedy_vertex_cover(triples)
            return N - vc_size
        
        # Initial construction: Greedy random
        # Start with empty list
        triples = []
        # Add random triples until we can't anymore or we have enough
        # Max possible is around 40-50?
        # Let's try to build a large linear system.
        
        # 1. Generate a random initial linear system
        def generate_initial_system():
            ts = []
            # Try to add as many triples as possible
            # Order points randomly
            points = list(range(N))
            random.shuffle(points)
            # This simple greedy might not be great.
            # Let's just pick random triples and check linearity.
            attempts = 0
            while attempts < 500:
                a, b, c = random.sample(range(N), 3)
                t = tuple(sorted([a, b, c]))
                if t in ts:
                    continue
                if is_linear(ts, t):
                    ts.append(t)
                attempts += 1
                if time.time() > deadline - 5:
                    break
            return ts
            
        # Run multiple restarts
        best_triples = []
        best_score = N # Max possible IS is 16
        
        for restart in range(20): # Up to 20 restarts
            if time.time() > deadline - 10:
                break
                
            current_triples = generate_initial_system()
            if not current_triples:
                continue
                
            current_score = fast_cost(current_triples)
            
            # Simulated Annealing
            T = 10.0
            T_min = 0.1
            alpha = 0.999
            steps = 5000
            
            for i in range(steps):
                if time.time() > deadline - 5:
                    break
                    
                # Move: Pick a random triple, replace it with a random new triple
                if not current_triples:
                    break
                    
                idx = random.randint(0, len(current_triples) - 1)
                old_t = current_triples[idx]
                
                # Generate new triple
                new_t = None
                for _ in range(10):
                    a, b, c = random.sample(range(N), 3)
                    cand = tuple(sorted([a, b, c]))
                    # Check if cand is already in list (except at idx)
                    # And check linearity with all other triples
                    temp_list = current_triples[:idx] + current_triples[idx+1:]
                    if cand not in temp_list and is_linear(temp_list, cand):
                        new_t = cand
                        break
                
                if new_t is None:
                    continue
                
                current_triples[idx] = new_t
                new_score = fast_cost(current_triples)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_triples = current_triples[:]
                else:
                    # Accept with probability
                    delta = new_score - current_score
                    if delta < 0:
                        accept = True
                    else:
                        accept = random.random() < pow(2.71828, -delta / T)
                    
                    if accept:
                        current_score = new_score
                    else:
                        current_triples[idx] = old_t
                        
                T *= alpha
                if T < T_min:
                    T = T_min
                    
            # End SA
            # Verify best score with exact calculation?
            # For now, we trust the fast cost for selection, but we should return the exact score.
            # The task says "score is the largest point set containing no triple".
            # So we must return the system that minimizes the EXACT independence number.
            # The fast cost is a proxy. It might be misleading.
            # However, minimizing the proxy (N - GreedyVC) is correlated with minimizing MaxIS.
            
            # Let's do a final exact check on the best_triples found so far?
            # No, we should check the current_triples at the end of each restart?
            # Or just keep the one with the best fast_score, and at the very end,
            # maybe do a local search with exact cost?
            
            # Given time constraints, let's assume fast_cost is a good proxy.
            # But to be safe, we can calculate exact IS for the best candidate at the end.
            
        # Final Exact Check and Optimization
        # We have best_triples from the proxy optimization.
        # Let's try to improve it with exact cost?
        # Exact cost is slow.
        
        # Let's define exact independence number calculation.
        def exact_independence(triples):
            if not triples:
                return N
            triple_masks = [get_mask(t) for t in triples]
            # Iterate all subsets
            # To speed up, we can prune.
            # If a subset S is independent, any subset of S is independent.
            # We want max size.
            # We can iterate by size descending?
            # Or just check all 2^16 masks.
            max_size = 0
            # Optimization: If a mask S contains a triple mask, it's not independent.
            # We can precompute a "bad" array? No.
            # Just loop.
            for mask in range(1, 1 << N):
                # Check if independent
                is_indep = True
                for tm in triple_masks:
                    if (mask & tm) == tm:
                        is_indep = False
                        break
                if is_indep:
                    size = bin(mask).count('1')
                    if size > max_size:
                        max_size = size
            return max_size
            
        # Let's check the exact score of best_triples
        if best_triples:
            exact_best = exact_independence(best_triples)
            # If exact_best is better than what we think?
            # We might have a system with low proxy score but high exact score?
            # Unlikely, but possible.
            
            # Let's try a few local moves with exact cost if time permits
            # But exact cost is ~65536 * T ops. If T=50, that's 3M ops.
            # In Python, 3M ops is ~0.1s? Maybe 0.5s.
            # We have ~10s left?
            # Let's do a limited local search with exact cost.
            
            current = best_triples[:]
            current_exact = exact_best
            
            # Local Search
            for i in range(100): # 100 steps
                if time.time() > deadline - 2:
                    break
                if not current:
                    break
                idx = random.randint(0, len(current) - 1)
                old_t = current[idx]
                new_t = None
                for _ in range(10):
                    a, b, c = random.sample(range(N), 3)
                    cand = tuple(sorted([a, b, c]))
                    temp_list = current[:idx] + current[idx+1:]
                    if cand not in temp_list and is_linear(temp_list, cand):
                        new_t = cand
                        break
                if new_t is None:
                    continue
                current[idx] = new_t
                new_exact = exact_independence(current)
                if new_exact < current_exact:
                    current_exact = new_exact
                    best_triples = current[:]
                    exact_best = new_exact
                else:
                    current[idx] = old_t
                    
        # If no triples found, return empty?
        # The seed returned 5 disjoint triples.
        # If we fail to construct any, return seed?
        if not best_triples:
             return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
             
        return [list(t) for t in best_triples]

# Keep the function name exactly as construct
