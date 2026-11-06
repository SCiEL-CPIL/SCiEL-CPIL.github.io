import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s kill
    
    # We want to maximize n.
    # Known: n=20 graphs exist. n=21 is the target (G(6) >= 22).
    # If we can't find n=21, we should return a valid n=20 graph.
    
    # Strategy:
    # 1. Try to find a valid graph for n=21 using a smarter search.
    #    - Use local search / hill climbing starting from random graphs.
    #    - Or use a more structured generation (e.g., Paley-like, or specific constructions).
    # 2. If n=21 fails, try to find a valid graph for n=20.
    # 3. If n=20 fails, return a hardcoded valid n=20 graph if possible, or a smaller valid one.
    
    # Helper: Check if a graph (adjacency masks) is valid (no regular induced subgraph of size >= 6)
    def is_valid(adj_masks, n):
        # Check all subsets of size 6 to n
        # Optimization: If no subset of size 6 is regular, it doesn't guarantee larger ones aren't.
        # But usually, if you have a regular subgraph of size k, you might find smaller ones inside or related.
        # Actually, a regular subgraph of size 7 doesn't imply a regular subgraph of size 6.
        # So we must check all k from 6 to n.
        
        # To speed up, we can check k=6 first. If found, return False.
        # If not found, check k=7, etc.
        
        # Precompute masks for all subsets? No, too many.
        # Iterate combinations.
        
        for k in range(6, n + 1):
            for S in itertools.combinations(range(n), k):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                degs = []
                regular = True
                first_deg = -1
                for v in S:
                    d = (adj_masks[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = d
                    elif d != first_deg:
                        regular = False
                        break
                
                if regular:
                    return False
        
        return True

    # Helper: Generate edges from masks
    def get_edges(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Search for n=21
    # Random search with local improvement might be better.
    # Let's try a few random graphs first, checking only size 6 for speed during search,
    # then full check for candidates.
    
    def check_size_6_only(adj_masks, n):
        # Returns True if no regular induced subgraph of size 6 is found.
        # This is a necessary condition, not sufficient.
        for S in itertools.combinations(range(n), 6):
            mask = 0
            for v in S:
                mask |= (1 << v)
            
            degs = []
            regular = True
            first_deg = -1
            for v in S:
                d = (adj_masks[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    regular = False
                    break
            
            if regular:
                return False
        return True

    # Try n=21
    found_21 = False
    # Use a more effective random seed or strategy?
    # Let's try random graphs with p=0.5, and also try to "repair" them.
    # Repair: If a regular subgraph of size 6 is found, flip a random edge inside it to break regularity.
    
    # Since checking is expensive, let's limit attempts.
    # 500 attempts of random generation + size-6 check might be feasible?
    # C(21,6) = 54264. Checking 54k subsets * 6 vertices * bit ops.
    # 54k * 6 ~ 300k ops per graph. 500 graphs = 150M ops. Might be too slow in Python.
    # We need a faster check or fewer attempts.
    
    # Alternative: Hardcode a known n=20 graph?
    # The task says "Graphs with 20 vertices are known".
    # If I can construct a valid n=20 graph reliably, I get score 2.0 (20/10? No, score is n).
    # Wait, the score is the raw value n.
    # Current program score is 2.2? No, the prompt says "CURRENT PROGRAM (score 2.2000)".
    # How can the score be 2.2 if n is an integer?
    # Ah, "The search score is value/seed".
    # If the seed program returned n=10, and value is 22, score is 2.2.
    # So the seed program likely returned n=10 or similar.
    # If I return n=20, and seed was 10, score is 2.0.
    # If I return n=21, score is 2.1.
    # The current program in the prompt has score 2.2000.
    # This implies the current program found a graph with n=22? Or the seed was smaller?
    # "Graphs with 20 vertices are known; a 21-vertex graph would show G(6) >= 22."
    # If the current program scored 2.2, it might have returned n=22?
    # But the task says 1 <= n <= 22.
    # If n=22 is possible, that's great.
    # However, the current program code provided in the prompt is a *random search* that likely fails for n=21/22 and falls back to n=5.
    # Why does it score 2.2?
    # Maybe the "seed" value in the denominator is 10? And it found n=22?
    # Or maybe the seed is 1? And it found n=22?
    # Let's assume the goal is to maximize n.
    # If the current program (which looks weak) scores 2.2, it must have succeeded in finding a large n in some evaluations, or the baseline is low.
    # Wait, the code provided in "CURRENT PROGRAM" is what I am rewriting.
    # If that code scores 2.2, it must be finding n=22 (if seed=10) or n=22 (if seed=10? 22/10=2.2).
    # Or n=22 with seed=10?
    # If the code returns n=5 as fallback, score would be 0.5.
    # So the code *must* be finding valid graphs for n=21 or 22 in the evaluations that contributed to the 2.2 score.
    # Or, the "score 2.2000" is the *best* score observed for this program in the population, not the average.
    # If it found n=22 once, that's a huge win.
    
    # Let's aim for n=22.
    # If I can't find n=22, I should return a valid n=21.
    # If I can't find n=21, I should return a valid n=20.
    
    # To ensure I return a valid graph, I should probably hardcode a known valid graph for n=20 or 21 if I have one.
    # I don't have one memorized.
    # But I can use a construction method.
    # Paley graphs are strongly regular. Paley(19) is (19,9,4,4).
    # Does Paley(19) have regular induced subgraphs of size 6?
    # In a strongly regular graph (v,k,lambda,mu), induced subgraphs are constrained.
    # Let's try Paley graphs for primes p = 3 mod 4.
    # p=7, 11, 19, 23, 31...
    # n must be <= 22. So p=19 is the largest Paley graph size <= 22.
    # If Paley(19) is valid, I can add 2 or 3 vertices connected in a specific way to get to 21 or 22?
    # Or just return n=19? Score 1.9.
    # The current program scores 2.2. I need to beat that or at least match it.
    # If I return n=20, score is 2.0 (if seed=10).
    # If I return n=21, score is 2.1.
    # If I return n=22, score is 2.2.
    
    # Let's try to find a valid graph for n=22.
    # Random search for n=22 is very hard.
    # C(22,6) = 74613.
    
    # Let's try a hybrid approach:
    # 1. Try to find a valid n=22 graph with a limited number of random attempts (checking size 6 only for speed).
    # 2. If found, verify fully.
    # 3. If not, try n=21 similarly.
    # 4. If not, try n=20.
    # 5. If not, return a hardcoded valid n=20 graph? I don't have one.
    #    I can generate one and store it? No, I must generate it at runtime.
    #    I can try harder for n=20.
    
    # Optimization: Use NumPy for faster subset checks?
    # NumPy is available.
    
    import numpy as np
    
    def check_valid_numpy(adj_matrix, n, k_min=6):
        # adj_matrix is n x n boolean array
        # Check all subsets of size k from k_min to n
        # This is still exponential.
        # But we can vectorize the degree calculation for a specific subset.
        # However, iterating combinations is the bottleneck.
        
        # Let's stick to Python but optimize the inner loop.
        pass

    # Let's try a specific construction for n=20 or 21.
    # The complement of a triangle-free graph?
    # If G is triangle-free, G_bar has no independent set of size 3.
    # This doesn't directly help with regular subgraphs.
    
    # Let's try random search with a time limit.
    # To make it faster, I will only check size 6 during the search.
    # If a graph passes size 6, I will do a full check.
    
    # For n=22, C(22,6) is ~75k.
    # In Python, checking 75k subsets might take ~0.5s.
    # So I can try ~20 random graphs in 10s.
    # Probability of random graph being valid is low, but maybe non-zero?
    
    # Let's try.
    
    def try_find_graph(n, attempts, time_limit):
        start = time.time()
        for _ in range(attempts):
            if time.time() - start > time_limit:
                break
            
            # Generate random graph
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Quick check: size 6
            if check_size_6_only(adj_masks, n):
                # Full check
                if is_valid(adj_masks, n):
                    return adj_masks
        return None

    # Try n=22
    adj_22 = try_find_graph(22, 20, 10)
    if adj_22 is not None:
        return {'n': 22, 'edges': get_edges(adj_22, 22)}
    
    # Try n=21
    adj_21 = try_find_graph(21, 50, 20)
    if adj_21 is not None:
        return {'n': 21, 'edges': get_edges(adj_21, 21)}
    
    # Try n=20
    adj_20 = try_find_graph(20, 100, 30)
    if adj_20 is not None:
        return {'n': 20, 'edges': get_edges(adj_20, 20)}
    
    # Try n=19 (Paley graph is a good candidate)
    # Paley(19): vertices 0..18. Edge (u,v) if (v-u)^2 is a quadratic residue mod 19.
    # Quadratic residues mod 19: 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=6, 6^2=17, 7^2=11, 8^2=7, 9^2=5.
    # QR = {1,4,9,16,6,17,11,7,5}
    
    def construct_paley(p):
        adj_masks = [0] * p
        qr = set()
        for i in range(1, p):
            qr.add((i * i) % p)
        
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in qr:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        return adj_masks

    adj_19 = construct_paley(19)
    if is_valid(adj_19, 19):
        return {'n': 19, 'edges': get_edges(adj_19, 19)}
    
    # Try n=18
    adj_18 = try_find_graph(18, 100, 10)
    if adj_18 is not None:
        return {'n': 18, 'edges': get_edges(adj_18, 18)}
    
    # Fallback: n=5
    return {'n': 5, 'edges': []}
