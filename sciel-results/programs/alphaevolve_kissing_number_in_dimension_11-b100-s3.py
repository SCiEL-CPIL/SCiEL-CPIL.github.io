"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # 1. Generate Candidates: Weight-4 vectors
    # C(11, 4) * 16 = 330 * 16 = 5280
    candidates = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)

    # 2. Build Conflict Graph
    # Conflict if dot product > 2
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    # Build adjacency lists
    neighbors = [[] for _ in range(n_cands)]
    for i in range(n_cands):
        js = np.where(G[i] > 2)[0]
        js = js[js != i]
        neighbors[i] = js.tolist()
        
    # Pre-compute initial degrees
    init_deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)

    # 3. Greedy Independent Set Strategies
    
    rng = np.random.default_rng(42)
    best_set = np.array([], dtype=np.int64)
    best_size = 0
    
    def run_greedy_strategy(strategy="min_deg", seed_offset=0):
        """
        Runs a greedy independent set algorithm.
        strategy: 
            'min_deg': Pick node with minimum degree.
            'min_deg_noise': Pick node with degree close to min (min_deg + 0 or 1).
            'spectral_min': Pick node with min degree, tie-break by lowest spectral centrality (if available).
            'spectral_max': Pick node with min degree, tie-break by highest spectral centrality.
        """
        if time.time() > deadline - 5:
            return None
            
        deg = init_deg.copy()
        selected = []
        remaining = set(range(n_cands))
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            # Find min degree among remaining
            # Optimization: if remaining is small, direct min is fine.
            # For 5000 nodes, scanning is O(N). 500 steps * 5000 = 2.5M ops. Acceptable.
            min_deg = min(deg[i] for i in remaining)
            
            # Get all nodes with min degree (or min_deg+1 for noise strategy)
            if strategy == "min_deg_noise":
                # Allow degree min_deg or min_deg + 1
                candidates = [i for i in remaining if deg[i] <= min_deg + 1]
            else:
                candidates = [i for i in remaining if deg[i] == min_deg]
            
            # Selection logic
            if strategy == "min_deg":
                u = rng.choice(candidates)
            elif strategy == "min_deg_noise":
                # Weighted choice: prefer min_deg over min_deg+1?
                # Simple: uniform among candidates is a good start.
                # Or: 80% min_deg, 20% min_deg+1?
                # Let's stick to uniform for now, but the set is larger.
                u = rng.choice(candidates)
            elif strategy in ["spectral_min", "spectral_max"]:
                # Use spectral centrality for tie-breaking
                # We need v_principal. Pass it or compute it? 
                # For simplicity in this combined block, we'll use a proxy: 
                # Actually, let's just use random for now in the main loop 
                # unless we specifically want spectral.
                # The prompt asks to combine. Let's assume 'spectral' strategies
                # use a pre-computed vector if available, else fallback to random.
                # To keep it simple and fast, we will implement a specific spectral run
                # outside this generic function if needed, or pass the vector.
                # Let's modify the function signature to accept a tie_breaker vector.
                # But for now, let's just use random. The "spectral" improvement
                # in the current program was actually worse (2.318 vs 2.414).
                # So we should prioritize the "noise" strategy from inspiration 1
                # which scored 2.414.
                
                # Re-evaluating: Inspiration 1 (2.414) used "min_deg" and "min_deg_noise" (degree <= min+1).
                # Current program (2.318) added spectral sorting and it got WORSE.
                # Therefore, the strongest idea is the "noise" strategy from Inspiration 1.
                # The "structural idea" of the current program is the spectral embedding, but it failed.
                # The directive says: "Combine the strategy of the current program with the strongest ideas of the inspirations".
                # Strongest idea: Noise/Exploration (Inspiration 1).
                # Current program's unique element: Spectral calculation (which hurt).
                # Maybe we can use spectral to *initialize* the random seed or ordering?
                # Or maybe the current program's spectral code is just dead weight that slowed it down?
                # Let's discard the spectral sorting for selection and focus on the Noise strategy,
                # but perhaps use spectral to *order* the initial graph or something?
                # Actually, the best move is to stick to the winning strategy (Noise) and ensure
                # we run many restarts.
                
                u = rng.choice(candidates)
            else:
                u = rng.choice(candidates)
            
            selected.append(u)
            remaining.remove(u)
            
            # Remove neighbors
            for v in neighbors[u]:
                if v in remaining:
                    remaining.remove(v)
                    for w in neighbors[v]:
                        if w in remaining:
                            deg[w] -= 1
                            
        return selected

    # Run multiple restarts with the winning "noise" strategy and standard "min_deg"
    # We will alternate between them.
    n_restarts = 200 # High count, guarded by time
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # 50% chance of noise strategy, 50% min_deg
        if r % 2 == 0:
            result = run_greedy_strategy("min_deg", r)
        else:
            result = run_greedy_strategy("min_deg_noise", r)
            
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_set = cand_arr[result].copy()
            
    # Fallback
    if best_size == 0:
        best_set = cand_arr[:1]
        best_size = 1

    return best_set

# EVOLVE-BLOCK-END
