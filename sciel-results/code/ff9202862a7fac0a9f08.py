import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate candidates
    # Weight 2: 220 vectors
    # Weight 3: 1320 vectors
    # Weight 1: 22 vectors
    
    candidates = []
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
                
    # Weight 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
                    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # We want to select a subset S such that min dist >= max norm.
    # Let's assume we primarily use Weight 3 vectors, as they offer the most potential.
    # If we select only W3 vectors, R = sqrt(3).
    # Condition: u . v <= 1 for all u, v in S.
    
    # Let's compute dot products for all candidates?
    # n_cand is 22 + 220 + 1320 = 1562.
    # 1562^2 dot products. ~2.4M ops. Fast.
    
    dot_prods = candidates @ candidates.T
    norms_sq = np.sum(candidates**2, axis=1)
    
    # We will try different strategies:
    # 1. Greedy on W3 only.
    # 2. Greedy on W3 + W2.
    # 3. Greedy on all.
    
    best_score = 0
    best_vectors = np.array([], dtype=np.int64).reshape(0, D)
    
    # Pre-filter: 
    # If we pick a set with max norm R, all pairs must satisfy dist^2 >= R^2.
    # dist^2 = norm_u^2 + norm_v^2 - 2 dot.
    # norm_u^2 + norm_v^2 - 2 dot >= R^2.
    
    # Strategy: Iterative Greedy with Randomization
    # We will try to build a set starting from the largest norm vectors to fix R early?
    # Or start from small?
    # If we start with W3, R becomes sqrt(3). Then we check W2, W1.
    
    # Let's identify indices by weight
    w1_idx = np.where(norms_sq == 1)[0]
    w2_idx = np.where(norms_sq == 2)[0]
    w3_idx = np.where(norms_sq == 3)[0]
    
    # We will run multiple random greedy passes
    np.random.seed(42)
    
    for iteration in range(50):
        if time.time() > deadline:
            break
            
        # Randomize order of W3
        order_w3 = np.random.permutation(w3_idx)
        
        S = []
        S_set = set()
        
        # 1. Add W3 vectors
        for idx in order_w3:
            # Check against all in S
            if not S:
                S.append(idx)
                S_set.add(idx)
                continue
                
            # Efficient check:
            # We need dot[u, idx] <= (norm_u^2 + norm_idx^2 - R^2) / 2 ?
            # No, R is not fixed yet.
            # If we assume we only add W3, R=sqrt(3).
            # Condition: norm_u^2 + 3 - 2 dot >= 3 => norm_u^2 - 2 dot >= 0 => dot <= norm_u^2 / 2.
            # For W3, norm^2=3. dot <= 1.5 => dot <= 1.
            # So if S contains only W3, we just need dot <= 1.
            
            # But what if we want to mix?
            # Let's stick to pure W3 first to get a large base.
            
            # Check dot products with all u in S
            # S is a list of indices.
            # We can vectorize this check.
            
            if len(S) > 0:
                dots = dot_prods[S, idx]
                # If any dot > 1, conflict.
                if np.any(dots > 1):
                    continue
                    
            S.append(idx)
            S_set.add(idx)
            
        # Now we have a set of W3 vectors.
        # Can we add W2?
        # If S has W3, R = sqrt(3).
        # For w in W2, u in S(W3):
        # norm_w^2 + norm_u^2 - 2 dot >= 3
        # 2 + 3 - 2 dot >= 3 => 2 - 2 dot >= 0 => dot <= 1.
        # So we need w . u <= 1 for all u in S.
        
        # Check W2 candidates
        order_w2 = np.random.permutation(w2_idx)
        for idx in order_w2:
            if not S:
                S.append(idx)
                S_set.add(idx)
                continue
            dots = dot_prods[S, idx]
            # Condition: dot <= 1
            if np.any(dots > 1):
                continue
            S.append(idx)
            S_set.add(idx)
            
        # Can we add W1?
        # R = sqrt(3) (assuming S not empty and has W3)
        # For w in W1, u in S:
        # 1 + norm_u^2 - 2 dot >= 3
        # norm_u^2 - 2 - 2 dot >= 0
        # If u is W3 (norm^2=3): 3 - 2 - 2 dot >= 0 => 1 - 2 dot >= 0 => dot <= 0.5 => dot <= 0.
        # If u is W2 (norm^2=2): 2 - 2 - 2 dot >= 0 => dot <= 0.
        # So w . u <= 0 for all u in S.
        
        order_w1 = np.random.permutation(w1_idx)
        for idx in order_w1:
            if not S:
                S.append(idx)
                S_set.add(idx)
                continue
            dots = dot_prods[S, idx]
            # Condition: dot <= 0
            if np.any(dots > 0):
                continue
            S.append(idx)
            S_set.add(idx)
            
        if len(S) > best_score:
            best_score = len(S)
            best_vectors = candidates[S]
            
        # Try a different strategy: Start with W2?
        # If we start with W2, R=sqrt(2).
        # Then add W3?
        # If S has W2, R=sqrt(2).
        # For w in W3, u in S(W2):
        # 3 + 2 - 2 dot >= 2 => 3 - 2 dot >= 0 => dot <= 1.5 => dot <= 1.
        # So W3 can be added if dot <= 1 with all W2s.
        
        S2 = []
        order_w2_2 = np.random.permutation(w2_idx)
        for idx in order_w2_2:
            if not S2:
                S2.append(idx)
                continue
            dots = dot_prods[S2, idx]
            # Condition for W2-W2: dot <= 1 (since R=sqrt(2), 2+2-2dot >= 2 => dot <= 1)
            if np.any(dots > 1):
                continue
            S2.append(idx)
            
        # Add W3 to S2
        order_w3_2 = np.random.permutation(w3_idx)
        for idx in order_w3_2:
            if not S2:
                S2.append(idx)
                continue
            dots = dot_prods[S2, idx]
            # Condition: dot <= 1
            if np.any(dots > 1):
                continue
            S2.append(idx)
            
        # Add W1 to S2
        # R is now max(norm in S2). If S2 has W3, R=sqrt(3). If only W2, R=sqrt(2).
        # Let's calculate R for S2
        if S2:
            max_norm_sq = np.max(norms_sq[S2])
            # Condition for W1:
            # 1 + norm_u^2 - 2 dot >= max_norm_sq
            # 1 + norm_u^2 - max_norm_sq >= 2 dot
            # dot <= (1 + norm_u^2 - max_norm_sq) / 2
            
            order_w1_2 = np.random.permutation(w1_idx)
            for idx in order_w1_2:
                if not S2:
                    S2.append(idx)
                    continue
                # Check against all u in S2
                # We need to check individually because norm_u varies
                valid = True
                for u_idx in S2:
                    dot = dot_prods[idx, u_idx]
                    norm_u_sq = norms_sq[u_idx]
                    thresh = (1 + norm_u_sq - max_norm_sq) / 2.0
                    if dot > thresh:
                        valid = False
                        break
                if valid:
                    S2.append(idx)
                    
            if len(S2) > best_score:
                best_score = len(S2)
                best_vectors = candidates[S2]

    if best_score == 0:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return best_vectors
