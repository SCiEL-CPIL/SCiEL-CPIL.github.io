import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    t0 = time.time()
    deadline = t0 + 580  # Safety margin
    
    def elapsed():
        return time.time() - t0
    
    # Generate candidate vectors by norm level
    # For each norm squared value, generate all integer vectors with that norm
    # Then do greedy selection
    
    # Let's try a different structural approach:
    # Use the fact that we can have points at different norms if cross-distances work.
    # 
    # Strategy: Start with a core set, then try to add points from larger norm shells.
    # 
    # Actually, let me try a completely different structural approach:
    # Use a lattice code. The kissing number in 11D over reals is 556 (Leech lattice related).
    # For integer points, we need a different approach.
    #
    # New idea: Use vectors from the E8 x A3 lattice or similar.
    # E8 has 240 minimal vectors of norm 2. In 8D.
    # 
    # Let me try: construct points using a product structure.
    # Split 11D into 8D + 3D. Use E8-like structure in 8D and A3 in 3D.
    #
    # Actually, simplest strong approach: use all vectors of a specific norm that form a valid set,
    # then try to augment with vectors from other norms using vectorized checks.
    
    # Phase 1: Get the 440 norm-4 Type B vectors
    pts_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    pts_list.append(v)
    
    pts = np.array(pts_list, dtype=np.int64)
    n = len(pts)
    max_norm = 4
    
    # Phase 2: Try to add vectors of norm 1 (unit vectors)
    # Check if any unit vector has dist² >= 4 to all existing points
    if elapsed() < deadline - 100:
        unit_vecs = []
        for i in range(d):
            for s in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = s
                unit_vecs.append(v)
        unit_vecs = np.array(unit_vecs, dtype=np.int64)
        
        # For each unit vector, compute min dist² to all current points
        # dist²(u, p) = 1 + 4 - 2*u·p = 5 - 2*u·p
        # Need 5 - 2*u·p >= 4 for all p, i.e., u·p <= 0.5, i.e., u·p <= 0
        # u·p: u has one ±1, p has two ±1s. 
        # If u's position matches one of p's positions: u·p = ±1
        # If u's position doesn't match: u·p = 0
        # So we need u·p <= 0 for all p. But there exist p with u·p = 1. So NO unit vector works.
        pass
    
    # Phase 3: Try norm 9 vectors
    # Type: (±3, 0, ...) - 22 vectors
    # dist²((3,0,..), (1,1,0,..)) where they share pos 0: (2,-1,0,..) norm²=5 < 9. Bad.
    # Type: (±2, ±1, 0, ...) - 11*10*4 = 440 vectors  
    # dist²((2,1,0,..), (1,1,0,..)): (1,0,0,..) norm²=1 < 9. Bad.
    # Type: (±1, ±1, ±1, 0, ...) - C(11,3)*8 = 1320 vectors
    # dist²((1,1,1,0,..), (1,1,0,0,..)): (0,0,1,0,..) norm²=1 < 9. Bad.
    # So no norm-9 vector can be added to the norm-4 set.
    
    # Phase 4: Try a completely different base - use norm 1 only but with more structure?
    # No, norm 1 gives only 22.
    
    # Phase 5: What if we use a different norm-4 subset plus other norms?
    # Actually, let me try: can we use ALL 462 norm-4 vectors (both types)?
    # We showed mixing types gives dist²=2 < 4. So no.
    
    # Phase 6: Try norm 2 vectors: (±1, ±1, 0, ...) is norm 2? No, that's norm 2.
    # Wait, (1,1,0,...) has norm 1+1=2. I was confusing.
    # Let me recheck: (1,1,0,...) norm² = 1+1 = 2, not 4!
    
    # I made an error! Let me redo.
    # (±1, ±1, 0, ...) has norm² = 2.
    # (±2, 0, ...) has norm² = 4.
    # (±1, ±1, ±1, 0, ...) has norm² = 3.
    
    # So for norm² = 2: 11*10*4 = 440 vectors of type (±1,±1,0,...)
    # Min dist between two such: (1,1,0,..) and (1,-1,0,..) -> (0,2,0,..) norm²=4 >= 2. ✓
    # (1,1,0,..) and (-1,1,0,..) -> (2,0,0,..) norm²=4 >= 2. ✓
    # (1,1,0,..) and (1,0,1,..) -> (0,1,-1,..) norm²=2 >= 2. ✓
    # (1,1,0,..) and (0,1,1,..) -> (1,0,-1,..) norm²=2 >= 2. ✓
    # So all 440 work with max_norm=2.
    
    # Can we add norm²=1 vectors (unit vectors)?
    # dist²((1,0,..), (1,1,0,..)) = (0,-1,0,..) norm²=1 < 2. Bad.
    # dist²((1,0,..), (0,1,0,..)) = (1,-1,0,..) norm²=2 >= 2. ✓
    # But (1,0,..) and (1,1,0,..) share coordinate 0, giving dist²=1. So no unit vector works.
    
    # Can we add norm²=4 vectors?
    # (2,0,..) and (1,1,0,..): dist²=(1,-1,0,..)=2 < 4. Bad.
    # (2,0,..) and (0,1,1,..): dist²=(2,-1,-1,..)=6 >= 4. ✓
    # But (2,0,..) and (1,1,0,..) fails. So no.
    
    # What about norm²=3 vectors (±1,±1,±1,0,...)?
    # C(11,3)*8 = 1320 vectors.
    # Min dist within: (1,1,1,0,..) and (1,1,-1,0,..) -> (0,0,2,0,..) norm²=4 >= 3. ✓
    # (1,1,1,0,..) and (1,-1,1,0,..) -> (0,2,0,0,..) norm²=4 >= 3. ✓
    # (1,1,1,0,..) and (1,1,0,1,..) -> (0,0,1,-1,..) norm²=2 < 3. BAD!
    # So not all 1320 work. Need to be more careful.
    
    # Let me just verify the 440 norm²=2 vectors work and return them.
    # That's the same as before but with correct norm understanding.
    
    # Actually wait - my original code had norm²=4 for (±1,±1,0,...) which is WRONG.
    # (1,1,0,...) has norm² = 1+1 = 2.
    # So the original code's comment was wrong but the vectors were correct.
    # The max_norm would be 2, not 4.
    # And min_dist = 2 (e.g., (1,1,0,..) and (1,0,1,..) -> dist²=2).
    # So min_dist = 2 = max_norm. Valid!
    
    # So we have 440 points with max_norm=2. Can we do better?
    
    # Try adding norm²=4 vectors to the norm²=2 set.
    # Need dist² >= 4 (since max_norm becomes 4).
    # (2,0,..) and (1,1,0,..) sharing pos 0: dist² = 1+1=2 < 4. Bad.
    # (2,0,..) and (0,1,1,..) not sharing: dist² = 4+1+1=6 >= 4. ✓
    # But any (2,0,..) shares a coordinate with some (1,1,..) vectors.
    # Specifically, (2,0,..) at pos i shares with all (1,±1,0,..) at pos i.
    # There are 2*(d-1) such vectors. All give dist²=2 < 4. So no.
    
    # Try a different approach: use ONLY norm²=4 vectors of type (±2,0,...)
    # 22 vectors. Min dist: (2,0,..) and (-2,0,..) -> 16. (2,0,..) and (0,2,..) -> 8.
    # Max norm = 4. Min dist = 8 >= 4. Valid. But only 22 points.
    
    # What about mixing (±2,0,..) with some (±1,±1,0,..) that don't conflict?
    # If we drop all (1,±1,0,..) that share a coordinate with our (2,0,..) vectors...
    # This gets complex. Let me just go with 440 for now and try to find a better construction.
    
    # NEW STRUCTURAL IDEA: Use a different norm level entirely.
    # Norm²=8: vectors like (±2,±2,0,...) norm²=8. 11*10*4=440 vectors.
    # Min dist: (2,2,0,..) and (2,-2,0,..) -> (0,4,0,..) norm²=16 >= 8. ✓
    # (2,2,0,..) and (2,0,2,..) -> (0,2,-2,..) norm²=8 >= 8. ✓
    # (2,2,0,..) and (-2,2,0,..) -> (4,0,0,..) norm²=16 >= 8. ✓
    # So all 440 work! Same count as norm²=2.
    
    # Norm²=12: (±2,±2,±2,0,...) norm²=12. C(11,3)*8=1320 vectors.
    # (2,2,2,0,..) and (2,2,-2,0,..) -> (0,0,4,0,..) norm²=16 >= 12. ✓
    # (2,2,2,0,..) and (2,2,0,2,..) -> (0,0,2,-2,..) norm²=8 < 12. BAD.
    
    # So norm²=8 gives 440, same as norm²=2.
    
    # Can we combine norm²=2 and norm²=8?
    # Max norm = 8. Need all cross-distances >= 8.
    # (1,1,0,..) and (2,2,0,..) sharing both coords: dist²=(1,1,0,..) norm²=2 < 8. Bad.
    
    # I think 440 is a strong answer. Let me just return the 440 norm²=2 vectors.
    # But let me double-check the count and validity.
    
    pts = np.array(pts_list, dtype=np.int64)
    n_pts = len(pts)
    
    # Verify
    norms = np.sum(pts ** 2, axis=1)
    max_norm = int(norms.max())
    
    min_dist = float('inf')
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            diff = pts[i] - pts[j]
            dist_sq = int(np.sum(diff ** 2))
            if dist_sq < min_dist:
                min_dist = dist_sq
    
    if min_dist < max_norm or n_pts < 2:
        # Fallback
        pts = np.array([np.eye(d, dtype=np.int64)[i] * s for i in range(d) for s in [1, -1]])
    
    return np.around(pts).astype(np.int64)
