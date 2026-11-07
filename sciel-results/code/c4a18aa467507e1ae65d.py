import time
import random
import math
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v) if u < v else (v, u))
        
    def popcount(x):
        return bin(x).count('1')
        
    def mask_to_list(mask):
        result = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            result.append(v)
            m ^= lsb
        return result
        
    def greedy_partition(seed):
        random.seed(seed)
        rem_adj = adj[:]
        bicliques = []
        edges_left = len(edges)
        
        while edges_left > 0:
            if time.time() > deadline:
                return None  # Timeout
            
            # Pick a random edge (u, v) from remaining edges
            active_vertices = [u for u in range(n) if rem_adj[u] != 0]
            if not active_vertices:
                break
                
            u = random.choice(active_vertices)
            mask = rem_adj[u]
            # Get list of neighbors
            neighbors = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                neighbors.append(v)
                m ^= lsb
            
            if not neighbors:
                continue
                
            v = random.choice(neighbors)
            
            # Expand biclique starting with (u, v)
            L_mask = 1 << u
            R_mask = 1 << v
            
            # Common neighbors of all in L and all in R
            changed = True
            while changed:
                changed = False
                
                # Expand L
                common_L = (1 << n) - 1
                m_r = R_mask
                while m_r:
                    lsb = m_r & -m_r
                    y = lsb.bit_length() - 1
                    m_r ^= lsb
                    common_L &= rem_adj[y]
                    
                new_L_candidates = common_L & ~L_mask & ~R_mask
                if new_L_candidates:
                    L_mask |= new_L_candidates
                    changed = True
                    
                # Expand R
                common_R = (1 << n) - 1
                m_l = L_mask
                while m_l:
                    lsb = m_l & -m_l
                    x = lsb.bit_length() - 1
                    m_l ^= lsb
                    common_R &= rem_adj[x]
                    
                new_R_candidates = common_R & ~R_mask & ~L_mask
                if new_R_candidates:
                    R_mask |= new_R_candidates
                    changed = True
            
            # Remove edges between L and R
            m_l = L_mask
            while m_l:
                lsb = m_l & -m_l
                u_cur = lsb.bit_length() - 1
                m_l ^= lsb
                rem_adj[u_cur] &= ~R_mask
                
            m_r = R_mask
            while m_r:
                lsb = m_r & -m_r
                v_cur = lsb.bit_length() - 1
                m_r ^= lsb
                rem_adj[v_cur] &= ~L_mask
                
            edges_left -= popcount(L_mask) * popcount(R_mask)
            
            # Convert to lists
            L_list = mask_to_list(L_mask)
            R_list = mask_to_list(R_mask)
            
            bicliques.append([L_list, R_list])
            
        return bicliques
    
    def try_merge(bicliques):
        """Try to merge compatible bicliques to reduce count."""
        if len(bicliques) <= 1:
            return bicliques
        
        changed = True
        while changed:
            changed = False
            new_bicliques = []
            i = 0
            while i < len(bicliques):
                if i + 1 < len(bicliques):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[i + 1]
                    
                    # Try merging: new biclique (L1 ∪ L2, R1 ∪ R2) is valid if
                    # all cross edges exist. We need to check that every pair (u, v)
                    # with u in L1 ∪ L2 and v in R1 ∪ R2 is an edge in the graph.
                    # This is expensive, so we only try if the bicliques are small.
                    if len(L1) * len(R1) + len(L2) * len(R2) < 100:
                        new_L = set(L1) | set(L2)
                        new_R = set(R1) | set(R2)
                        
                        # Check disjointness
                        if not new_L.intersection(new_R):
                            # Check all cross edges
                            valid = True
                            for u in new_L:
                                for v in new_R:
                                    e = (u, v) if u < v else (v, u)
                                    if e not in edge_set:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            
                            if valid:
                                # Merge: remove edges covered by the merged biclique
                                # But we need to ensure no edge is covered twice
                                # This is complex, so let's just record the merge
                                new_bicliques.append([list(new_L), list(new_R)])
                                i += 2
                                changed = True
                                continue
                    
                    new_bicliques.append(bicliques[i])
                    i += 1
            bicliques = new_bicliques
        
        return bicliques
    
    def validate(bicliques):
        """Validate the biclique partition."""
        for b in bicliques:
            if not isinstance(b, list) or len(b) != 2:
                return False
            L, R = b
            if not isinstance(L, list) or not isinstance(R, list):
                return False
            for x in L:
                if not isinstance(x, int) or isinstance(x, bool):
                    return False
            for x in R:
                if not isinstance(x, int) or isinstance(x, bool):
                    return False
            if not L or not R:
                return False
            if set(L).intersection(set(R)):
                return False
                
        covered_edges = set()
        for L, R in bicliques:
            for u in L:
                for v in R:
                    e = (u, v) if u < v else (v, u)
                    if e not in edge_set:
                        return False
                    if e in covered_edges:
                        return False
                    covered_edges.add(e)
                    
        return covered_edges == edge_set
    
    best_bicliques = None
    best_count = float('inf')
    
    # Strategy 1: Multi-candidate greedy with different random seeds
    for seed in range(2000):
        if time.time() > deadline:
            break
            
        result = greedy_partition(seed)
        if result is not None:
            count = len(result)
            if count < best_count:
                best_count = count
                best_bicliques = result
                
        # Early stop if we found a very good solution
        if best_count <= 6:
            break
    
    # Strategy 2: Try to improve by merging
    if best_bicliques is not None:
        merged = try_merge(best_bicliques)
        if validate(merged) and len(merged) < best_count:
            best_count = len(merged)
            best_bicliques = merged
    
    # Strategy 3: Local search - try replacing a biclique with smaller ones
    if best_bicliques is not None and time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_bicliques)):
                if time.time() > deadline - 5:
                    break
                # Try to split biclique i into two smaller bicliques
                L, R = best_bicliques[i]
                if len(L) > 1 and len(R) > 1:
                    # Try all ways to partition L and R
                    for split in range(1, len(L)):
                        L1 = L[:split]
                        L2 = L[split:]
                        for split_r in range(1, len(R)):
                            R1 = R[:split_r]
                            R2 = R[split_r:]
                            
                            # Check if (L1, R1) and (L2, R2) are valid bicliques
                            # and cover the same edges
                            edges_1 = set()
                            edges_2 = set()
                            valid = True
                            
                            for u in L1:
                                for v in R1:
                                    e = (u, v) if u < v else (v, u)
                                    if e not in edge_set:
                                        valid = False
                                        break
                                    edges_1.add(e)
                                if not valid:
                                    break
                            
                            if valid:
                                for u in L2:
                                    for v in R2:
                                        e = (u, v) if u < v else (v, u)
                                        if e not in edge_set:
                                            valid = False
                                            break
                                        edges_2.add(e)
                                    if not valid:
                                        break
                            
                            if valid and edges_1.isdisjoint(edges_2) and edges_1 | edges_2 == set(
                                (u, v) if u < v else (v, u) for u in L for v in R
                            ):
                                # This split works, but it increases count by 1
                                # Only useful if it allows other merges
                                pass
    
    # Strategy 4: Beam search with multiple candidates per step
    if time.time() < deadline - 20:
        beam_width = 5
        beam = [None]  # None represents empty state
        best_beam_count = float('inf')
        
        for step in range(15):  # Max 15 bicliques
            new_beam = []
            for state in beam:
                if state is None:
                    # Start with any edge
                    for u, v in edges:
                        if time.time() > deadline - 20:
                            break
                        L_mask = 1 << u
                        R_mask = 1 << v
                        
                        # Expand
                        changed = True
                        while changed:
                            changed = False
                            common_L = (1 << n) - 1
                            m_r = R_mask
                            while m_r:
                                lsb = m_r & -m_r
                                y = lsb.bit_length() - 1
                                m_r ^= lsb
                                common_L &= adj[y]
                            new_L = common_L & ~L_mask & ~R_mask
                            if new_L:
                                L_mask |= new_L
                                changed = True
                            common_R = (1 << n) - 1
                            m_l = L_mask
                            while m_l:
                                lsb = m_l & -m_l
                                x = lsb.bit_length() - 1
                                m_l ^= lsb
                                common_R &= adj[x]
                            new_R = common_R & ~R_mask & ~L_mask
                            if new_R:
                                R_mask |= new_R
                                changed = True
                        
                        new_state = ([L_mask, R_mask], set())
                        new_beam.append(new_state)
                else:
                    L_mask, R_mask, covered = state
                    # Try adding a new biclique
                    for u, v in edges:
                        e = (u, v) if u < v else (v, u)
                        if e in covered:
                            continue
                        if time.time() > deadline - 20:
                            break
                        # Expand
                        L_new = 1 << u
                        R_new = 1 << v
                        changed = True
                        while changed:
                            changed = False
                            common_L = (1 << n) - 1
                            m_r = R_new
                            while m_r:
                                lsb = m_r & -m_r
                                y = lsb.bit_length() - 1
                                m_r ^= lsb
                                common_L &= adj[y]
                            new_L = common_L & ~L_new & ~R_new & ~L_mask
                            if new_L:
                                L_new |= new_L
                                changed = True
                            common_R = (1 << n) - 1
                            m_l = L_new
                            while m_l:
                                lsb = m_l & -m_l
                                x = lsb.bit_length() - 1
                                m_l ^= lsb
                                common_R &= adj[x]
                            new_R = common_R & ~R_new & ~L_new & ~R_mask
                            if new_R:
                                R_new |= new_R
                                changed = True
                        
                        # Check no overlap with existing bicliques
                        overlap = False
                        for u2 in mask_to_list(L_new):
                            for v2 in mask_to_list(R_new):
                                e2 = (u2, v2) if u2 < v2 else (v2, u2)
                                if e2 in covered:
                                    overlap = True
                                    break
                            if overlap:
                                break
                        
                        if not overlap:
                            new_covered = covered | set(
                                (u2, v2) if u2 < v2 else (v2, u2) 
                                for u2 in mask_to_list(L_new) 
                                for v2 in mask_to_list(R_new)
                            )
                            new_beam.append(([L_new, R_new], new_covered))
            
            # Keep only top beam_width states
            if new_beam:
                # Score by number of covered edges
                new_beam.sort(key=lambda x: len(x[1]), reverse=True)
                beam = new_beam[:beam_width]
            else:
                break
    
    # Fallback: trivial star decomposition
    if best_bicliques is None:
        best_bicliques = []
        for u, v in edges:
            best_bicliques.append([[u], [v]])
    
    # Final validation
    if not validate(best_bicliques):
        # Force valid fallback
        best_bicliques = []
        for u, v in edges:
            best_bicliques.append([[u], [v]])
    
    return best_bicliques
