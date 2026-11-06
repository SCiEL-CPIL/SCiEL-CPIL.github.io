"""Optimized tournament construction for Schütte k=4 using Paley + greedy pruning."""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def _build_in_bits(n, orientation):
        """Build in-neighborhood bitsets from orientation list.
        in_bits[j] has bit i set if i -> j.
        """
        in_bits = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    # i -> j
                    in_bits[j] |= (1 << i)
                else:
                    # j -> i
                    in_bits[i] |= (1 << j)
        return in_bits
    
    def _count_violations(n, in_bits):
        """Count exact number of violated 4-sets using bitsets."""
        violations = 0
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                for c in range(b + 1, n - 1):
                    for d in range(c + 1, n):
                        if (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]) == 0:
                            violations += 1
        return violations
    
    def _orient_from_adj(n, adj):
        """Convert adjacency matrix to orientation list."""
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                orientation.append(0 if adj[i, j] == 1 else 1)
        return orientation
    
    def _paleyn_tournament(p):
        """Paley tournament on p vertices (p prime, p ≡ 3 mod 4)."""
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                orientation.append(0 if diff in squares else 1)
        return p, orientation
    
    def _can_remove_vertex(n, in_bits, v):
        """Check if vertex v can be removed.
        v can be removed iff for every 4-set S disjoint from {v},
        there exists w in V\{v} such that w -> all vertices in S.
        Equivalently: no 4-set S (disjoint from v) has v as its ONLY dominator.
        """
        # For each 4-set not containing v, check if it has a dominator other than v
        remaining = [i for i in range(n) if i != v]
        n_rem = len(remaining)
        
        # Map remaining indices to 0..n_rem-1 for bit operations
        # But we can work with original indices using bitsets
        
        # A vertex w dominates 4-set {a,b,c,d} if w is in in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
        # We need: for all 4-sets {a,b,c,d} with none equal to v,
        #   (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]) has some bit set other than bit v
        
        for a in range(n - 3):
            if a == v:
                continue
            for b in range(a + 1, n - 2):
                if b == v:
                    continue
                for c in range(b + 1, n - 1):
                    if c == v:
                        continue
                    for d in range(c + 1, n):
                        if d == v:
                            continue
                        common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                        # Check if common has any bit set other than bit v
                        if (common & ~(1 << v)) == 0:
                            # The only possible dominator is v (or none)
                            return False
        return True
    
    def _prune_tournament(n, orientation, time_limit=60.0):
        """Greedy pruning: remove vertices that aren't uniquely needed."""
        in_bits = _build_in_bits(n, orientation)
        adj = np.zeros((n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    adj[i, j] = 1
                else:
                    adj[j, i] = 1
                idx += 1
        
        active = [True] * n
        removed = []
        
        while time.time() < start_time + time_limit:
            progress = False
            for v in range(n):
                if not active[v]:
                    continue
                if time.time() > start_time + time_limit:
                    break
                
                # Check if v can be removed
                if _can_remove_vertex(n, in_bits, v):
                    # Remove v
                    active[v] = False
                    removed.append(v)
                    progress = True
            
            if not progress:
                break
        
        # Build the pruned tournament
        remaining = [i for i in range(n) if active[i]]
        n_new = len(remaining)
        
        # Build new orientation
        new_orientation = []
        for idx_i, i in enumerate(remaining):
            for idx_j in range(idx_i + 1, n_new):
                j = remaining[idx_j]
                new_orientation.append(0 if adj[i, j] == 1 else 1)
        
        return n_new, new_orientation, remaining
    
    # Start with Paley(67)
    n_p, orient_p = _paleyn_tournament(67)
    in_bits_p = _build_in_bits(n_p, orient_p)
    violations_p = _count_violations(n_p, in_bits_p)
    
    if violations_p == 0:
        # Try to prune
        n_pruned, orient_pruned, remaining = _prune_tournament(n_p, orient_p, time_limit=130.0)
        
        # Verify the pruned tournament
        in_bits_new = _build_in_bits(n_pruned, orient_pruned)
        violations_new = _count_violations(n_pruned, in_bits_new)
        
        if violations_new == 0:
            return {'n': n_pruned, 'orientation': orient_pruned}
        else:
            # Pruning failed validation, return original
            return {'n': n_p, 'orientation': orient_p}
    else:
        # Paley(67) has violations, shouldn't happen but handle it
        return {'n': n_p, 'orientation': orient_p}
