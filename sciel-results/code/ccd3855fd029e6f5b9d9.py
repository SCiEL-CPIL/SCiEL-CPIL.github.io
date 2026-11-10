"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = 1
    for i in range(1, 13):
        period = lcm(period, i)
    # period = 27720

    def has_matching(m, L):
        """
        Check if there exist distinct b_1,...,b_12 in (m, m+L] with i | b_i.
        Uses DFS-based bipartite matching.
        Left: indices 0..11 (representing i=1..12)
        Right: positions 0..L-1 (representing values m+1..m+L)
        Edge (i, pos) exists if (m+1+pos) % (i+1) == 0.
        """
        # match_right[j] = i if position j is matched to index i, else -1
        match_right = [-1] * L
        
        # For each index i, we will try to find an augmenting path
        # We need to check if we can match all 12 indices.
        
        # To speed up, we can order indices by number of available positions?
        # Or just iterate 0..11.
        
        def dfs(i, visited):
            # Try to find a position for index i
            # i is 0-indexed, so divisor is i+1
            div = i + 1
            for pos in range(L):
                if visited[pos]:
                    continue
                val = m + 1 + pos
                if val % div != 0:
                    continue
                visited[pos] = True
                # If position pos is free, or if the index currently matched to pos can find another position
                if match_right[pos] == -1 or dfs(match_right[pos], visited):
                    match_right[pos] = i
                    return True
            return False

        # We need to match all 12 indices.
        # Standard Kuhn's algorithm.
        # We iterate over indices.
        # To improve performance, we can skip if an index has no available positions.
        # Also, we can break early if we fail to match an index?
        # No, we need to try all.
        
        # Optimization: Precompute available positions for each index?
        # That costs O(12*L) per call.
        # DFS costs O(E) per index, total O(12*E).
        
        for i in range(n):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        
        return True

    def find_max_L_and_m():
        """
        Find the maximum L such that there exists m in [0, period) with no matching.
        Then find such an m.
        """
        # Binary search for the largest L where P(L) is True.
        # P(L): exists m in [0, period) such that NOT has_matching(m, L)
        
        lo, hi = 1, 144
        # We know P(1) is True (for any m, L=1, only 1 value, can't match 12).
        # We want the largest L.
        
        # First, check if P(144) is True.
        # If P(144) is False, then all m have matching for L=144.
        # Then f(12, m) <= 144 for all m.
        # We want to find the max f.
        
        # Let's binary search for the largest L such that P(L) is True.
        # If P(L) is True, there is an m with f(12, m) > L.
        # So f(12, m) >= L+1.
        
        # If P(L) is False, then for all m, f(12, m) <= L.
        
        # We want to maximize f(12, m).
        # Let L* be the largest L such that P(L) is True.
        # Then max f >= L* + 1.
        # And since P(L*+1) is False, max f <= L* + 1.
        # So max f = L* + 1.
        
        # Edge case: If P(144) is True, then L* = 144, max f = 145?
        # But problem says f is at most 144.
        # So if P(144) is True, it means there is m with no matching for 144.
        # Then f(12, m) > 144. But max is 144.
        # This implies P(144) must be False for the problem constraints?
        # Or maybe my definition of P(L) is slightly off.
        # If no matching for L=144, then f(12, m) is not defined or > 144.
        # The problem states f is the LEAST positive L such that matching exists.
        # If no such L exists up to 144, what is f?
        # "Maximize this independently computed L. ... raw value is f(12,m), which is at most 144."
        # This implies that for the optimal m, f(12,m) is well-defined and <= 144.
        # So there must be a matching for L=144 for all m?
        # Or maybe for the specific m we pick, it exists.
        # If there exists an m with NO matching for L=144, then f(12,m) is not <= 144.
        # So we should only consider m where matching exists for L=144?
        # No, the evaluator computes f(12,m). If it doesn't exist, it's invalid?
        # "The evaluator computes f(12,m)... Maximize this".
        # If f is undefined, the score is 0.
        # So we must ensure the returned m has a matching for L=144.
        
        # So we want to maximize f(12,m) subject to f(12,m) <= 144.
        # This is equivalent to finding m such that the first matching is as late as possible,
        # but it must exist by 144.
        
        # So we look for m such that:
        # 1. Matching exists for L=144.
        # 2. Matching does NOT exist for L=f(12,m)-1.
        
        # We can iterate L from 144 down to 1.
        # For a given L, check if there exists m such that:
        # a) has_matching(m, L) is True
        # b) has_matching(m, L-1) is False
        # If we find such an m, then f(12,m) = L.
        # Since we iterate L downwards, the first such L we find is the maximum.
        
        for L in range(144, 0, -1):
            if time.time() > deadline:
                break
            # Check if exists m with matching for L but not L-1
            # If L=1, L-1=0. has_matching(m, 0) is always False (empty interval).
            # So for L=1, we just need has_matching(m, 1) to be True.
            
            found_m = -1
            for m in range(period):
                if time.time() > deadline:
                    return -1 # Timeout, return invalid or best found
                # Check has_matching(m, L)
                if not has_matching(m, L):
                    continue
                # Check has_matching(m, L-1)
                if L == 1:
                    # has_matching(m, 0) is False
                    found_m = m
                    break
                else:
                    if not has_matching(m, L-1):
                        found_m = m
                        break
            
            if found_m != -1:
                return L, found_m
        
        # If no L found, something is wrong. Return a default.
        return 1, 0

    result = find_max_L_and_m()
    if result[0] == -1:
        return 0 # Timeout
    
    L_max, m_best = result
    # Verify m_best is in range
    if m_best > max_m:
        m_best = m_best % (max_m + 1)
        
    return int(m_best)

# EVOLVE-BLOCK-END
