"""Improved program using DFS with Backtracking."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    max_val = 256
    
    def check_valid(seq, x, sums):
        new_sums = set()
        total = x
        new_sums.add(total)
        for a in reversed(seq):
            total += a
            new_sums.add(total)
        if new_sums & sums:
            return False, None
        return True, new_sums

    def dfs(idx, last, sums, seq, target_len, node_limit):
        if node_limit[0] <= 0:
            return None
        
        if idx == target_len:
            return list(seq)
        
        # Pruning: Not enough numbers left to reach target_len
        if max_val - last < target_len - idx:
            return None
            
        # Try x from max_val down to last + 1
        # Heuristic: Large x creates large new sums, reducing collision probability with existing small sums
        for x in range(max_val, last, -1):
            node_limit[0] -= 1
            if node_limit[0] <= 0:
                return None
            
            valid, new_sums = check_valid(seq, x, sums)
            if valid:
                seq.append(x)
                result = dfs(idx + 1, x, sums | new_sums, seq, target_len, node_limit)
                if result is not None:
                    return result
                seq.pop()
                
        return None

    # Try to find a sequence of length L, starting from high L
    # L=9 is guaranteed possible (powers of 2)
    for L in range(12, 8, -1):
        node_limit = [200000]
        result = dfs(0, 0, set(), [], L, node_limit)
        if result is not None:
            return result
            
    # Fallback: Powers of 2
    seq = []
    val = 1
    while val <= max_val:
        seq.append(val)
        val *= 2
    return seq

# EVOLVE-BLOCK-END
