"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
def construct():
    def odd_merge(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        n1 = (n + 1) // 2
        n2 = n // 2
        
        comms = []
        
        # Sort first half (indices 0 to n1-1)
        comms.extend(odd_merge(n1))
        
        # Sort second half (indices n1 to n-1)
        # Map indices from 0..n2-1 to n1..n-1
        for i, j in odd_merge(n2):
            comms.append((i + n1, j + n1))
        
        # Merge step
        # 1. Sort even-indexed elements of the combined sequence
        # Even indices: 0, 2, ..., 2*(n1-1)
        # Map indices from 0..n1-1 to 0, 2, ...
        for i, j in odd_merge(n1):
            comms.append((2 * i, 2 * j))
            
        # 2. Sort odd-indexed elements of the combined sequence
        # Odd indices: 1, 3, ..., 2*n2-1
        # Map indices from 0..n2-1 to 1, 3, ...
        for i, j in odd_merge(n2):
            comms.append((2 * i + 1, 2 * j + 1))
            
        # 3. Zig-zag comparators
        # For i=1 to n//2 - 1:
        # Compare (2*i - 1, 2*i)
        for i in range(1, n // 2):
            comms.append((2 * i - 1, 2 * i))
            
        return comms

    return odd_merge(N)
# EVOLVE-BLOCK-END
