import time
from math import gcd
import sys

def construct():
    deadline = time.time() + 80
    n = 255
    
    # Candidates: numbers in [1,255] with gcd(v,255)>1
    # i.e., divisible by 3, 5, or 17
    candidates = [v for v in range(1, n+1) if v % 3 == 0 or v % 5 == 0 or v % 17 == 0]
    
    # Build adjacency using bitsets (Python ints)
    m = len(candidates)
    adj_bits = [0] * m
    for i in range(m):
        for j in range(i+1, m):
            if gcd(candidates[i], candidates[j]) > 1:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # Find max clique containing 255
    idx_255 = candidates.index(255)
    
    # Try multiple initial solutions to get a good lower bound
    best = set()
    best_sz = 0
    
    # Initial: all multiples of 3
    mult3 = set(i for i in range(m) if candidates[i] % 3 == 0)
    if idx_255 in mult3:
        best_sz = len(mult3)
        best = mult3.copy()
    
    # Initial: all multiples of 5
    mult5 = set(i for i in range(m) if candidates[i] % 5 == 0)
    if idx_255 in mult5 and len(mult5) > best_sz:
        best_sz = len(mult5)
        best = mult5.copy()
    
    # Initial: all multiples of 17
    mult17 = set(i for i in range(m) if candidates[i] % 17 == 0)
    if idx_255 in mult17 and len(mult17) > best_sz:
        best_sz = len(mult17)
        best = mult17.copy()
    
    sys.setrecursionlimit(10000)
    
    # Bitset-based Bron-Kerbosch with pivot
    # R, P, X are represented as Python int bitsets
    def bk(R_bits, P_bits, X_bits):
        nonlocal best, best_sz
        if time.time() > deadline - 1:
            return
        # Check if done: P and X are both empty
        if P_bits == 0 and X_bits == 0:
            r_size = bin(R_bits).count('1')
            if r_size > best_sz:
                best_sz = r_size
                best = set(i for i in range(m) if (R_bits >> i) & 1)
            return
        
        # Pivot selection: choose vertex in P with max degree in P
        if P_bits:
            # Find vertex u in P that maximizes |adj[u] & P|
            # Iterate through bits in P
            P_tmp = P_bits
            u = -1
            max_deg = -1
            while P_tmp:
                v = (P_tmp & (-P_tmp)).bit_length() - 1
                P_tmp &= P_tmp - 1
                deg = bin(adj_bits[v] & P_bits).count('1')
                if deg > max_deg:
                    max_deg = deg
                    u = v
            Q_bits = P_bits & ~(adj_bits[u])
        else:
            Q_bits = 0
        
        # Iterate through vertices in Q
        Q_tmp = Q_bits
        while Q_tmp:
            v = (Q_tmp & (-Q_tmp)).bit_length() - 1
            Q_tmp &= Q_tmp - 1
            bk(R_bits | (1 << v), P_bits & adj_bits[v], X_bits & adj_bits[v])
            P_bits &= ~(1 << v)
            X_bits |= (1 << v)
    
    # Must include 255
    P0 = adj_bits[idx_255]
    bk(1 << idx_255, P0, 0)
    
    result = sorted([candidates[i] for i in best])
    
    # Verify
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert gcd(result[i], result[j]) > 1
    
    return result
