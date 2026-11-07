import time
import random

def construct():
    deadline = time.time() + 55.0
    n = 10
    
    # Base square
    A0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [A0]
    
    # Find a second square of the form A1[i][j] = (s[i] + j) % n
    # Conditions:
    # 1. s is a permutation of 0..n-1
    # 2. k[i] = (s[i] - i) % n is a permutation of 0..n-1
    
    s = [0] * n
    k = [0] * n
    used_s = [False] * n
    used_k = [False] * n
    
    def backtrack(i):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        for val in range(n):
            if not used_s[val]:
                k_val = (val - i) % n
                if not used_k[k_val]:
                    s[i] = val
                    k[i] = k_val
                    used_s[val] = True
                    used_k[k_val] = True
                    
                    if backtrack(i + 1):
                        return True
                    
                    used_s[val] = False
                    used_k[k_val] = False
        return False
    
    if backtrack(0):
        A1 = [[(s[i] + j) % n for j in range(n)] for i in range(n)]
        # Verify A1 is a Latin square (it is by construction of s)
        # Verify Orthogonality (it is by construction of k)
        squares.append(A1)
        
    return squares
