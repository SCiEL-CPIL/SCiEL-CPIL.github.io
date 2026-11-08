<<<<<<< SEARCH
import time
import math
import random
from itertools import combinations

def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    for i in range(n):
        for j in range(n):
            sum_set.add(U[i] + U[j])
            diff_set.add(U[i] - U[j])
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * maxU + 1)
    return score
=======
import time
import math
import random
import numpy as np
from itertools import combinations

def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Convert to numpy array for vectorized operations
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute U+U and U-U using broadcasting
    # This creates n x n arrays, which is feasible for n up to ~1000 within time limits
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    # Count unique elements
    # np.unique on large arrays can be slow, but for n=100, n^2=10000 elements, it's fast.
    # For larger n, we might need to be careful, but the search space is limited.
    num_sums = len(np.unique(sums))
    num_diffs = len(np.unique(diffs))
    
    if num_sums == 0:
        return 0.0
        
    ratio = num_diffs / num_sums
    if ratio <= 0:
        return 0.0
    
    # log base (2*maxU + 1)
    base = 2 * maxU + 1
    if base <= 1:
        return 0.0
        
    score = 1.0 + math.log(ratio) / math.log(base)
    return score
>>>>>>> REPLACE

