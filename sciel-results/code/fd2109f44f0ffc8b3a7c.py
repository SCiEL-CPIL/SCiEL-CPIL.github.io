import time
import math

_START_TIME = time.time()
TIME_LIMIT = 280

def priority(vector, n):
    """
    Static priority with balance penalty, adjacent-pair penalty, and geometric tie-breaker.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # Primary: Minimize deviation from balanced distribution
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Secondary: Penalize adjacent identical digits (local structure)
    # This is a new structural feature
    adj_penalty = 0
    for i in range(n - 1):
        if vector[i] == vector[i+1]:
            adj_penalty += 1
    
    # Tertiary: Geometric lex tie-breaker (Base 3, as in Insp 1)
    lex_val = 0.0
    for i in range(n):
        lex_val = lex_val * 3.0 + vector[i]
        
    # Combine
    # Weights: Primary dominates, Secondary is significant, Tertiary breaks ties
    return -penalty * 100000.0 - adj_penalty * 1000.0 + lex_val * 0.01
