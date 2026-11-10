import math

def construct():
    # We know the answer is 143, but we should implement a general search 
    # to be robust, or just return 143 if we are certain.
    # Given the constraints and the "Minimize n" requirement, 
    # and the fact that 143 is the first one found by analysis,
    # a simple loop is sufficient and safe within time limits.
    
    def phi(n):
        result = n
        p = 2
        nn = n
        while p * p <= nn:
            if nn % p == 0:
                while nn % p == 0:
                    nn //= p
                result -= result // p
            p += 1
        if nn > 1:
            result -= result // nn
        return result

    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
            
    return 1000000 # Fallback
