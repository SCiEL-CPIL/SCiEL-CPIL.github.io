import time

def construct(n):
    """
    Constructs a superpermutation of order n using the standard recursive method.
    The length of the resulting word is sum_{k=1}^n k!.
    For n=7, this is 5913.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Handle small n with hardcoded optimal solutions where known and simple
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        # Standard recursive gives "1213121" (len 7) which is optimal? 
        # 1!+2!+3! = 1+2+6=9. 
        # Wait, standard recursive for n=3:
        # m=2: word=[1]. perms of size 1: [1]. block=[1,2,1]. word=[1,2,1].
        # m=3: word=[1,2,1]. perms of size 2: [1,2], [2,1]. 
        # p=[1,2]: block=[1,2,3,1,2]. Overlap with [1,2,1] is 2 ([1,2]). Add [3,1,2]. word=[1,2,3,1,2].
        # p=[2,1]: block=[2,1,3,2,1]. Overlap with [1,2,3,1,2] is 1 ([2]). Add [1,3,2,1]. word=[1,2,3,1,2,1,3,2,1].
        # Length 9.
        # Is there a shorter one? 123121321 is length 9.
        return "123121321"
    
    # Recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we time out, we return the current word. 
            # However, for n=7, this loop is very fast (< 0.1s).
            # If n is large, this might timeout, but the task implies n=7.
            pass
            
        perms, seen = [], set()
        w_size = m - 1
        # Extract distinct permutations of size m-1 from the current word
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            # Check if it's a valid permutation (all distinct elements)
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            # The block to append is p, then m, then p
            block = list(p) + [m] + list(p)
            
            # Find the maximum overlap between the end of new_word and the start of block
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            # Check from largest possible overlap down to 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
    return "".join(map(str, word))

if __name__ == "__main__":
    # Test for small n
    print(construct(3))
    print(len(construct(4)))
    print(len(construct(5)))
    print(len(construct(6)))
    print(len(construct(7)))
