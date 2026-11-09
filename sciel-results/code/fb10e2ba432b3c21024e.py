import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 128
    # Bitmask for the full domain 1..128
    # Bit i represents integer i+1? Or integer i?
    # Let's use bit i for integer i. So bit 1 is int 1, bit 128 is int 128.
    # Python ints are arbitrary precision, so this is fine.
    
    def get_set(mask):
        return [i for i in range(1, N+1) if (mask >> i) & 1]
        
    def can_add(x, mask):
        # Check if adding x to mask creates a forbidden triple
        # Triple (x, a, b) with a, b in mask, a != b
        # Sums: x+a, x+b, a+b must be in mask OR {x}
        # Since x+a > x (as a>=1), x+a cannot be x.
        # So x+a must be in mask.
        # Similarly x+b must be in mask.
        # a+b can be x? No, a,b >= 1, a!=b. Min sum 1+2=3.
        # If x=3, a=1, b=2, a+b=3=x.
        # So a+b can be x.
        
        # Iterate a in mask
        # To iterate set bits efficiently:
        # We can just loop 1..N and check bit. N=128 is small.
        for a in range(1, N+1):
            if not ((mask >> a) & 1):
                continue
            for b in range(a+1, N+1):
                if not ((mask >> b) & 1):
                    continue
                # Check sums
                sa = x + a
                sb = x + b
                ab = a + b
                
                # sa, sb must be in mask (since they are > x, they can't be x)
                # Unless sa or sb > N? If > N, they are not in mask, so condition fails (good).
                if sa > N or sb > N:
                    continue
                if not ((mask >> sa) & 1):
                    continue
                if not ((mask >> sb) & 1):
                    continue
                
                # ab must be in mask OR ab == x
                if ab == x:
                    return False
                if ab > N:
                    continue
                if (mask >> ab) & 1:
                    return False
        return True

    def greedy_random():
        elements = list(range(1, N+1))
        random.shuffle(elements)
        mask = 0
        for x in elements:
            if can_add(x, mask):
                mask |= (1 << x)
        return mask

    def greedy_sorted(desc=True):
        elements = list(range(1, N+1))
        if desc:
            elements.sort(reverse=True)
        else:
            elements.sort()
        mask = 0
        for x in elements:
            if can_add(x, mask):
                mask |= (1 << x)
        return mask

    best_mask = 0
    best_size = 0
    
    # 1. Try standard heuristics
    for desc in [True, False]:
        mask = greedy_sorted(desc)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # 2. Randomized Greedy
    # Estimate time per run:
    # can_add is O(N^2) bit checks. 128^2 = 16384 ops.
    # 128 calls per run. ~2M ops per run.
    # Python can do ~10-50M simple ops/sec.
    # So ~0.05s per run.
    # We have ~80s. Can do ~1000 runs.
    
    runs_done = 0
    while time.time() < deadline:
        mask = greedy_random()
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            # print(f"New best: {size}")
        runs_done += 1
        # Optional: try to improve current best with local search?
        # Let's just do more random restarts for now, it's robust.
        
    # 3. Local Search on best_mask
    # Try to swap elements to increase size or maintain size to find better configuration?
    # Actually, if we found a larger size, we are done.
    # If stuck, try to add elements by removing conflicting ones.
    
    # Try to add any missing element to best_mask
    # If conflict, try to remove one element from the conflicting set and retry.
    # This is complex. Random restarts usually suffice for N=128.
    
    # Convert best_mask to list of ints
    result = [i for i in range(1, N+1) if (best_mask >> i) & 1]
    
    # Ensure types are correct
    return [int(x) for x in result]

if __name__ == "__main__":
    res = construct()
    print(len(res))
