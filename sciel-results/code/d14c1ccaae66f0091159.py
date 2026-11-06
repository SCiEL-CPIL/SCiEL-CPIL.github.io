import time

N = 16

def construct():
    # 1. Generate Batcher's Odd-Even Merge Sort network
    def batcher_oes(n):
        if n <= 1:
            return []
        k = 1
        while k < n:
            k *= 2
        # Batcher's OES is recursive
        # But for fixed N=16, we can just use the known 63 comparator list
        # Or generate it recursively.
        # Let's generate it recursively to be safe.
        def oes(n):
            if n <= 1:
                return []
            if n == 2:
                return [(0, 1)]
            k = 1
            while k < n:
                k *= 2
            # Split into two halves of size k
            left = oes(k)
            right = [(i + k, j + k) for (i, j) in left]
            # Merge the two halves
            # The merge step for OES involves a specific pattern
            # Actually, Batcher's OES is:
            # 1. Sort left half
            # 2. Sort right half
            # 3. Odd-Even Transposition Sort on the combined list? No.
            # Batcher's Odd-Even Merge Sort:
            # 1. Recursively sort odd-indexed and even-indexed elements? No, that's odd-even transposition.
            # Batcher's OES splits into two halves, sorts them, then merges.
            # The merge step is:
            # For i in range(0, n//2, 2): compare (i, i+n//2)
            # Then recursively merge the results?
            
            # Let's use a known 63-comparator network for N=16 to avoid implementation errors.
            # I will hardcode the Batcher 63-comparator network for N=16.
            pass
        
        # Hardcoded Batcher OES for N=16 (63 comparators)
        # Source: Standard Batcher OES construction
        # Phase 1: Sort pairs (0,1), (2,3), ...
        # This is tedious to hardcode correctly without a generator.
        # Let's implement a simple recursive Batcher OES.
        
        def batcher_oes_rec(n):
            if n == 1:
                return []
            if n == 2:
                return [(0, 1)]
            # Find largest power of 2 <= n
            k = 1
            while k < n:
                k *= 2
            # Recursively sort first k and last n-k
            left = batcher_oes_rec(k)
            right = [(i + k, j + k) for i, j in batcher_oes_rec(n - k)]
            # Merge
            # The merge step in Batcher's OES is a bit complex.
            # It involves comparing (i, i+k) for i in 0..n-k-1 with some parity?
            # Actually, for powers of 2, Batcher's OES is equivalent to Bitonic Sort?
            # No, Bitonic is 63 for N=16?
            # Bitonic Sort for N=16:
            # Depth 1: (0,1), (2,3)... 8 comps
            # Depth 2: (0,2), (1,3)... 12 comps?
            # Bitonic sort for N=16 uses 63 comparators.
            # So I can just use Bitonic Sort.
            
            pass
        
        # Let's use Bitonic Sort for N=16. It's easy to implement and has 63 comparators.
        def bitonic_sort(n):
            if n <= 1:
                return []
            if n == 2:
                return [(0, 1)]
            # Recursively sort first half and second half
            half = n // 2
            left = bitonic_sort(half)
            right = [(i + half, j + half) for i, j in bitonic_sort(half)]
            # Now merge
            # Bitonic merge:
            # The sequence is bitonic.
            # We need to perform a bitonic merge.
            # This is recursive.
            
            pass
        
        # Easier: Hardcode the 63 comparators of Bitonic Sort for N=16.
        # Or generate them.
        # Bitonic Sort generation:
        def gen_bitonic(n):
            comps = []
            def rec(n, offset, direction):
                if n <= 1:
                    return
                if n == 2:
                    if direction == 1: # ascending
                        comps.append((offset, offset + 1))
                    else:
                        comps.append((offset, offset + 1)) # min to i, max to j. Direction matters for logic, but comparator is symmetric.
                    return
                half = n // 2
                rec(half, offset, 1)
                rec(half, offset + half, -1)
                # Merge
                # The merge step of bitonic sort involves comparing (offset+i, offset+i+half) for i in 0..half-1?
                # No, it's a recursive merge.
                # Actually, the standard bitonic merge is:
                # For k in range(half, 1, -1):
                #   for i in range(0, n-k):
                #      compare (offset+i, offset+i+k)
                
                # Let's use the standard iterative bitonic sort generation.
                pass
            
            # I will use a known list of 63 comparators for Bitonic Sort N=16.
            # I'll generate it using a reliable algorithm in the code.
            
            comps = []
            # Bitonic sort phases
            for length in range(2, n + 1, 2):
                for start in range(0, n, length):
                    # This is getting complicated.
                    pass
            
            return comps
        
        # Let's just use the Odd-Even Transposition Network (120) and then prune it.
        # The pruning strategy is robust.
        
        def oet_network(n):
            return [(i, i + 1) for phase in range(n)
                    for i in range(phase % 2, n - 1, 2)]
                    
        initial_net = oet_network(N)
        
        # 2. Verification function
        def is_valid(net):
            # Check all 2^16 boolean inputs
            for mask in range(1 << N):
                # Convert mask to list of bits
                # Actually, we can simulate on the mask directly
                # But min/max on bits is easy.
                # Let's use a list of 0/1 for clarity, though mask is faster.
                # Mask simulation:
                m = mask
                for i, j in net:
                    # Extract bits
                    bi = (m >> i) & 1
                    bj = (m >> j) & 1
                    if bi > bj:
                        # Swap: set i to 0, j to 1
                        # Clear i, set j
                        m &= ~(1 << i)
                        m |= (1 << j)
                    # else: i=0, j=1 is already correct (0<1) or 0<0, 1<1
                    # Wait, if bi=0, bj=1, min=0, max=1. Correct.
                    # If bi=1, bj=0, min=0, max=1. Swap.
                    # If bi=0, bj=0, no change.
                    # If bi=1, bj=1, no change.
                
                # Check if sorted: all 0s then all 1s
                # This means the mask should be of the form:
                # 000...011...1
                # i.e., if we have k ones, they must be in the lowest k bits?
                # No, sorted ascending: 000...011...1
                # So the mask must be `(1 << k) - 1` for some k.
                # Let's check if m is of this form.
                # A number is of the form 2^k - 1 iff m & (m+1) == 0.
                if (m & (m + 1)) != 0:
                    return False
            return True

        # 3. Pruning
        current_net = initial_net
        start_time = time.time()
        deadline = start_time + 50 # 50s limit
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_net)):
                if time.time() > deadline:
                    break
                candidate = current_net[:i] + current_net[i+1:]
                if is_valid(candidate):
                    current_net = candidate
                    improved = True
                    break # Restart loop
            
        return current_net

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    N = 16
    
    def oet_network(n):
        return [(i, i + 1) for phase in range(n)
                for i in range(phase % 2, n - 1, 2)]
                
    initial_net = oet_network(N)
    
    def is_valid(net):
        for mask in range(1 << N):
            m = mask
            for i, j in net:
                bi = (m >> i) & 1
                bj = (m >> j) & 1
                if bi > bj:
                    m &= ~(1 << i)
                    m |= (1 << j)
            if (m & (m + 1)) != 0:
                return False
        return True

    current_net = initial_net
    start_time = time.time()
    deadline = start_time + 50
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_net)):
            if time.time() > deadline:
                break
            candidate = current_net[:i] + current_net[i+1:]
            if is_valid(candidate):
                current_net = candidate
                improved = True
                break
                
    return current_net
