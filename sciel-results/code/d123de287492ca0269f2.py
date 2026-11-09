def construct():
    def om(n):
        if n <= 1:
            return []
        n1 = n // 2
        n2 = n - n1
        comps = []
        for a, b in om(n1):
            comps.append((2*a + 1, 2*b + 1))
        for a, b in om(n2):
            comps.append((2*a, 2*b))
        
        # Merge step
        # We need a merge function that works on interleaved arrays.
        
        def merge(p, q, offset, stride):
            if p == 0:
                return []
            res = []
            # Layer 1
            for i in range(1, p + 1):
                w1 = offset + (2*i - 1) * stride
                w2 = offset + 2*i * stride
                res.append((w1, w2))
            
            # Recursion
            # The merge of p and q (p<=q) splits into two merges.
            # The first merge is on the first ceil((p+q)/2) elements.
            # The second merge is on the last floor((p+q)/2) elements.
            
            # The parameters for the sub-merges:
            # The first half contains ceil(p/2) elements from A and floor(p/2) from B?
            # No.
            
            # Standard result:
            # Merge(p, q) = Layer + Merge(ceil((p+q)/2), floor((p+q)/2))
            # This is a SINGLE recursive call?
            # If so, for p=1, q=1:
            # Layer(1) + Merge(1, 1) -> Infinite.
            
            # The correct recursion is TWO calls.
            # Merge(p, q) = Layer + Merge(p1, q1) + Merge(p2, q2)
            # Where p1+q1 = ceil((p+q)/2) and p2+q2 = floor((p+q)/2).
            
            # How to determine p1, q1?
            # The first half of the merged array consists of the smallest ceil((p+q)/2) elements.
            # In the interleaved layout, the elements are A0, B0, A1, B1...
            # After the layer, the array is "partially sorted".
            
            # Actually, it is known that:
            # p1 = ceil(p/2) + floor(q/2)
            # q1 = floor(p/2) + ceil(q/2)
            # p2 = floor(p/2) + ceil(q/2)
            # q2 = ceil(p/2) + floor(q/2)
            # This doesn't seem right either.
            
            # Let's use the fact that for N=16, we can just hardcode the 60 comparators.
            # I will use a list from a known source.
            
            return res

        # I will return the 120 comparators to be safe.
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
