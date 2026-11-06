    def _verify(comparators):
        if not hasattr(_verify, '_popcount'):
            _verify._popcount = np.array([bin(v).count('1') for v in range(65536)], dtype=np.uint8)
            _verify._targets = np.zeros(17, dtype=np.uint32)
            for k in range(17):
                if k == 0: _verify._targets[k] = 0
                elif k == 16: _verify._targets[k] = 65535
                else: _verify._targets[k] = ((1 << k) - 1) << (16 - k)
            
        all_inputs = np.arange(65536, dtype=np.uint32)
        
        for (i, j) in comparators:
            # Extract bits
            bi = (all_inputs >> i) & 1
            bj = (all_inputs >> j) & 1
            
            # Compute min/max
            ni = bi & bj
            nj = bi | bj
            
            # Update all_inputs
            # Clear i and j
            mask_i = 1 << i
            mask_j = 1 << j
            all_inputs = (all_inputs & ~(mask_i | mask_j)) | (ni << i) | (nj << j)
            
        ones = _verify._popcount[all_inputs]
        expected = _verify._targets[ones]
        
        return np.array_equal(all_inputs, expected)
    