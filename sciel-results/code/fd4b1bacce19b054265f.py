<<<<<<< SEARCH
    def _verify_booleans_numpy(comparators, N, num_samples=10000):
        """Verify using random Boolean inputs with NumPy."""
        if not comparators:
            return True
        
        # Test with random Boolean inputs
        rng = np.random.RandomState(42)
        for _ in range(num_samples):
            arr = rng.randint(0, 2, size=N).astype(np.int32)
            for i, j in comparators:
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
            # Check if sorted
            if not np.all(arr[:-1] <= arr[1:]):
                return False
        return True
    
    def _verify_all_booleans(comparators, N):
        """Full verification over all 2^N Boolean inputs."""
        if not comparators:
            return True
        
        for mask in range(1 << N):
            arr = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int32)
            for i, j in comparators:
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
            if not np.all(arr[:-1] <= arr[1:]):
                return False
        return True
=======
    def _verify_all_booleans_fast(comparators, N):
        """
        Verify the sorting network over all 2^N Boolean inputs using NumPy vectorization.
        Returns True if the network sorts all Boolean inputs.
        """
        if not comparators:
            return True
            
        # Generate all 2^N bitmasks
        inputs = np.arange(1 << N, dtype=np.int32)
        inputs_col = inputs[:, None]
        
        # Unpack bits
        # np.arange(N) shape (N,)
        # inputs_col shape (2^N, 1)
        # Result shape (2^N, N)
        arr = (inputs_col >> np.arange(N, dtype=np.int32)) & 1
        
        # Apply comparators
        for i, j in comparators:
            mask = arr[:, i] > arr[:, j]
            if np.any(mask):
                arr[mask, i] = 0
                arr[mask, j] = 1
        
        # Check if sorted
        diffs = np.diff(arr, axis=1)
        return not np.any(diffs < 0)
>>>>>>> REPLACE

