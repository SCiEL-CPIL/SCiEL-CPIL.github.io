    def compute_phi_segment_numpy(L, R, primes):
        W = R - L + 1
        phi = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            start = ((L + p - 1) // p) * p
            indices = np.arange(start, R + 1, p)
            indices = indices - L
            phi[indices] -= phi[indices] // p
            
        # Correction
        # This part is hard to vectorize easily due to the trial division logic
        # But we can do it in a loop, which is slower.
        # However, the sieve part is now very fast.
        
        phi_list = phi.tolist()
        for i in range(W):
            if phi_list[i] > 1:
                n = L + i
                rem = n
                # Trial division
                for p in primes:
                    if p * p > rem:
                        break
                    if rem % p == 0:
                        while rem % p == 0:
                            rem //= p
                if rem > 1:
                    phi_list[i] -= phi_list[i] // rem
        return phi_list
