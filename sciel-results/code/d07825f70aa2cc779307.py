<<<<<<< SEARCH
        if best_code is None:
            # Fallback: tetracode product
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)
=======
        if best_code is None:
            # Fallback: tetracode product
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)
        
        # Final verification and cleanup
        # Ensure all words are covered (should be, but verify)
        final_coverage = compute_coverage(best_code, neighbors)
        if np.any(final_coverage == 0):
            # Emergency: add words for uncovered positions
            uncovered_idx = np.where(final_coverage == 0)[0]
            for u in uncovered_idx:
                best_code.add(int(u))
            final_coverage = compute_coverage(best_code, neighbors)
>>>>>>> REPLACE

