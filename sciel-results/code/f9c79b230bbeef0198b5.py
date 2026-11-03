<<<<<<< SEARCH
    # Verify the 3x3x3 decomposition using the full tensor verification
    # U1 is 9x21 (for A entries), V1 is 9x21 (for B entries in 3x3 block), W1 is 9x21 (for C entries in 3x3 block)
    # But for the full 3x3x7, the V and W indices are different.
    # Let's just build the full thing and verify it directly.
    
    # Build the full decomposition
    rank = 21 + 36
=======
    # Verify the 3x3x3 sub-decomposition in isolation
    def _verify_3x3x3(U1, V1, W1):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0.0
                    for t in range(21):
                        total += U1[i*3+j][t] * V1[j*3+k][t] * W1[k*3+i][t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not _verify_3x3x3(U1, V1, W1):
        # Fall back to full schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Build the full decomposition
    rank = 21 + 36
>>>>>>> REPLACE

Let me now write the final clean version with all fixes:

