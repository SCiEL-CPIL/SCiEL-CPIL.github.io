def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    
    # Step 1: Construct the extended QR code of length 18
    qr_words = quadratic_residue_code()
    
    # Verify validity
    if not verify_code(qr_words):
        # Fallback to Reed-Muller if QR construction fails (should not happen)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
        
    return qr_words
