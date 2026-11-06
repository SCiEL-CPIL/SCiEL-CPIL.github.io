<<<<<<< SEARCH
    L = len(sequences[labels[0]])
    # Encode each sequence as 4 binary indicator arrays
    enc = {}
    for lab in labels:
        s = sequences[lab]
        enc[lab] = (
            (s == 'A').astype(np.float64),
            (s == 'C').astype(np.float64),
            (s == 'G').astype(np.float64),
            (s == 'T').astype(np.float64),
        )
=======
    L = len(sequences[labels[0]])
    # Encode each sequence as 4 binary indicator arrays
    enc = {}
    for lab in labels:
        s = np.array(list(sequences[lab]), dtype='U1')
        enc[lab] = (
            (s == 'A').astype(np.float64),
            (s == 'C').astype(np.float64),
            (s == 'G').astype(np.float64),
            (s == 'T').astype(np.float64),
        )
>>>>>>> REPLACE
