import numpy as np

def construct():
    """
    Find the minimum start s (1 <= s <= 1004) such that s, s+1, ..., s+4 are
    all quadratic residues modulo 1009, with explicit non-zero roots for each.
    
    Strategy:
    1. Use NumPy to vectorize the computation of quadratic residues modulo 1009.
    2. Create a boolean mask indicating which numbers in [1, 1008] are quadratic residues.
    3. Use vectorized logical AND to find the first index s where 5 consecutive numbers are QRs.
    4. Retrieve the smallest positive root for each of the 5 residues.
    """
    p = 1009
    
    # Generate x from 1 to p-1
    x = np.arange(1, p, dtype=np.int64)
    
    # Compute residues x^2 mod p
    residues = (x * x) % p
    
    # Create boolean mask for quadratic residues
    # is_qr[i] is True if i is a non-zero quadratic residue mod p
    is_qr = np.zeros(p, dtype=bool)
    is_qr[residues] = True
    
    # We need to find the smallest s in [1, 1004] such that
    # is_qr[s], is_qr[s+1], is_qr[s+2], is_qr[s+3], is_qr[s+4] are all True.
    # This corresponds to indices s-1 to s-1+4 in the is_qr array (0-indexed).
    # So we check is_qr[s-1] & is_qr[s] & ... & is_qr[s+3]
    # Let's construct a boolean array 'valid' where valid[i] is True if
    # is_qr[i] & is_qr[i+1] & is_qr[i+2] & is_qr[i+3] & is_qr[i+4]
    # i corresponds to s-1. So s = i+1.
    # We need i such that i+4 <= p-1 => i <= p-5 = 1004.
    # Also s >= 1 => i >= 0.
    # Max s is 1004, so max i is 1003.
    # is_qr indices used: i, i+1, i+2, i+3, i+4.
    # Max index accessed: 1003 + 4 = 1007. This is < p=1009. Safe.
    
    # Vectorized check for all possible starting indices i (0 to 1003)
    # valid[i] = is_qr[i] & is_qr[i+1] & is_qr[i+2] & is_qr[i+3] & is_qr[i+4]
    
    # We can do this by slicing the is_qr array.
    # i ranges from 0 to 1003.
    # is_qr[0:1004] corresponds to indices 0..1003
    # is_qr[1:1005] corresponds to indices 1..1004
    # is_qr[2:1006] corresponds to indices 2..1005
    # is_qr[3:1007] corresponds to indices 3..1006
    # is_qr[4:1008] corresponds to indices 4..1007
    
    valid = (is_qr[0:1004] & 
             is_qr[1:1005] & 
             is_qr[2:1006] & 
             is_qr[3:1007] & 
             is_qr[4:1008])
    
    # Find the first True in valid
    # If no True, np.argmax returns 0, but we should check.
    if not np.any(valid):
        # This case should not happen for p=1009, but handle gracefully
        raise RuntimeError("No valid run of 5 consecutive quadratic residues found")
    
    idx = np.argmax(valid)
    s = idx + 1  # Because idx is 0-based index for the start of the window
    
    # Now find the smallest positive root for each of s, s+1, ..., s+4
    # We can use the residues and x arrays.
    # For each target residue r, find the smallest x such that x^2 % p == r.
    
    # Create an array to store roots, initialized to 0
    roots_arr = np.zeros(p, dtype=np.int64)
    
    # Since x is sorted 1 to p-1, the first occurrence of a residue in the 'residues' array
    # corresponds to the smallest x.
    # We can use np.unique with return_index to find the first occurrence of each unique residue.
    unique_residues, first_indices = np.unique(residues, return_index=True)
    
    # Map the first x to the residue
    roots_arr[unique_residues] = x[first_indices]
    
    # Extract roots for s, s+1, s+2, s+3, s+4
    targets = np.array([s, s+1, s+2, s+3, s+4])
    res_roots = roots_arr[targets]
    
    return {'start': s, 'roots': res_roots.tolist()}
