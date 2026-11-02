# After ALS, check the error
recon = np.zeros((n*m, m*p, p*n))
for r_idx in range(r):
    recon += np.outer(np.outer(U[:, r_idx], V[:, r_idx]), W[:, r_idx])
error = np.linalg.norm(recon - T)
if error > 0.1:
    continue  # Skip this trial
