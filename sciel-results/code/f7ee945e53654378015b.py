where a = i*m + j, b = j*p + k, c = k*n + i.

This computes sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] and checks it equals 1.

In my rank-3 construction:
- U[0*m+j, 0] = 1 for all j, U[1*m+j, 1] = 1 for all j, U[2*m+j, 2] = 1 for all j
- V[j*p+k, 0] = 1 for all j,k, V[j*p+k, 1] = 1 for all j,k, V[j*p+k, 2] = 1 for all j,k
- W[k*n+i, 0] = 1 for all k,i, W[k*n+i, 1] = 1 for all k,i, W[k*n+i, 2] = 1 for all k,i

For i=0, j=0, k=0: sum_r U[0, r] * V[0, r] * W[0, r] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] + U[0,2]*V[0,2]*W[0,2] = 1*1*1 + 0*1*1 + 0*1*1 = 1. Good.

For i=1, j=0, k=0: sum_r U[4, r] * V[0, r] * W[3, r] = U[4,0]*V[0,0]*W[3,0] + U[4,1]*V[0,1]*W[3,1] + U[4,2]*V[0,2]*W[3,2] = 0*1*1 + 1*1*1 + 0*1*1 = 1. Good.

This works! So the rank is 3. But this is clearly not what the task is about, since the current program uses rank 44.

I think I need to re-read the task more carefully. Maybe the tensor is not T[i,j,k] = 1 for all i,j,k, but rather the tensor for a specific matrix multiplication algorithm or something else.

Actually, wait. I just realized - maybe the issue is that the "matrix multiplication tensor" in this context is not the trivial T[i,j,k]=1 tensor, but rather the tensor that encodes the structure of matrix multiplication in a way that respects the algebraic structure. But no, the check function clearly shows it's just checking that the sum equals 1 for all i,j,k.

I'm confused. Let me just proceed with the improvements to the existing code and not try to change the fundamental approach. The rank-3 solution I found might be valid but perhaps there's a constraint I'm missing (like the entries must be half-integers, which they are in my construction).

Actually, let me test this. If rank 3 works, then the score would be 44/3 ≈ 14.67, which would be a massive improvement. But the task says the current score is 1.0909 which corresponds to rank 44 (since 44/44 = 1.0, and 1.0909 = 44/40.35... hmm that doesn't work out).

Wait, the score is "seed/value" where seed is presumably 44 (the rank of the seed program). So score = 44/rank. Current score 1.0909 means rank = 44/1.0909 ≈ 40.35. That's not an integer. Let me recalculate: 44/1.0909 = 40.34. Hmm.

Actually, looking at the observed data, the score is 1.0909090909090908 which is 44/40.34... no. 1.09090909... = 1 + 1/11 = 12/11. So 44/(12/11) = 44*11/12 = 484/12 = 40.33. That's not right either.

Wait, maybe the seed rank is not 44. Let me re-read: "score is the rank" and "The evaluator measures the raw value (lower is better). The search score is seed/value". So if seed = 44 and value = rank, then score = 44/rank. For score = 1.0909, rank = 44/1.0909 ≈ 40.34. That's not an integer.

Hmm, maybe I'm misreading. Let me look at the observed score: 1.0909090909090908. This is exactly 12/11. If seed = 44, then rank = 44 * 11/12 = 40.33. Not integer.

Maybe seed is not 44. Or maybe the score formula is different. Let me just focus on improving the existing code.

Given my confusion about the rank-3 solution, let me not pursue that path and instead focus on the directive's suggestion of improving the search within the current family. The key improvement is the vectorized check function, which will make the local search much faster and allow more iterations.

Let me also add a more systematic local search that tries structured perturbations rather than just random single-entry changes.

