
Let me verify C[0,0] = ae + bg:
- m1 contributes: a·e (from a·e)
- m2 contributes: a·e + a·f + b·e + b·f + c·e + c·f + d·e + d·f
- m3 contributes: a·e - a·f + 2d·e - 2d·f
- m4 contributes: c·f + c·h + d·f + d·h
- m5 contributes: a·e + c·e
- m6 contributes: b·f + d·f
- m7 contributes: a·e + a·g + 2b·e + 2b·g + c·e + c·g + 2b·f + 2b·h... 

This is getting complex. Let me just trust the numerical verification in the code. The key question is whether my `make_laderman()` function produces the correct terms. Let me reconsider.

Actually, looking at this more carefully, the Laderman decomposition I wrote may not be correct. Let me use a well-known verified form. The issue is that the "Laderman" terms need to be expressed as rank-1 tensor factors, which is non-trivial.

Let me take a different approach: use the known fact that the 2×2×2 MM tensor has rank 7, and use a verified decomposition from the literature. One standard form (from various sources) is:

For A = [[a,b],[c,d]] and B = [[e,f],[g,h]]:
- p1 = (a+d)(e+h)
- p2 = (a+b+c+d)(e+f+g+h)  
- p3 = (a-b+2d)(e-f+2h)
- p4 = (c+d)(f+h)
- p5 = (a+c)(e+g)
- p6 = (b+d)(f+g)
- p7 = (a+c+2b)(e+g+2f)

And:
- C[0,0] = p1 - p2 + p3 + p4 - p5 + p6 - p7
- C[0,1] = p1 - p2 - p3 - p4 - p5 + p6 + p7
- C[1,0] = p1 + p2 + p3 - p4 + p5 - p6 - p7
- C[1,1] = p1 + p2 - p3 + p4 + p5 + p6 + p7

Let me verify C[0,0]:
= (a+d)(e+h) - (a+b+c+d)(e+f+g+h) + (a-b+2d)(e-f+2h) + (c+d)(f+h) - (a+c)(e+g) + (b+d)(f+g) - (a+c+2b)(e+g+2f)

This is tedious to verify by hand. The numerical verification in the code will catch errors. But I need to make sure my `make_laderman()` produces the right tensor factors.

The problem is that each pᵢ is a product of a linear form in A and a linear form in B, which means each pᵢ corresponds to a rank-1 tensor where:
- The A part is a vector uᵢ over (i,j)
- The B part is a vector vᵢ over (j,k)
- The C part is a vector wᵢ over (k,i) with appropriate signs

But the signs in the C combination mean that wᵢ is NOT just [1,1,1,1] - it depends on which output element we're computing.

Actually, I think I'm overcomplicating this. In the tensor decomposition T = Σ uᵣ ⊗ vᵣ ⊗ wᵣ, each term contributes to ALL (i,j,k) simultaneously. The coefficient T[i,j,k] = Σᵣ uᵣ[i,j] · vᵣ[j,k] · wᵣ[k,i] must equal 1 for all i,j,k.

So my `make_laderman()` should work if the vectors are correct. Let me just ensure the numerical verification passes. If it doesn't, the fallback to rank-8 (basis terms) will be used, giving total rank 8+8+4+1 = 21, which is still better than the naive 30.

Actually, let me reconsider the whole approach. The current program scores 1.0, which means it's producing rank 30 (the naive decomposition) because the verification is failing. My fix should at least get to rank 19 or 21.

Let me make the Laderman verification more robust and add a search for a valid rank-7 decomposition if the hardcoded one fails.

