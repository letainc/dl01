# Linear Algebra for ML — Quick Cheat Sheet

Use this as a **memory refresher**, not as the full lesson.

---

## 1. Vector = one observation in feature space

Example:

[
x=
\begin{bmatrix}
2\\
300
\end{bmatrix}
]

Interpretation:

```text
2   = order_item_quantity
300 = order_value
```

If a dataset has (n) features:

[
x\in\mathbb{R}^n
]

A vector can be viewed as both:

```text
a point in feature space
and
a direction/displacement from the origin
```

---

## 2. Magnitude and distance

Magnitude:

[
\|x\|_2
=
\sqrt{x_1^2+x_2^2+\cdots+x_n^2}
]

Distance:

[
d(x,y)
=
\|x-y\|_2
]

Example:

[
x=(2,300),\qquad y=(5,900)
]

Then:

[
y-x=(3,600)
]

Important ML lesson:

> Distance depends on feature units.

Changing dollars to cents changes Euclidean geometry even if the real-world observation is unchanged.

---

## 3. Linear combination, span, basis

Linear combination:

[
a v_1+b v_2
]

Example:

[
2
\begin{bmatrix}
1\\0
\end{bmatrix}
+
3
\begin{bmatrix}
0\\1
\end{bmatrix}
=
\begin{bmatrix}
2\\3
\end{bmatrix}
]

**Span** = all vectors reachable by linear combinations.

**Basis** = independent vectors sufficient to describe the space.

Standard basis:

[
e_1=
\begin{bmatrix}
1\\0
\end{bmatrix},
\qquad
e_2=
\begin{bmatrix}
0\\1
\end{bmatrix}
]

Coordinates are coefficients relative to a basis.

---

## 4. Matrix = linear transformation

For:

[
A=
\begin{bmatrix}
2&0\\
0&0.5
\end{bmatrix}
]

and:

[
x=
\begin{bmatrix}
2\\
300
\end{bmatrix}
]

then:

[
Ax=
\begin{bmatrix}
4\\
150
\end{bmatrix}
]

Interpretation:

```text
x-coordinate → ×2
y-coordinate → ×0.5
```

Matrix columns tell us where the basis vectors go:

[
Ae_1=\text{first column of }A
]

[
Ae_2=\text{second column of }A
]

---

## 5. Matrix shapes

If:

[
A\in\mathbb{R}^{m\times n}
]

then:

[
A:\mathbb{R}^n\rightarrow\mathbb{R}^m
]

Remember:

```text
columns = input dimensions
rows    = output dimensions
```

Example:

[
A\in\mathbb{R}^{3\times2}
]

maps:

[
\mathbb{R}^2\rightarrow\mathbb{R}^3
]

but rank can still be at most 2.

---

## 6. Matrix multiplication = composition

[
ABx
]

means:

```text
apply B first
then A
```

Usually:

[
AB\neq BA
]

because transformation order matters.

---

## 7. Determinant = signed area/volume scaling

For:

[
A=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix}
]

[
\det(A)=ad-bc
]

Interpretation:

[
|\det(A)|
=
\text{area scale factor}
]

Key cases:

```text
det(A) = 1   → area preserved
det(A) < 0   → orientation flipped
det(A) = 0   → dimension collapsed
```

If:

[
\det(A)=0
]

then (A) is not invertible.

---

## 8. Inverse

If:

[
Ax=y
]

and (A) is invertible:

[
x=A^{-1}y
]

Interpretation:

> (A^{-1}) undoes (A).

If multiple inputs collapse to the same output, the transformation cannot be uniquely undone.

---

## 9. Rank and null space

**Rank** = number of independent output directions.

[
\operatorname{rank}(A)
]

**Null space**:

[
\mathcal{N}(A)
=
\{x:Ax=0\}
]

Interpretation:

> Null-space directions are variations that the transformation cannot see.

Projection example:

[
P=
\begin{bmatrix}
1&0\\
0&0
\end{bmatrix}
]

The entire (y)-axis is invisible to (P).

---

## 10. Dot product

[
v\cdot w
=
v^T w
]

Coordinate form:

[
v\cdot w
=
\sum_i v_iw_i
]

Geometric form:

[
v\cdot w
=
\|v\|\|w\|\cos\theta
]

Interpretation:

```text
positive → aligned
zero     → perpendicular
negative → opposing
```

For unit (w):

[
v\cdot w
]

is the signed scalar projection of (v) onto (w).

ML connection:

[
w^Tx+b
]

is both a weighted sum and a geometric linear score.

---

## 11. Orthogonal transformations preserve Euclidean geometry

A square matrix (Q) is orthogonal when:

[
Q^TQ=I
]

Then:

[
(Qv)\cdot(Qw)=v\cdot w
]

and it preserves:

```text
length
distance
angle
dot product
orthogonality
```

Rotations and reflections are typical examples.

General scaling is not orthogonal.

---

## 12. Z-score changes geometry

[
z=
\frac{x-\mu}{\sigma}
]

Think:

```text
subtract mean
→ center

divide by standard deviation
→ rescale axes
```

This changes distances, angles, dot products, covariance geometry, and optimization behavior.

That is why feature scaling matters.

---

## 13. Change of basis

Suppose (P) contains new basis vectors as columns:

[
P=[b_1\;b_2\;\cdots]
]

Then:

[
[x]_{old}=P[x]_{new}
]

and:

[
[x]_{new}=P^{-1}[x]_{old}
]

Same geometric vector, different coordinates.

For an orthonormal basis:

[
P^{-1}=P^T
]

---

## 14. Same transformation, different basis

If (A_E) is the matrix in basis (E), then in basis (B):

[
A_B=P^{-1}A_EP
]

Read right to left:

```text
P      → B coordinates to E coordinates
A_E    → apply transformation
P⁻¹    → E coordinates back to B
```

A matrix is a coordinate representation of a transformation, not the transformation itself.

---

## 15. Eigenvectors and eigenvalues

[
Av=\lambda v
]

Interpretation:

```text
v       = special direction
λ       = scale factor along that direction
```

The transformation does not rotate (v) away from its own span.

Cases:

```text
λ > 1       → stretch
0 < λ < 1   → shrink
λ < 0       → flip + scale
λ = 0       → collapse
```

---

## 16. Eigenbasis and diagonalization

If there are enough independent eigenvectors, put them into:

[
P=[v_1\;v_2\;\cdots]
]

Then:

[
D=P^{-1}AP
]

where (D) is diagonal:

[
D=
\begin{bmatrix}
\lambda_1&0&\cdots\\
0&\lambda_2&\cdots\\
\vdots&\vdots&\ddots
\end{bmatrix}
]

Meaning:

> In the eigenbasis, each eigendirection scales independently.

Repeated transformation:

[
A^n v=\lambda^n v
]

and if:

[
A=PDP^{-1}
]

then:

[
A^n=PD^nP^{-1}
]

---

## 17. Trace and determinant

For a (2\times2) matrix with eigenvalues (lambda_1,lambda_2):

[
\operatorname{tr}(A)
=
\lambda_1+\lambda_2
]

[
\det(A)
=
\lambda_1\lambda_2
]

Trace = eigenvalue sum.

Determinant = eigenvalue product.

---

## 18. PCA connection

For covariance matrix (C):

[
Cv_i=\lambda_i v_i
]

Interpretation:

```text
v_i       → principal direction
lambda_i  → variance along that direction
```

PCA:

```text
covariance
→ eigenvectors
→ rotate into principal directions
→ optionally drop low-variance directions
```

Keeping all components = orthonormal basis change.

Dropping components = dimensionality reduction and information loss.

---

## 19. (X^TX), Hessian, conditioning

For centered data, (X^TX) is closely related to covariance.

Its eigenvectors reveal important feature-space directions.

Later in Calculus:

[
Hv=\lambda v
]

for Hessian (H).

Interpret:

```text
eigenvector → curvature direction
eigenvalue  → curvature strength
```

Large spread between eigenvalues:

[
\lambda_{max}\gg\lambda_{min}
]

usually means poor conditioning and more difficult gradient descent.

---

## 20. Python ↔ math

NumPy:

```python
A @ v
```

means:

[
Av
]

---

```python
v @ w
```

means:

[
v\cdot w
]

---

```python
A @ B
```

means:

[
AB
]

---

Useful functions:

```python
np.linalg.norm(v)
np.linalg.det(A)
np.linalg.inv(A)
np.linalg.matrix_rank(A)
np.linalg.solve(A, b)
np.linalg.eig(A)
np.linalg.eigvals(A)
np.linalg.svd(A)
```

For ML data stored as rows:

[
X=
\begin{bmatrix}
x_1^T\\
x_2^T\\
\vdots
\end{bmatrix}
]

a column-vector transformation (A) is commonly applied as:

```python
X @ A.T
```

---

# One-page mental model

```text
real observation
→ feature vector
→ feature space
→ geometry
→ basis
→ matrix transformation
→ determinant / rank / null space
→ dot product / projection
→ change of basis
→ eigenvectors / eigenvalues
→ eigenbasis / diagonalization
→ covariance / PCA / X^T X / Hessian
→ ML geometry and optimization
```

The central ML lesson:

> Feature representation determines geometry, and geometry influences what many models learn and how optimization behaves.

---

## Learn later when needed

The current foundation is enough to proceed to Calculus and return to Feature Engineering.

Add these topics later when the model or problem requires them:

```text
SVD
least squares
pseudoinverse
positive-semidefinite matrices
spectral theorem / spectral decomposition
matrix norms
```
