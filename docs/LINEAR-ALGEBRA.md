# 📐 Linear Algebra for Machine Learning

## Overview

Linear algebra is the mathematical foundation of machine learning. Understanding it deeply improves model design and debugging.

---

## 🏗️ Fundamental Concepts

### Vectors
- **Definition**: Ordered collection of numbers
- **Notation**: Column vector **v** or row vector **v**ᵀ
- **Operations**: Addition, scalar multiplication, dot product
- **Norms**: L1, L2, L∞ norms
- **Properties**: Magnitude, direction, orthogonality

### Matrices
- **Definition**: Rectangular array of numbers
- **Shape**: (m × n) for m rows, n columns
- **Operations**: Addition, multiplication, transpose, inverse
- **Determinant**: Measure of invertibility
- **Trace**: Sum of diagonal elements

### Tensors
- **0D**: Scalar (number)
- **1D**: Vector (list)
- **2D**: Matrix (table)
- **3D+**: Higher-order tensors
- **Operations**: Element-wise, contractions, outer products

---

## 📊 Matrix Decompositions

### Singular Value Decomposition (SVD)
**Formula**: A = UΣVᵀ
- **U**: Left singular vectors
- **Σ**: Singular values (diagonal)
- **V**: Right singular vectors
- **Applications**: Dimensionality reduction, image compression, low-rank approximation
- **Properties**: Always exists, works for rectangular matrices

### Eigenvalue Decomposition
**Formula**: A = PDP⁻¹
- **P**: Matrix of eigenvectors
- **D**: Diagonal matrix of eigenvalues
- **Condition**: Works for square matrices
- **Applications**: PCA, power iteration, stability analysis
- **Properties**: Real eigenvalues for symmetric matrices

### QR Decomposition
**Formula**: A = QR
- **Q**: Orthogonal matrix (QᵀQ = I)
- **R**: Upper triangular matrix
- **Applications**: Solving linear systems, least squares, numerical stability
- **Stability**: More numerically stable than normal equations

### Cholesky Decomposition
**Formula**: A = LLᵀ (for positive definite A)
- **L**: Lower triangular matrix
- **Conditions**: A must be symmetric positive definite
- **Applications**: Solving linear systems, simulation, optimization
- **Efficiency**: 2x faster than LU decomposition

### LU Decomposition
**Formula**: A = LU
- **L**: Lower triangular (with 1s on diagonal)
- **U**: Upper triangular
- **Applications**: Solving linear systems, computing determinants
- **Efficiency**: O(n³) computation

### Polar Decomposition
**Formula**: A = UP
- **U**: Unitary matrix (preserves lengths)
- **P**: Positive semidefinite matrix
- **Applications**: Rigid transformations, orientation extraction

---

## 🔍 Spectral Analysis

### Eigenvalues & Eigenvectors
- **Definition**: Av = λv (v is eigenvector, λ is eigenvalue)
- **Geometric**: Direction that doesn't change under transformation
- **Computation**: Solve det(A - λI) = 0
- **Applications**: Stability analysis, vibration analysis, PageRank

### Spectral Theorem
- For symmetric matrix A: A = PDP⁻¹ where P is orthogonal
- **Corollary**: Symmetric matrices have real eigenvalues & orthogonal eigenvectors
- **Applications**: Optimization, quadratic forms

### Power Iteration
- Find largest eigenvalue iteratively
- **Algorithm**: x_{n+1} = Ax_n / ||Ax_n||
- **Convergence**: Exponential
- **Applications**: PageRank, recommendation systems

### Condition Number
- **κ(A) = σ_max / σ_min** (ratio of largest to smallest singular value)
- **Interpretation**: Sensitivity to input perturbations
- **Well-conditioned**: κ(A) ≈ 1
- **Ill-conditioned**: κ(A) is large (numerical instability)

---

## 🎯 Norms & Distances

### Vector Norms
- **L1 Norm**: ||v||₁ = Σ|vᵢ| (Manhattan distance)
- **L2 Norm**: ||v||₂ = √(Σvᵢ²) (Euclidean distance)
- **L∞ Norm**: ||v||∞ = max|vᵢ| (Chebyshev distance)
- **Lp Norm**: ||v||_p = (Σ|vᵢ|^p)^(1/p)

### Matrix Norms
- **Frobenius**: ||A||_F = √(Σᵢⱼ aᵢⱼ²)
- **Spectral**: ||A||₂ = σ_max(A)
- **Nuclear**: ||A||₊ = Σ σᵢ (sum of singular values)

### Distance Metrics
- **Euclidean**: √(Σ(xᵢ - yᵢ)²)
- **Manhattan**: Σ|xᵢ - yᵢ|
- **Cosine**: 1 - (x·y)/(||x||||y||)
- **Mahalanobis**: √((x-μ)ᵀΣ⁻¹(x-μ))

---

## 🧮 Optimization & Calculus

### Gradients & Jacobians
- **Gradient**: ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]
- **Jacobian**: Matrix of all first-order partial derivatives
- **Hessian**: Matrix of second-order partial derivatives
- **Chain Rule**: For composite functions

### Convexity
- **Convex Function**: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
- **Convex Set**: All points between any two points are in set
- **Convex Optimization**: Global optimum guaranteed
- **Applications**: Regression, SVM, deep learning loss

### Quadratic Forms
- **Definition**: x^T A x (where A is symmetric)
- **Positive Definite**: x^T A x > 0 for all x ≠ 0
- **Applications**: Covariance matrices, regularization terms
- **Properties**: Eigenvalues determine definiteness

### Matrix Calculus Rules
- ∇_X tr(AX) = A^T
- ∇_X tr(X^T A X) = 2AX
- ∇_X ||AX - b||² = 2A^T(AX - b)
- ∇_X log det(X) = (X^T)^(-1)

---

## 📈 Linear Systems & Solutions

### Solving Ax = b
1. **Square matrices (n×n)**: x = A⁻¹b (if invertible)
2. **Overdetermined (m>n)**: Least squares x = (A^T A)⁻¹ A^T b
3. **Underdetermined (m<n)**: Minimum norm solution
4. **Singular**: Use pseudo-inverse A⁺

### Pseudo-Inverse (Moore-Penrose)
- **Definition**: A⁺ = V Σ⁺ U^T (from SVD)
- **Properties**: Always exists, generalizes inverse
- **Least Squares**: Minimizes ||Ax - b||²
- **Minimum Norm**: Among solutions with minimum norm

### Regularization
- **Ridge (L2)**: (A^T A + λI)⁻¹ A^T b
- **Lasso (L1)**: Sparse solutions via optimization
- **Elastic Net**: Combination of L1 & L2

---

## 🎓 Applications in ML

### Principal Component Analysis (PCA)
- Eigendecomposition of covariance matrix
- Find directions of maximum variance
- Dimensionality reduction

### Singular Value Decomposition Applications
- **SVD for recommendations**: User-item matrix factorization
- **Image compression**: Keep top k singular values
- **Noise reduction**: Remove small singular values

### Kernel Methods
- **Kernel Matrix**: K_ij = k(x_i, x_j)
- **Gram Matrix**: X X^T
- **Properties**: Symmetric, positive semi-definite
- **Applications**: SVM, kernel ridge regression, Gaussian processes

### Neural Networks
- **Weight matrices**: Feature transformation
- **Initialization**: Random matrices with specific distributions
- **Backpropagation**: Chain rule through matrix multiplications

---

## 🚀 Computational Considerations

### Numerical Stability
- Use QR decomposition instead of normal equations
- Avoid computing inverse explicitly
- Use stable algorithms for eigendecomposition

### Computational Complexity
- **Matrix multiplication**: O(n³) for dense
- **SVD**: O(mn²) for m≥n
- **Eigendecomposition**: O(n³)
- **Sparse operations**: Much faster for sparse matrices

### Libraries
- **NumPy**: Dense linear algebra
- **SciPy**: Advanced operations
- **scikit-learn**: ML-focused
- **CuPy**: GPU-accelerated (CUDA)
- **JAX**: Automatic differentiation + linear algebra

---

## 📚 Important Theorems

- **Spectral Theorem**: Symmetric matrices diagonalizable
- **Singular Value Theorem**: Every matrix has SVD
- **Rank-Nullity**: rank(A) + nullity(A) = n
- **Cayley-Hamilton**: Every matrix satisfies its characteristic equation

---

*Detailed implementations and applications in projects folder.*
