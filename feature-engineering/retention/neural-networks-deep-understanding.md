# Neural Networks — Deep Understanding Notebook

## Program navigation

This notebook is the deep neural-network foundation within the broader learning program.

Master syllabus:

- [ML Deep Understanding — Master Learning Syllabus](./ml-deep-understanding-master-syllabus.md)

Use the master syllabus for overall progression, source links, CS229 / CS231n integration, Python practice rules, modernization policy, and mastery checkpoints.

---

> Durable notes for the ML / Feature Engineering learning program.
>
> Teaching rule: intuition → visualization → simple numerical example → mathematics → manual calculation → Linear Algebra connection → Calculus connection → ML / Feature Engineering → engineering implementation → limitations → future concepts.

## Learning map

```text
vectors / matrices
        ↓
linear transformations
        ↓
affine transformations
        ↓
nonlinear activation functions
        ↓
composed neural-network function
        ↓
loss / cost function
        ↓
partial derivatives
        ↓
gradient
        ↓
gradient descent
        ↓
backpropagation
        ↓
optimization
        ↓
trained model
```

---

# 1. What a basic neural network really is

A basic dense neural network is best understood as a giant parameterized function.

For an input vector (x), one layer computes:

```text
z = Wx + b
a = f(z)
```

where:

- (x): incoming representation;
- (W): learned weight matrix;
- (b): learned bias vector;
- (z): affine-transformed values before activation;
- (f): nonlinear activation such as ReLU or GELU;
- (a): outgoing representation.

A whole network composes many such transformations:

```text
x
↓
W₁x + b₁
↓
activation
↓
a₁
↓
W₂a₁ + b₂
↓
activation
↓
a₂
↓
...
↓
prediction
```

The network is therefore not mysterious "circles talking to each other". It is a large composition of vector/matrix operations and nonlinear functions.

## Linear Algebra connection

The term (Wx) is a matrix-vector multiplication.

A matrix acts as a linear transformation. Depending on the matrix, it can mix coordinates, stretch, compress, rotate, reflect, or shear the representation.

The bias (b) then shifts the result.

That makes (Wx+b) an **affine transformation**, not strictly a linear transformation.

---

# 2. Linear transformation vs affine transformation

A linear transformation is:

```text
y = Wx
```

A defining property is that the origin stays fixed:

```text
x = 0
→
Wx = 0
```

An affine transformation is:

```text
y = Wx + b
```

Now:

```text
x = 0
→
y = b
```

so the origin can move.

### One-dimensional example

Linear:

```text
y = 2x

x=0 → y=0
x=1 → y=2
x=2 → y=4
```

Affine:

```text
y = 2x + 3

x=0 → y=3
x=1 → y=5
x=2 → y=7
```

The multiplication stretches; the bias translates.

In a neural network, people often casually call (Wx+b) a "linear layer", but mathematically it is affine when the bias is enabled.

---

# 3. Why activation functions are necessary

If every layer were only affine, stacking layers would not give the network the expressive power we want.

Consider two layers:

```text
z₁ = W₁x + b₁
y  = W₂z₁ + b₂
```

Substitute the first into the second:

```text
y = W₂(W₁x + b₁) + b₂
  = W₂W₁x + W₂b₁ + b₂
```

Define:

```text
W₃ = W₂W₁
b₃ = W₂b₁ + b₂
```

Then:

```text
y = W₃x + b₃
```

So many affine layers without nonlinear operations collapse mathematically into one affine transformation.

Depth alone would therefore add little representational power.

## Geometry

Matrices can transform space by stretching, rotating, shearing, compressing, reflecting, and mixing coordinates.

A nonlinear activation allows the representation to be bent, gated, or reshaped in a way that cannot be represented by one matrix multiplication.

Mental model:

```text
affine transformation
→ stretch / rotate / mix / shift

nonlinearity
→ bend / gate / reshape

affine
→ nonlinear
→ affine
→ nonlinear
→ ...
```

This is what allows a network to learn complicated decision boundaries and useful hidden representations.

---

# 4. ReLU and GELU

## ReLU

```text
ReLU(x) = max(0, x)
```

Examples:

| x | ReLU(x) |
|---:|---:|
| -3 | 0 |
| -1 | 0 |
| 0 | 0 |
| 1 | 1 |
| 3 | 3 |

Mental model:

```text
negative → suppress completely
positive → pass unchanged
```

It acts as a hard gate.

## GELU

GELU means **Gaussian Error Linear Unit**.

Conceptually:

```text
GELU(x) = x Φ(x)
```

where (Phi(x)) is the cumulative distribution function of the standard normal distribution.

Approximate values:

| x | ReLU(x) | GELU(x) |
|---:|---:|---:|
| -3 | 0 | -0.004 |
| -1 | 0 | -0.159 |
| 0 | 0 | 0 |
| 1 | 1 | 0.841 |
| 3 | 3 | 2.996 |

Mental model:

```text
very negative
→ almost suppress

slightly negative
→ preserve a little negative signal

near zero
→ smooth transition

positive
→ increasingly preserve

very positive
→ almost fully preserve
```

ReLU is a hard gate; GELU is a smooth input-dependent gate.

Both provide the essential nonlinearity that prevents the network from collapsing into one affine transformation.

---

# 5. Data, parameters, prediction, and target

A useful notation is:

```text
X = input data
Y = target / correct answer
W = weight matrices
b = bias vectors
ŷ = prediction
```

The network computes:

```text
ŷ = F(X; W, b)
```

The semicolon is conceptually useful:

```text
F(X ; W,b)
  │     │
  │     └─ learned parameters controlling the function
  └─────── input data
```

(X) and (Y) come from the dataset.

(W) and (b) are learned.

A multi-layer network has many parameter tensors:

```text
W₁, b₁
W₂, b₂
W₃, b₃
...
```

It is common to bundle all trainable parameters into one symbol:

```text
θ = {W₁,b₁,W₂,b₂,...}
```

Then the network can be written compactly as:

```text
ŷ = F(X; θ)
```

---

# 6. Why a loss function is needed

A model cannot optimize the vague statement:

> "This prediction is wrong."

We need a numerical function that measures how bad a prediction is.

A loss function maps:

```text
prediction + target
        ↓
one number
```

with the intended behavior:

```text
good prediction
→ small loss

bad prediction
→ large loss
```

This is critical because calculus operates on functions and numerical change.

The loss turns "model quality" into something differentiable that we can reason about mathematically.

## Example: squared error

Suppose a scalar target is:

```text
y = 5
```

and our model predicts:

```text
ŷ = 8
```

A simple squared loss is:

```text
L = (ŷ - y)²
  = (8 - 5)²
  = 9
```

If the prediction improves to 6:

```text
L = (6 - 5)²
  = 1
```

The numerical loss has fallen from 9 to 1.

For classification, modern systems commonly use losses such as cross-entropy rather than plain squared error. The purpose is still the same: create a numerical objective whose decrease corresponds to improved predictions.

---

# 7. Cost / objective over many examples

For one training example we may write:

```text
Lᵢ = loss for example i
```

For many examples we often aggregate them:

```text
C(θ) = average loss over training examples
```

A simplified expression is:

```text
C(θ) = (1/n) Σᵢ L(F(xᵢ; θ), yᵢ)
```

Read this from the inside outward:

1. take training input (x_i);
2. run it through the network (F);
3. get prediction (hat y_i);
4. compare it with target (y_i);
5. get the example's loss;
6. repeat across examples;
7. average.

Now training has become an optimization problem:

> Find parameter values (	heta) that make (C(	heta)) small.

This is the key conceptual bridge:

```text
"train the neural network"
        =
"find good values of W and b"
        =
"minimize a cost function over parameter space"
```

---

# 8. The cost landscape

Start with only one parameter (w).

Imagine the cost is:

```text
C(w) = (w - 3)²
```

The minimum occurs at (w=3).

Values:

| w | C(w) |
|---:|---:|
| 0 | 9 |
| 1 | 4 |
| 2 | 1 |
| 3 | 0 |
| 4 | 1 |
| 5 | 4 |

This creates a bowl-shaped curve.

Training means moving the parameter along this curve toward a lower value of the cost.

With two parameters (w_1,w_2), the cost becomes a surface.

With millions or billions of parameters, it becomes a high-dimensional landscape that we cannot directly visualize, but the mathematics is the same.

The important change of viewpoint is:

```text
the network parameters are coordinates
in parameter space

the cost function gives a height
at each coordinate
```

So a parameter vector (	heta) is a **position** in parameter space.

---

# 9. Calculus connection: derivative as local sensitivity

For one variable:

```text
C(w)
```

the derivative:

```text
dC/dw
```

means:

> Around the current value of (w), how sensitive is the cost to a small change in (w)?

Example:

```text
C(w) = (w - 3)²
```

Differentiate:

```text
dC/dw = 2(w - 3)
```

At (w=5):

```text
dC/dw = 2(5-3)
      = 4
```

Interpretation:

A small positive change in (w) tends to increase the cost.

So if we want to lower the cost, we should move (w) in the negative direction.

At (w=1):

```text
dC/dw = 2(1-3)
      = -4
```

The derivative is negative.

Increasing (w) now decreases the cost.

This is why the sign of the derivative gives a local direction.

---

# 10. From derivative to partial derivatives

A neural network has many parameters.

Instead of:

```text
C(w)
```

we have:

```text
C(w₁,w₂,w₃,...,b₁,b₂,...)
```

We therefore need a partial derivative for every coordinate:

```text
∂C/∂w₁
∂C/∂w₂
∂C/∂w₃
...
∂C/∂b₁
...
```

A partial derivative asks:

> If I change this one parameter slightly while locally holding the others fixed, how does the cost change?

These local sensitivities are gathered into one vector: the gradient.

---

# 11. The gradient is a vector

For a function of two parameters:

```text
C(w₁,w₂)
```

the gradient is:

```text
∇C =
[
  ∂C/∂w₁,
  ∂C/∂w₂
]
```

This is a vector in the same parameter space in which ([w₁,w₂]) is a position vector.

That is the crucial Linear Algebra connection:

```text
parameter vector θ
= a position in parameter space

gradient ∇C
= a direction vector in the same space
```

---

# 12. Why the gradient points toward steepest increase

This should not be memorized as a magical theorem.

For a tiny movement vector (Delta 	heta), the local change in the cost is approximately:

```text
ΔC ≈ ∇C · Δθ
```

The right-hand side is a **dot product**.

Recall the Linear Algebra meaning:

```text
a · b = ||a|| ||b|| cos(φ)
```

So:

```text
ΔC ≈ ||∇C|| ||Δθ|| cos(φ)
```

If we fix the step length (||Delta	heta||), the only thing we choose is the angle (phi).

### Largest increase

When:

```text
φ = 0°
cos(φ) = 1
```

the movement points in the same direction as the gradient.

Then the local increase in the function is largest.

### Largest decrease

When:

```text
φ = 180°
cos(φ) = -1
```

the movement points directly opposite the gradient.

Then the local change is as negative as possible for that small fixed step.

Therefore:

```text
gradient ∇C
→ steepest local increase

negative gradient -∇C
→ steepest local decrease
```

This is where gradient descent comes from.

It is a direct consequence of the local linear approximation plus the geometry of the dot product.

---

# 13. Gradient descent update

The basic update is:

```text
θ_new = θ_old - η ∇C(θ_old)
```

where:

```text
η = learning rate
```

The gradient gives direction and relative sensitivities.

The learning rate controls how large a step we take.

## Manual example

Again use:

```text
C(w) = (w - 3)²
```

Derivative:

```text
dC/dw = 2(w-3)
```

Start at:

```text
w₀ = 7
η  = 0.1
```

### Step 1

Gradient:

```text
2(7-3) = 8
```

Update:

```text
w₁ = 7 - 0.1×8
   = 6.2
```

Cost:

```text
C(7)   = 16
C(6.2) = 10.24
```

The parameter moved toward 3 and the cost fell.

### Step 2

Gradient:

```text
2(6.2-3) = 6.4
```

Update:

```text
w₂ = 6.2 - 0.1×6.4
   = 5.56
```

Cost:

```text
C(5.56) = 6.5536
```

Again the cost decreases.

As (w) gets closer to 3, the gradient becomes smaller, so the update naturally shrinks for this simple quadratic.

---

# 14. Learning rate as step-size engineering

The learning rate (eta) does not tell us the downhill direction. The gradient already does that.

The learning rate determines how aggressively we follow that direction.

### Too small

```text
tiny steps
→ stable
→ but potentially extremely slow
```

### Too large

```text
large steps
→ may overshoot
→ oscillate
→ or diverge
```

### Appropriate scale

```text
large enough to make progress
small enough to remain stable
```

Modern optimizers such as Adam and AdamW modify how these updates are scaled across parameters and across time, but they still rely on gradients.

---

# 15. Gradient descent vs backpropagation

These are different concepts.

## Backpropagation

Answers:

> How do we efficiently compute all the derivatives of the loss with respect to all parameters?

It uses the chain rule repeatedly through the computational graph.

## Gradient descent / optimizer

Answers:

> Once we have those gradients, how do we change the parameters?

```text
backpropagation
→ computes gradients

optimizer
→ uses gradients to update parameters
```

Do not merge these concepts mentally.

---

# 16. Full-batch vs mini-batch training

The clean mathematical cost may average over the entire training dataset:

```text
C(θ) = average over all training samples
```

For large datasets, recomputing the exact full-dataset gradient for every single update is expensive.

Modern training typically uses mini-batches:

```text
take a batch
↓
forward pass
↓
batch loss
↓
backpropagation
↓
gradient estimate
↓
optimizer update
↓
next batch
```

The mini-batch gradient is a noisy estimate of the full-data gradient, but it is vastly more computationally practical.

This is the bridge from the mathematical idealization to engineering reality.

---

# 17. Numerical gradient vs analytic gradient

A derivative can be estimated numerically:

```text
dC/dw ≈ [C(w+h)-C(w)] / h
```

This directly connects to the finite-difference and secant-slope ideas from Calculus.

But doing this independently for millions of parameters would require enormous numbers of forward evaluations.

So production neural-network training uses analytic/automatic differentiation and backpropagation.

Numerical gradients remain useful for **gradient checking**: testing whether an analytic gradient implementation is likely correct.

---

# 18. Feature Engineering connection

Gradient-based training does not make Feature Engineering irrelevant.

The geometry and scale of input features can materially affect optimization.

Examples:

- badly scaled numeric features can create poorly conditioned optimization geometry;
- standardized features often make gradient-based optimization easier;
- leakage can reduce training loss while destroying real-world validity;
- transformations change the representation the model receives and therefore change the loss landscape being optimized.

So our earlier Feature Engineering work connects here:

```text
feature representation
        ↓
input geometry
        ↓
model computation
        ↓
loss landscape
        ↓
gradient behavior
        ↓
optimization
```

This is why scaling, leakage control, transformations, and data quality are engineering concerns rather than preprocessing trivia.

---

# 19. What gradient descent does NOT guarantee

Gradient descent is a local method.

Important limitations:

1. A small gradient does not automatically mean we found the globally best parameters.
2. Non-convex neural-network landscapes can contain saddles, flat regions, and complicated curvature.
3. Poor learning rates can make training unstable or painfully slow.
4. The training objective can be minimized while validation performance worsens: overfitting.
5. If the loss does not encode the real product objective well, optimizing it efficiently can still produce the wrong behavior.
6. Bad or leaked training data can yield low loss and a useless production model.

Optimization solves the objective we give it, not the real-world problem automatically.

---

# 20. Engineering implementation map

```text
mathematical idea
→ computational representation
→ algorithm
→ implementation
→ behavior
```

### Mathematical idea

```text
minimize C(θ)
```

### Computational representation

Parameters are tensors in memory.

```text
W₁, b₁, W₂, b₂, ...
```

### Forward algorithm

```text
X
→ affine transformation
→ activation
→ ...
→ prediction
→ loss
```

### Backward algorithm

Automatic differentiation traverses the computation backward using chain-rule derivatives.

### Parameter update

Conceptually:

```python
for parameter in model.parameters():
    parameter -= learning_rate * parameter.grad
```

Frameworks implement this with optimizers and automatic differentiation.

### Production concerns

Engineers monitor:

- training loss;
- validation loss;
- gradient magnitudes;
- learning-rate schedule;
- batch size;
- training throughput;
- numerical stability;
- convergence;
- overfitting;
- checkpointing;
- reproducibility.

---

# 21. Mental model to preserve

Do not think:

```text
"AI somehow learns patterns"
```

Think:

```text
data X
↓
parameterized differentiable function F(X;θ)
↓
prediction ŷ
↓
loss L(ŷ,Y)
↓
local sensitivities ∇θL
↓
move θ opposite those sensitivities
↓
repeat
```

The mathematics is connected:

```text
vectors
↓
parameter vector θ

dot product
↓
why -gradient is steepest local descent

matrices
↓
weight tensors / linear transformations

functions
↓
network and loss composition

derivatives
↓
local sensitivity

partial derivatives
↓
one sensitivity per parameter

gradient
↓
all partial derivatives as one direction vector

optimization
↓
changing parameters to reduce loss
```

---

# 22. Future concepts this unlocks

This foundation is required for understanding:

- backpropagation;
- computational graphs;
- automatic differentiation;
- stochastic gradient descent;
- momentum;
- Adam / AdamW;
- learning-rate schedules;
- vanishing and exploding gradients;
- initialization;
- normalization;
- regularization;
- Hessians and curvature;
- conditioning;
- convergence;
- transformer training;
- fine-tuning;
- feature scaling effects on optimization.

The next deep lesson should be **backpropagation as chain-rule accounting through a computational graph**.

---

# Sources / anchors

- 3Blue1Brown, *Gradient descent, how neural networks learn*: https://www.3blue1brown.com/lessons/gradient-descent/
- Michael Nielsen, *Neural Networks and Deep Learning*, Chapter 1: https://neuralnetworksanddeeplearning.com/chap1
- Stanford CS231n, *Optimization*: https://cs231n.github.io/optimization-1/
- Stanford CS231n, *Backpropagation, Intuitions*: https://cs231n.github.io/optimization-2/


---

# Modernization Rule — 2026

The historical learning materials in this notebook are used for mathematical foundations, not as a claim that their example architectures are current frontier practice.

For every classical concept, the learning program must now add three explicit layers:

```text
FOUNDATION
What is the mathematical idea?

        ↓

MODERN TRANSLATION
Where does the same idea appear in current architectures?

        ↓

ENGINEERING REALITY
How is it actually implemented efficiently on current hardware/software?
```

## What remains foundational

The following ideas are not obsolete:

- vectors, matrices, tensors, dot products, and matrix multiplication;
- affine transformations;
- nonlinear functions;
- function composition;
- losses / objectives;
- derivatives and partial derivatives;
- gradients;
- chain rule and reverse-mode automatic differentiation;
- gradient-based optimization;
- representations and embeddings.

Current frameworks such as PyTorch still train differentiable neural networks by recording tensor operations in a computational graph and using reverse-mode automatic differentiation / backpropagation to compute parameter gradients.

## What has changed since early neural-network tutorials

### Activations

Older tutorials often emphasize sigmoid and ReLU.

Modern Transformer-family models commonly use smoother or gated feed-forward transformations such as GELU, SiLU/Swish, and especially gated variants such as SwiGLU.

The underlying purpose is unchanged:

```text
affine transformation
        ↓
nonlinear / gated transformation
        ↓
new representation
```

The implementation and gating strategy evolved; the mathematical need for nonlinearity did not.

### Architecture

A classical multilayer perceptron is useful for learning the foundations, but modern foundation models are usually built from Transformer-family blocks rather than simply stacking dense layers.

Important modern components include:

- token/patch embeddings;
- self-attention;
- residual connections;
- normalization, frequently RMSNorm or related variants;
- rotary or other relative positional mechanisms such as RoPE;
- feed-forward / gated MLP blocks;
- grouped-query, multi-query, or latent attention variants;
- Mixture-of-Experts (MoE) routing in many large models.

These mechanisms still reduce to combinations of tensor transformations, nonlinearities, normalization, routing, and differentiable objectives.

### Optimization and differentiation

We learn manual derivatives and gradient descent because they explain the mechanism.

In real engineering, frameworks normally use automatic differentiation. Backpropagation is not manually differentiated parameter-by-parameter.

Training commonly uses first-order optimizers such as AdamW-family methods, learning-rate schedules, gradient clipping, mixed precision, and distributed optimization.

### Hardware-aware algorithms

Modern AI engineering is not only about mathematical FLOPs.

Memory movement, communication, tensor layout, kernel fusion, and accelerator utilization matter enormously.

FlashAttention is a key example: it computes exact attention while reducing expensive memory traffic by using IO-aware tiling. Newer implementations further exploit modern GPU hardware and lower precision.

### Precision

Training and inference increasingly use lower-precision arithmetic such as BF16, FP16, FP8, and quantized integer/floating formats where appropriate.

The mathematical operation may remain "matrix multiply", but production engineering asks:

```text
Which precision?
Which memory hierarchy?
Which kernel?
Which parallelism strategy?
Which communication cost?
What numerical error is acceptable?
```

### Sparse / routed computation

Dense networks activate every parameter involved in a layer.

Modern MoE models can hold very large total parameter counts while routing each token through only a subset of experts.

Examples include DeepSeek-V3 and Llama 4.

The foundational view still applies:

```text
representation
      ↓
routing decision
      ↓
selected parameterized transformations
      ↓
new representation
```

but computation is sparse rather than fully dense.

### Attention efficiency

The original Transformer uses standard multi-head attention.

Modern systems may use:

- grouped-query attention (GQA);
- multi-query attention (MQA);
- latent/compressed attention variants such as MLA;
- KV caching for autoregressive inference;
- FlashAttention-style kernels;
- long-context positional and attention modifications.

The course should teach ordinary attention first because the modern variants are modifications of that foundation.

### Pretraining is only part of the system

Modern foundation-model development usually separates stages such as:

```text
pretraining
   ↓
supervised / instruction tuning
   ↓
preference or reinforcement-based post-training
   ↓
reasoning / tool-use / agent behavior
   ↓
inference-time strategies
```

A model's behavior therefore cannot be understood from architecture alone.

### Reasoning-time compute

Modern reasoning systems may deliberately spend different amounts of inference compute on different prompts.

The engineering question is no longer only:

> How large is the trained model?

It can also be:

> How much compute should this model spend on this problem at inference time?

Qwen3, for example, explicitly describes thinking and non-thinking modes with configurable thinking budgets.

## Current architecture examples to revisit later

These are not topics to memorize now. They are forward anchors.

### DeepSeek-V3

Important ideas to study later:

- Mixture-of-Experts;
- Multi-head Latent Attention (MLA);
- multi-token prediction;
- FP8 training;
- hardware-aware model/system co-design.

### Llama 4

Important ideas to study later:

- Mixture-of-Experts;
- sparse expert activation;
- multimodal modeling;
- inference efficiency.

### Qwen3

Important ideas to study later:

- dense and MoE model families;
- unified thinking / non-thinking behavior;
- inference-time thinking budgets;
- multilingual and agent-oriented evaluation.

## Course rule going forward

Every major topic should explicitly answer:

```text
1. What did the classical formulation teach us?

2. Is that formulation still used directly today?

3. If not, what replaced or extended it?

4. Which mathematical principle survived unchanged?

5. What modern engineering constraint caused the evolution?

6. Where does the idea appear in current LLMs,
   multimodal models, classical ML, or Feature Engineering?
```

Examples:

```text
sigmoid / ReLU
      ↓
GELU / SiLU / SwiGLU

plain LayerNorm
      ↓
RMSNorm and related normalization choices

absolute position encoding
      ↓
RoPE / relative-position approaches

dense Transformer
      ↓
GQA / MLA / MoE and other efficiency adaptations

naive attention implementation
      ↓
FlashAttention / fused hardware-aware kernels

manual derivative derivation
      ↓
autograd / reverse-mode AD in production

FP32 everywhere
      ↓
BF16 / FP8 / quantization where numerically safe
```

The purpose of the historical material is therefore to establish the invariant mathematics. The purpose of the modern track is to show how current AI systems reshape those invariants around scale, hardware, memory, data, and inference constraints.


---

# Partial Derivatives, Norms, Dot Products, and Why cos(theta) Appears

This section connects multivariable calculus directly back to vector geometry.

## 1. Partial derivative: sensitivity along one coordinate

For a function with several inputs:

```text
C(w1, w2)
```

a partial derivative asks:

> If I change only one coordinate by a tiny amount, while locally holding the others fixed, how does the function change?

Example:

```text
C(w1,w2) = (w1 - 3)^2 + 2(w2 + 1)^2
```

Then:

```text
∂C/∂w1 = 2(w1 - 3)
∂C/∂w2 = 4(w2 + 1)
```

At:

```text
(w1,w2) = (5,2)
```

we get:

```text
∂C/∂w1 = 4
∂C/∂w2 = 12
```

Interpretation:

- locally, increasing w1 by 0.01 changes C by about 4 × 0.01 = 0.04;
- locally, increasing w2 by 0.01 changes C by about 12 × 0.01 = 0.12.

The second coordinate is therefore locally more sensitive.

The gradient collects these partial derivatives:

```text
∇C = [4, 12]
```

This is a vector in parameter space.

## 2. Norm notation is not transpose

For a vector:

```text
a = [a1, a2, ..., an]
```

the notation:

```text
||a||
```

means the vector's magnitude / Euclidean length.

For example:

```text
a = [3,4]
||a|| = sqrt(3^2 + 4^2) = 5
```

This is the multidimensional analogue of absolute value.

Transpose is different notation:

```text
a^T
```

For column vectors a and b:

```text
a · b = a^T b
```

So:

```text
||a||   = length
a^T     = transpose
a · b   = dot product
```

## 3. Dot product: component formula

For:

```text
a = [a1,a2,...,an]
b = [b1,b2,...,bn]
```

the dot product is:

```text
a · b = a1 b1 + a2 b2 + ... + an bn
```

It produces one scalar.

## 4. Dot product: geometric formula

The same scalar can be written:

```text
a · b = ||a|| ||b|| cos(theta)
```

where theta is the angle between the two vectors.

The cosine appears because the dot product measures alignment.

```text
theta = 0 degrees
cos(theta) = 1
→ maximum positive alignment

theta = 90 degrees
cos(theta) = 0
→ perpendicular
→ zero alignment

theta = 180 degrees
cos(theta) = -1
→ opposite direction
→ maximum negative alignment
```

## 5. Why cosine appears: projection

The amount of vector a pointing along b's direction is:

```text
||a|| cos(theta)
```

Multiplying by ||b|| gives:

```text
||a|| ||b|| cos(theta)
```

which is the dot product.

So the dot product can be read as:

> how much one vector points in the direction of the other, scaled by their lengths.

If b is converted to a unit vector:

```text
b_hat = b / ||b||
```

then:

```text
a · b_hat = ||a|| cos(theta)
```

is the scalar projection of a onto b's direction.

## 6. Why the component and geometric dot-product formulas are the same

Use the Law of Cosines on vectors a, b, and a-b:

```text
||a-b||^2 = ||a||^2 + ||b||^2 - 2||a||||b||cos(theta)
```

Expand the left side algebraically:

```text
||a-b||^2
= (a-b) · (a-b)
= a·a - 2a·b + b·b
= ||a||^2 - 2a·b + ||b||^2
```

Equating both expressions gives:

```text
a · b = ||a||||b||cos(theta)
```

So cosine is not inserted arbitrarily; it comes from Euclidean geometry.

## 7. Gradient and directional change

For a small parameter movement:

```text
Δtheta
```

the loss changes approximately by:

```text
ΔC ≈ ∇C · Δtheta
```

Using the geometric form:

```text
ΔC ≈ ||∇C|| ||Δtheta|| cos(phi)
```

where phi is the angle between the gradient and the movement direction.

For a fixed step size ||Δtheta||:

- phi = 0 degrees gives the greatest local increase;
- phi = 90 degrees gives approximately no first-order change;
- phi = 180 degrees gives the greatest local decrease.

Therefore:

```text
∇C
→ steepest local ascent

-∇C
→ steepest local descent
```

This is the geometric foundation of gradient descent.

## 8. Neural-network interpretation

If:

```text
theta = [all weights and biases]
```

then:

```text
∇_theta L
```

contains one partial derivative for every trainable parameter.

Each component says:

> locally, how sensitive is the loss to this parameter?

The gradient vector combines all those coordinate-wise sensitivities into one geometric object that tells the optimizer how the loss changes in parameter space.


---

# Backpropagation — Reverse Sensitivity Through the Network

## Core purpose

Gradient descent tells us what to do once we know the gradient:

```text
theta_new = theta_old - learning_rate * gradient
```

Backpropagation solves the preceding problem:

> How do we efficiently compute the partial derivative of the loss with respect to every trainable weight and bias?

Backpropagation is the recursive application of the chain rule through the network's computation graph, starting from the scalar loss and moving backward toward earlier parameters.

## Forward pass vs backward pass

Forward:

```text
input
  ↓
affine transform
  ↓
activation
  ↓
affine transform
  ↓
prediction
  ↓
loss
```

Backward:

```text
loss sensitivity
  ↓
prediction sensitivity
  ↓
later-layer parameter sensitivities
  ↓
hidden representation sensitivity
  ↓
earlier-layer parameter sensitivities
```

The forward pass computes values.
The backward pass computes sensitivities.

## A tiny network

Use one scalar input:

```text
x = 2
target y = 30

w1 = 3
b1 = 1
w2 = 4
b2 = 0
```

Network:

```text
z1 = w1*x + b1
a1 = ReLU(z1)
y_hat = w2*a1 + b2
L = 1/2 * (y_hat - y)^2
```

### Forward pass

```text
z1 = 3*2 + 1 = 7
a1 = ReLU(7) = 7
y_hat = 4*7 + 0 = 28
L = 1/2 * (28 - 30)^2 = 2
```

## Backward pass

Start at the loss.

```text
dL/dy_hat = y_hat - y = -2
```

Interpretation:

Locally, increasing the prediction would decrease the loss because the prediction 28 is below the target 30.

### Gradient for w2

Because:

```text
y_hat = w2*a1 + b2
```

we have:

```text
dy_hat/dw2 = a1 = 7
```

Chain rule:

```text
dL/dw2
= dL/dy_hat * dy_hat/dw2
= (-2) * 7
= -14
```

### Gradient for b2

```text
dy_hat/db2 = 1

dL/db2
= dL/dy_hat * dy_hat/db2
= (-2) * 1
= -2
```

### Propagate sensitivity into a1

```text
dy_hat/da1 = w2 = 4

dL/da1
= dL/dy_hat * dy_hat/da1
= (-2) * 4
= -8
```

This is the key backprop idea: an upstream sensitivity is multiplied by the local derivative of the next operation.

### Through ReLU

Since:

```text
a1 = ReLU(z1)
```

and z1 = 7 > 0:

```text
da1/dz1 = 1
```

Therefore:

```text
dL/dz1
= dL/da1 * da1/dz1
= (-8) * 1
= -8
```

If z1 had been negative, ReLU's local derivative would be 0 and this path would transmit zero first-order gradient backward.

### Gradient for w1

Because:

```text
z1 = w1*x + b1
```

we have:

```text
dz1/dw1 = x = 2
```

Therefore:

```text
dL/dw1
= dL/dz1 * dz1/dw1
= (-8) * 2
= -16
```

### Gradient for b1

```text
dz1/db1 = 1

dL/db1
= dL/dz1 * dz1/db1
= -8
```

The parameter gradient is therefore:

```text
dL/dw1 = -16
dL/db1 = -8
dL/dw2 = -14
dL/db2 = -2
```

## One update

With learning rate:

```text
eta = 0.01
```

gradient descent gives:

```text
w1_new = 3 - 0.01*(-16) = 3.16
b1_new = 1 - 0.01*(-8)  = 1.08
w2_new = 4 - 0.01*(-14) = 4.14
b2_new = 0 - 0.01*(-2)  = 0.02
```

Run forward again:

```text
z1 = 3.16*2 + 1.08 = 7.40
a1 = 7.40
y_hat = 4.14*7.40 + 0.02 = 30.656
L = 1/2 * (30.656 - 30)^2
  ≈ 0.215
```

Loss fell from:

```text
2.0 → approximately 0.215
```

The gradient update overshot the target slightly, but substantially reduced the loss.

## The local-rule mental model

Every operation in the forward computation has a local derivative.

During the backward pass:

```text
incoming / upstream gradient
        ×
local derivative
        =
gradient passed farther backward
```

This is the chain rule implemented as an algorithm.

## Computational graph

A network can be represented as small operations:

```text
x ----       multiply/add --> z1 --> ReLU --> a1 ----w1 ---/                                         multiply/add --> y_hat --> loss
b1 --------------------------------------------/                ↑
w2 -------------------------------------------------------------/
b2 ------------------------------------------------------------/
```

The forward pass evaluates this graph from inputs to loss.

The backward pass traverses it in reverse, applying each operation's derivative rule.

## Branches: gradients add

If a variable influences the loss through multiple downstream paths, each path contributes to its total sensitivity.

Therefore the gradients from those paths add.

This follows directly from the derivative of a sum and becomes extremely important in residual networks, attention blocks, and any graph with branching connections.

## Backpropagation is not gradient descent

```text
backpropagation
→ computes gradients

optimizer / gradient descent
→ uses gradients to change parameters
```

They are separate stages.

## Modern engineering translation

Modern frameworks do not normally require engineers to manually derive every parameter gradient.

PyTorch autograd, JAX grad/VJP, and similar systems implement automatic differentiation.

The core mechanism remains reverse-mode differentiation through a computational graph.

This is especially efficient for neural-network training because a model has many parameters but the training objective ultimately reduces to a scalar loss.

## Why reverse mode fits ML

Conceptually, neural-network training often maps:

```text
millions/billions of parameters
        ↓
one scalar loss
```

Reverse-mode automatic differentiation starts from that scalar output and propagates sensitivity backward, producing gradients with respect to many earlier inputs/parameters efficiently.

## Forward connection

This lesson prepares for:

- formal chain-rule notation for multilayer networks;
- Jacobians and vector-Jacobian products;
- automatic differentiation;
- vanishing and exploding gradients;
- residual connections;
- gradient checkpointing;
- distributed backpropagation;
- why modern frameworks expose backward(), grad(), and VJP operations.

---

# Chain Rule — Sensitivity Through Composed Functions

The chain rule is the central calculus mechanism behind backpropagation.

## Core idea

Suppose u = g(x) and y = f(u). Then x changes u, and u changes y.

dy/dx = (dy/du) * (du/dx)

Interpretation:

sensitivity of y to x = sensitivity of y to u × sensitivity of u to x.

## Units intuition

If dy/du = 5 units of y per unit of u, and du/dx = 3 units of u per unit of x, then dy/dx = 15 units of y per unit of x.

(y/u) * (u/x) = y/x

The middle unit conceptually cancels, which is a useful memory device.

## Numerical example

Let u = 3x + 1 and y = u^2. At x = 2, u = 7.

du/dx = 3
dy/du = 2u = 14

Therefore:

dy/dx = (dy/du)(du/dx) = 14 * 3 = 42

Direct verification: y = (3x+1)^2, so dy/dx = 2(3x+1)*3. At x = 2 this is 42.

## Tiny-change interpretation

At x = 2, let dx = 0.001.

du ≈ (du/dx) dx = 3 * 0.001 = 0.003

dy ≈ (dy/du) du = 14 * 0.003 = 0.042

Combining gives dy ≈ (dy/du)(du/dx) dx, which is why dy/dx = (dy/du)(du/dx).

## Longer chain

For x → u → v → y:

dy/dx = (dy/dv)(dv/du)(du/dx)

The total sensitivity is the product of the local sensitivities along the dependency path.

## Neural-network chain

For w → z = wx+b → a = activation(z) → y_hat → L:

dL/dw = (dL/dy_hat)(dy_hat/da)(da/dz)(dz/dw)

Each factor is one local sensitivity.

Using the earlier numerical example:

dL/dy_hat = -2
dy_hat/da = 4
da/dz = 1
dz/dw = 2

Therefore dL/dw = (-2) * 4 * 1 * 2 = -16.

## Multiply along a path, add across paths

For one dependency path, sensitivities multiply.

If x affects y through multiple paths, each path contributes to the total derivative and the contributions add.

Example:

dy/dx = (dy/da)(da/dx) + (dy/db)(db/dx)

Durable rule: multiply along a path; add across paths.

## Multivariable chain rule

If z = f(x,y), x = g(t), and y = h(t), then:

dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)

This is the same dependency-graph rule: multiply sensitivities along each route and add contributions from all routes.

## Connection to backpropagation

Forward pass computes values. Backward pass computes sensitivities.

At each node:

upstream gradient × local derivative = gradient sent backward.

If several backward paths meet at one variable, their gradient contributions add.

Backpropagation is therefore an efficient algorithm for repeatedly applying the chain rule over a computational graph.

## Durable mental model

A changes B; B changes C; C changes D.

How sensitive is D to A?

= how D reacts to C × how C reacts to B × how B reacts to A.

For branching graphs: multiply along each path, add across paths.


---

# Scalar Does Not Mean Constant

A scalar is a quantity with one numerical component. It can be either constant or variable.

Examples:

```text
3          = scalar constant
b          = scalar variable / parameter
w          = scalar variable / parameter
x          = scalar variable
```

The rule:

```text
derivative of a constant = 0
```

only applies when the quantity is constant **with respect to the variable being differentiated**.

For:

```text
y_hat = w2*a1 + b2
```

when differentiating with respect to b2, treat w2 and a1 as locally fixed:

```text
∂y_hat/∂b2
=
∂(w2*a1)/∂b2
+
∂b2/∂b2
=
0 + 1
=
1
```

Why is ∂b2/∂b2 = 1?

Because changing b2 by a tiny amount Δb2 changes y_hat by exactly the same amount:

```text
Δy_hat = Δb2
```

so:

```text
Δy_hat / Δb2 = 1
```

The important distinction is:

```text
d(5)/db = 0       because 5 does not depend on b

d(b)/db = 1       because b changes one-for-one with itself

d(3b)/db = 3      because changing b by 1 changes 3b by 3
```

In partial derivatives, other independent variables are held fixed locally, but the variable named in the denominator is allowed to change.


---

# Learning Program After the 3Blue1Brown Foundation

The 3Blue1Brown neural-network material is used as the intuition-first foundation. We do not need to finish every remaining video before moving forward if the core mental models are already secure.

The learning program now combines three complementary tracks:

```text
CS229
→ statistical / classical ML foundations
→ actual models and their mathematics

CS231n
→ deep-learning mechanics
→ neural-network optimization and modern architectures

our Python notebooks
→ manual calculation
→ NumPy implementation
→ sklearn / PyTorch implementation
→ experiments and diagnostics
```

## Core rule

For every major model or algorithm:

```text
intuition
→ derive the mathematics
→ calculate a tiny example manually
→ connect to Linear Algebra / Calculus / Probability
→ implement from simple primitives
→ implement with NumPy
→ implement with sklearn / PyTorch
→ inspect intermediate values
→ test failure cases
→ connect to Feature Engineering
→ connect to modern AI
→ retain in durable notes
```

## Classical ML spine from CS229

Priority sequence:

1. Linear regression
2. Logistic regression
3. Probabilistic interpretation / maximum likelihood
4. Generalized linear models
5. Generative vs discriminative models
6. Gaussian discriminant analysis
7. Naive Bayes
8. Regularization
9. Bias / variance
10. Model selection and cross-validation
11. Support vector machines and kernels
12. Trees, random forests, boosting, gradient boosting
13. Clustering and k-means
14. Gaussian mixtures and EM
15. PCA and dimensionality reduction
16. Probability / Bayesian foundations where needed
17. Learning theory selectively, with emphasis on practical generalization intuition

The historical Stanford SEE lectures remain valuable for mathematical derivations, but examples and implementation details should be modernized rather than copied uncritically.

## Deep-learning spine from CS231n

Continue the existing stages:

1. Neural-network foundations
2. Loss and optimization
3. Backpropagation
4. Training correctly
5. CNNs
6. Sequence models
7. Transformers
8. Attention deeply
9. Transformer MLPs
10. Modern scaling
11. Representation learning
12. Generative AI
13. Frontier translation

## Python practice rule

Do not begin with one-line framework APIs.

For important algorithms, use a progression such as:

```text
manual arithmetic
→ small Python functions
→ NumPy vectorization
→ sklearn / PyTorch
→ diagnostics / visualization
→ realistic dataset
```

The objective is to understand what the library is doing, then use the library efficiently.

## Why both CS229 and CS231n

CS229 answers:

> What is the model mathematically, what assumptions does it make, what objective does it optimize, and when does it generalize?

CS231n answers:

> How do neural networks compute, differentiate, optimize, and scale into modern deep-learning systems?

Together they prevent the learning program from becoming either:

- classical ML without modern AI; or
- modern deep learning without statistical and mathematical foundations.
