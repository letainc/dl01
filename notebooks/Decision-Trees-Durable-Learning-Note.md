# Decision Trees — Durable Learning Note

> **Status:** In progress  
> **Track:** CS229-aligned supervised-learning study, supplemented with tree/ensemble material not explicitly covered in the uploaded CS229 notes  
> **Purpose:** A long-form relearning document. This note grows as the topic is learned. It is not a cheat sheet.

## Learning contract

We will build this topic in the same order used for the Linear Regression study:

```text
intuition
→ visual structure
→ small manual example
→ mathematics
→ symbol-by-symbol meaning
→ algorithm behavior
→ Python from first principles
→ scikit-learn
→ diagnostics
→ failure cases
→ production considerations
→ bridge to boosting and XGBoost
```

The goal is not to memorize an API. The goal is to understand what a tree is doing well enough to predict its behavior before running code.

---

# 1. Where Decision Trees Fit

A supervised-learning dataset contains examples of the form

```text
features x  →  target y
```

For the bank-churn problem, a row could represent one customer:

| Feature | Meaning |
|---|---|
| Age | customer age |
| Balance | account balance |
| IsActiveMember | activity indicator |
| Geography | customer geography |
| ... | other customer attributes |
| Exited | target: 1 if the customer left, 0 otherwise |

Because `Exited` can take only two values,

```text
0 = stayed
1 = exited
```

this is **binary classification**.

The uploaded CS229 notes establish the general supervised-learning framing: learn a hypothesis/function from training examples so that it predicts the corresponding target on new inputs. They also distinguish regression from classification by whether the target is continuous or takes one of a small number of discrete values.

Decision trees solve the same supervised-learning problem, but they represent the prediction function differently from a linear or logistic model.

---

# 2. Linear Rule vs Tree Rule

A linear-style model tries to build a prediction from a weighted combination of features.

Conceptually:

```text
score
=
weight_1 × age
+ weight_2 × balance
+ weight_3 × activity
+ ...
```

A decision tree instead learns a sequence of **questions**.

Example:

```text
                 IsActiveMember <= 0.5?
                    /           \
                  yes            no
                  /               \
          Balance > 100000?      predict stay
             /       \
           yes        no
            |          |
       predict exit   Age > 55?
                      /      \
                    yes       no
                    |          |
              predict exit   predict stay
```

This is the first core mental model:

> **A decision tree partitions the feature space by repeatedly asking feature-threshold questions.**

Each question divides the currently available rows into smaller groups.

---

# 3. Tree Vocabulary

## Root node

The first question in the tree.

```text
IsActiveMember <= 0.5?
```

Everything starts here.

## Internal node

Any later decision/question.

```text
Balance > 100000?
```

## Split

The operation that sends observations into different child groups according to a condition.

For a numerical feature:

```text
Age <= 42
```

might produce:

```text
left child  = customers with Age <= 42
right child = customers with Age > 42
```

## Leaf

A terminal region where the tree stops splitting and makes a prediction.

For classification, a leaf can store the class distribution of the training examples that reached it.

Example:

```text
leaf contains 10 customers

8 stayed
2 exited
```

The estimated class proportions inside the leaf are:

```text
P(stay) ≈ 8 / 10 = 0.80
P(exit) ≈ 2 / 10 = 0.20
```

This will later connect directly to `predict_proba()`.

## Depth

How many decision levels the tree has below the root.

Greater depth means the model can make increasingly specific partitions.

That gives more expressive power, but also increases the danger of fitting noise instead of reusable structure.

---

# 4. First Manual Example

Consider eight bank customers:

| Customer | Active | Balance ($k) | Exited |
|---|---:|---:|---:|
| A | 1 | 20 | 0 |
| B | 1 | 35 | 0 |
| C | 1 | 80 | 0 |
| D | 1 | 120 | 1 |
| E | 0 | 15 | 0 |
| F | 0 | 60 | 1 |
| G | 0 | 110 | 1 |
| H | 0 | 160 | 1 |

The target vector is therefore:

```text
y = [0, 0, 0, 1, 0, 1, 1, 1]
```

At the root, all eight examples are mixed:

```text
4 stayed
4 exited
```

Now consider the candidate question:

```text
Active <= 0.5?
```

Because `Active` is encoded as 0 or 1, this separates inactive and active customers.

The two resulting groups are:

```text
inactive:
E, F, G, H
targets = [0, 1, 1, 1]

active:
A, B, C, D
targets = [0, 0, 0, 1]
```

The split has created two groups that are noticeably less mixed than the original 4/4 group:

```text
inactive → mostly Exited
active   → mostly Stayed
```

This gives the central idea behind tree training:

> A useful split makes the child groups more homogeneous with respect to the target.

The word often used is **purity**.

A node containing only one class is perfectly pure.

Examples:

```text
[0, 0, 0, 0] → pure
[1, 1, 1]    → pure
[0, 1, 0, 1] → mixed / impure
```

The tree therefore needs a numerical way to answer:

> “How impure is this node?”

and then:

> “How much impurity would this candidate split remove?”

That leads to **Gini impurity**, **entropy**, and **information gain**.

We will derive those rather than treating them as formulas to memorize.

---

# 5. The Question the Training Algorithm Must Solve

Imagine many possible root questions:

```text
Age <= 35?
Age <= 42?
Age <= 55?
Balance <= 50?
Balance <= 90?
Balance <= 130?
Active <= 0.5?
```

The algorithm cannot choose a split merely because it looks reasonable to a human.

It needs an objective numerical criterion.

For every candidate split, conceptually it does:

```text
1. split the current rows
2. inspect the class mixture in each child
3. measure child impurity
4. combine the child impurity values
5. compare this split against every other candidate
6. choose the best improvement
```

The next learning step is to build that measurement ourselves.

---

# 6. Connection to Bias and Variance

A very shallow decision tree has limited flexibility.

It may miss genuine structure:

```text
underfitting
→ high bias
```

A very deep tree can keep splitting until tiny groups, or even individual observations, are isolated.

It can make training error extremely small while learning accidental details:

```text
overfitting
→ high variance
```

This is why parameters such as

```text
max_depth
min_samples_split
min_samples_leaf
```

exist.

We will not memorize those knobs. We will first understand the failure mode that creates the need for each one.

This becomes essential later because boosted-tree systems such as XGBoost intentionally combine many small trees, and model capacity must be controlled deliberately.

---

# 7. Bridge to Boosting and XGBoost

The eventual progression is:

```text
single decision tree
        ↓
understand tree mistakes
        ↓
ensembles
        ↓
bagging vs boosting
        ↓
gradient boosting
        ↓
gradient-boosted decision trees
        ↓
XGBoost
```

Do not jump directly to `XGBClassifier`.

A useful mental checkpoint will be:

> If I cannot explain how one tree chooses one split, then an ensemble of hundreds of trees is still a black box.

---

# 8. Learning Checkpoint — Before Gini

Use the eight-customer example above.

Answer without calculating a formal impurity formula yet:

1. Is the root node with four `0`s and four `1`s highly pure or highly impure?
2. After splitting on `Active`, which child is mostly class 1?
3. Which child is mostly class 0?
4. Why is this split intuitively useful?
5. Would a child containing `[1, 1, 1, 1]` be better or worse than `[0, 1, 1, 1]` if our goal is purity?
6. What do you think a sensible impurity score should equal for a perfectly pure node: a large number or zero?

Do not proceed to the formula until these answers are intuitive.

---

# 9. Next Section to Add

Next:

**Gini impurity from first principles**

We will derive:

```text
class proportion
→ probability of being wrong under random labeling
→ Gini impurity
→ weighted child impurity
→ impurity reduction
→ choosing the best split
```

Then we will calculate an entire candidate split by hand before using Python.
