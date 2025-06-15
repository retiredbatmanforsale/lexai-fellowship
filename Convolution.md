# Understanding Convolution Through Dice Probability

## Introduction
Understanding convolution through the two-dice probability example is a beautiful way to build intuition, especially for convolution in Convolutional Neural Networks (CNNs). Let's walk through this step-by-step.

## Problem Setup: Sum of Two Dice 🎲

You're rolling two fair 6-sided dice (Die A and Die B). Each die has numbers from 1 to 6.

You want to calculate the probability distribution of their sum:
- Minimum sum = 1 + 1 = 2
- Maximum sum = 6 + 6 = 12

You want to know the probability of each sum from 2 to 12.

## Step 1: Represent Each Die as a Probability Vector 🔢

Since each die is fair, each die has a discrete uniform distribution over values 1 to 6.

So we can represent the probability of each outcome as a vector:

Die A (and Die B) probabilities:
```
Index:      0     1     2     3     4     5
Face:       1     2     3     4     5     6
P(value):  1/6   1/6   1/6   1/6   1/6   1/6
```

Let's define this as a vector a:
```
a = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
```

Same for the second die b.

## Step 2: What is Convolution? 🔁

The convolution operation combines these two vectors to produce a new vector c such that:

```
c[n] = ∑ a[k] * b[n - k]
```

This is summing over all the ways to combine values from Die A and Die B to get a total sum.

But since this is discrete and finite, we do discrete convolution of two sequences a and b:

```
c = a * b  ← (discrete convolution)
```

This c gives us probabilities for each possible sum, shifted appropriately.

## Intuition: What Does Convolution Do? 🧠

When you convolve two probability distributions (like the two dice), the result is a distribution of their sum. That is exactly what we want.

## Step 3: Perform Discrete Convolution 🔢

We convolve:
```
a = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
b = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
```

The result c will have length 6 + 6 - 1 = 11 elements. Each index i in the result corresponds to the sum = i + 2.

Let's write it as sum probabilities:

| Sum | Ways | Probability |
|-----|------|-------------|
| 2   | (1,1) | 1/36 |
| 3   | (1,2), (2,1) | 2/36 |
| 4   | (1,3), (2,2), (3,1) | 3/36 |
| 5   | (1,4), (2,3), (3,2), (4,1) | 4/36 |
| 6   | … | 5/36 |
| 7   | … | 6/36 |
| 8   | … | 5/36 |
| 9   | … | 4/36 |
| 10  | … | 3/36 |
| 11  | … | 2/36 |
| 12  | (6,6) | 1/36 |

So:
```
c = [1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36]
```

## Step 4: Match It to CNN Convolution 🔁

In CNNs, you are convolving a filter (kernel) with the input (like an image patch).

Dice example:
- a = input distribution (Die A)
- b = filter (Die B)
- a * b = convolved output (sum distribution)

This is exactly what CNN does, except:
- In images: the input is a 2D grid of pixel values
- The kernel is usually a small 2D filter (e.g., 3x3)
- The convolution aggregates local neighborhoods (just like dice values aggregate sum probabilities)

So convolution in CNNs generalizes this same "overlapping and summing" idea across image regions.

## Summary ✨

| Dice Analogy | CNN Convolution |
|--------------|-----------------|
| Die A values | Input values (like pixels) |
| Die B values | Kernel/filter values |
| Sum of two dice | Output value from convolution |
| Probability of each sum | Output activation after applying filter |
| Convolution of two vectors | Convolution of input and kernel |

---

## Advanced Example: Non-Uniform Probabilities 🎲

Let's go deeper into convolution by removing the uniformity assumption. This example will further cement your understanding of how convolution works as a weighted sum over overlapping elements.

### Example: Two Dice with Non-Uniform Probabilities

Let's define the probabilities for Die A and Die B as follows:

#### Die A Probabilities (biased die) 🔸

| Face | 1 | 2 | 3 | 4 | 5 | 6 |
|------|---|---|---|---|---|---|
| P    | 0.10 | 0.15 | 0.20 | 0.20 | 0.15 | 0.20 |

So the vector a = [0.10, 0.15, 0.20, 0.20, 0.15, 0.20]

#### Die B Probabilities (another biased die) 🔹

| Face | 1 | 2 | 3 | 4 | 5 | 6 |
|------|---|---|---|---|---|---|
| P    | 0.05 | 0.10 | 0.30 | 0.25 | 0.20 | 0.10 |

So the vector b = [0.05, 0.10, 0.30, 0.25, 0.20, 0.10]

### Compute the Convolution: c = a * b 🔁

The convolution c[n] = ∑ a[k] * b[n-k] computes the probability of each sum from 2 to 12, just like before.

The resulting vector c will have size:
```
length(c) = len(a) + len(b) - 1 = 6 + 6 - 1 = 11
```

We'll denote each index in c as corresponding to the sum s = i + 2

### Convolution Calculations 🧮

Let's walk through a few to understand.

For sum = 2 (index 0 in c):
- Only one combination: (1,1)
- c[0] = a[0] * b[0] = 0.10 * 0.05 = 0.005

For sum = 3 (index 1 in c):
- Two combinations: (1,2), (2,1)
- c[1] = a[0] * b[1] + a[1] * b[0] 
-      = 0.10 * 0.10 + 0.15 * 0.05 
-      = 0.01 + 0.0075 = 0.0175

For sum = 4 (index 2):
- (1,3), (2,2), (3,1)
- c[2] = a[0]*b[2] + a[1]*b[1] + a[2]*b[0]
-      = 0.10*0.30 + 0.15*0.10 + 0.20*0.05
-      = 0.03 + 0.015 + 0.01 = 0.055

### Final Distribution: (Partial) 🔚

So far we have:

| Sum | c[i] (Probability) |
|-----|-------------------|
| 2   | 0.005 |
| 3   | 0.0175 |
| 4   | 0.055 |
| …   | … |
| 12  | (last value) |

You'd continue this process until you compute all 11 values (from sum = 2 to sum = 12). The total of all values in c will still sum up to 1 (as it's a valid probability distribution).

### Why is this Important for CNNs? 🤖

This example reinforces a key idea:

Convolution is a way to combine two sets of values — weighting one against the other across all alignments.

In CNNs:
- Input = image patches (e.g., intensity values, like a)
- Kernel = filter weights (like b)
- Output = feature map (like c), which highlights how strongly that patch matches the filter

And just like our dice example, each value in the output is a weighted combination of input values, where the weights are defined by the kernel. 