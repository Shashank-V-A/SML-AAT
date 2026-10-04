# PROJECT REPORT
## Statistical Machine Learning — AAT 2

---

# 1. Title

**Permutation Test Simulation: Comparing Placebo vs Drug Effect on Blood Pressure Reduction**

---

# 2. Problem Statement

In medical research, a common question is:

> Does a new drug work better than a placebo, or is the difference just due to chance?

In this project, we study a **simulated clinical trial** with two groups of patients:

- **Group A — Placebo:** patients receive a fake pill (no active medicine)
- **Group B — Drug:** patients receive a new blood-pressure medicine

For each patient, we measure:

**Systolic blood pressure reduction (in mmHg)**  
Higher reduction = better result (blood pressure went down more).

### Why use a permutation test?

A normal two-sample t-test needs assumptions (like approximate normality).  
A **permutation test** is a simple statistical method that answers:

> If the drug and placebo were actually the same, how often would we still see a difference as large as the one we observed, just by random assignment of labels?

It does this by **shuffling group labels many times** and rebuilding the distribution of differences under the null hypothesis. This makes it a practical Statistical Machine Learning / statistical inference project using simulated data.

---

# 3. Objectives

1. Generate two groups of simulated patient data (Placebo vs Drug).
2. Calculate the **observed mean difference** between the two groups.
3. Perform many random **permutations** of treatment labels under the null hypothesis.
4. Plot the **permutation (null) distribution**.
5. Estimate the **two-sided p-value** and decide whether the drug effect is statistically significant.
6. Interpret the results clearly using tables and graphs.

---

# 4. Dataset / Data Generation

We did **not** use a real hospital dataset. As allowed by the AAT, we used a **simulated dataset**.

### How data was generated

| Group | Treatment | Sample size (n) | How scores were simulated |
|-------|-----------|-----------------|---------------------------|
| A | Placebo | 30 patients | Normal distribution with mean = 4.0 mmHg, SD = 5.0 |
| B | Drug | 30 patients | Normal distribution with mean = 9.5 mmHg, SD = 5.0 |

- Total patients = **60**
- Values were clipped to a realistic range **[-5, 25] mmHg**
- Random seed = **42** (so results are reproducible every run)
- File saved: `outputs/bp_reduction.csv`

### Important point (for viva)

The true means (4.0 and 9.5) are used **only to create fake data**.  
The permutation test itself **does not know** these true means.  
It only looks at the generated sample values and their labels.

### What each row in the dataset means

Example columns in `bp_reduction.csv`:

| Column | Meaning |
|--------|---------|
| `bp_reduction_mmHg` | How much systolic BP reduced for that patient |
| `group` | A or B |
| `arm` | Placebo or Drug |

---

# 5. Methodology / Algorithm

## 5.1 Hypotheses

- **Null hypothesis (H₀):**  
  There is **no difference** in mean BP reduction between Placebo and Drug.  
  \[
  \mu_A = \mu_B
  \]

- **Alternative hypothesis (H₁):**  
  There **is a difference** (drug can be better or worse).  
  \[
  \mu_A \neq \mu_B
  \]

We use a **two-sided** test because we are checking for any difference, not only “drug is better.”

## 5.2 Test statistic

We use the difference of sample means:

\[
\Delta = \bar{x}_B - \bar{x}_A
\]

Where:

- \(\bar{x}_A\) = average BP reduction in Placebo group  
- \(\bar{x}_B\) = average BP reduction in Drug group  
- \(\Delta\) = observed difference (Drug − Placebo)

If \(\Delta\) is large positive, Drug reduced BP more than Placebo.

## 5.3 Permutation test steps (simple explanation)

Think of it like this:

1. First calculate the **real** difference using actual Placebo/Drug labels.  
   This is called the **observed difference**.

2. Now pretend the labels do not matter.  
   Put all 60 BP values into one big pool.

3. Randomly give 30 values the label “Placebo” and 30 the label “Drug”  
   (this is one **permutation** / shuffle).

4. Calculate the mean difference again for this fake labeling.

5. Repeat this shuffle **10,000 times**.  
   This creates a big list of “differences that happen by chance.”

6. Compare the real difference with this chance distribution.

7. Count how often a shuffled difference is **as extreme or more extreme** than the real one (in either direction, because it is two-sided).

8. Convert that count into a **p-value**.

### p-value formula used

\[
p = \frac{1 + \text{number of permutations where } |\Delta_{\text{perm}}| \ge |\Delta_{\text{obs}}|}{N + 1}
\]

Where \(N = 10{,}000\).

The “+1” makes the p-value slightly conservative (standard practice).

### Decision rule

- Significance level: \(\alpha = 0.05\)
- If \(p < 0.05\) → **Reject H₀** (difference looks real, not just luck)
- If \(p \ge 0.05\) → **Fail to reject H₀** (not enough evidence)

### Why this method works

Under H₀, treatment labels are meaningless / interchangeable.  
So random re-labeling shows what differences look like by chance.  
If the real difference is rare in that chance world, we reject H₀.

---

# 6. Implementation

### Tools used
- Language: **Python**
- Libraries:
  - `numpy` → random data + calculations
  - `pandas` → tables / CSV
  - `matplotlib` → graphs

### Main file
- `permutation_test.py`

### How to run
```bash
pip install -r requirements.txt
python permutation_test.py
```

### What the program does
1. Generates Placebo and Drug data
2. Prints descriptive statistics (count, mean, std, min, max)
3. Computes observed mean difference
4. Runs 10,000 permutations
5. Computes two-sided p-value
6. Saves results and graphs into `outputs/`

### Output files

| File | Purpose |
|------|---------|
| `outputs/bp_reduction.csv` | Simulated patient-level data |
| `outputs/results.csv` | Final numerical summary |
| `outputs/group_comparison.png` | Boxplot comparing Placebo vs Drug |
| `outputs/permutation_distribution.png` | Null distribution + observed difference |

---

# 7. Results

## 7.1 Descriptive statistics

| Treatment | n | Mean BP reduction (mmHg) | Std. Dev. | Min | Max |
|-----------|---|--------------------------|-----------|-----|-----|
| Placebo (A) | 30 | **4.109** | 3.820 | -5.000 | 10.113 |
| Drug (B) | 30 | **10.069** | 4.024 | 2.214 | 20.208 |

### Simple meaning
- Placebo patients improved by about **4.1 mmHg** on average.
- Drug patients improved by about **10.1 mmHg** on average.
- So visually and numerically, Drug looks better.

## 7.2 Observed difference

\[
\Delta_{\text{obs}} = 10.069 - 4.109 = \mathbf{5.960\ \text{mmHg}}
\]

Meaning: on average, the Drug group reduced blood pressure by about **6 mmHg more** than Placebo.

## 7.3 Permutation test output (from terminal)

| Quantity | Value | Meaning in simple words |
|----------|-------|-------------------------|
| Number of permutations | 10,000 | How many random shuffles we did |
| Observed delta (B − A) | 5.9595 mmHg | Real difference from actual labels |
| Null mean | -0.0102 ≈ 0 | Average chance difference (should be near 0) |
| Null SD | 1.2745 | Typical size of chance differences |
| Two-sided p-value | **0.0001** | Chance of getting difference this extreme by luck |
| Alpha (α) | 0.05 | Cutoff for significance |
| Decision | **Reject H₀** | Difference is statistically significant |

## 7.4 Graph 1 — Group comparison (`group_comparison.png`)

This graph shows the raw data:

- Boxplot of Placebo vs Drug
- Individual patient points overlaid

**What to write in report:**  
The Drug group’s BP reductions are clearly higher than Placebo. The median and overall distribution of Drug sit above Placebo, suggesting a treatment effect before formal testing.

## 7.5 Graph 2 — Permutation distribution (`permutation_distribution.png`)

This graph shows:

- Blue histogram = differences obtained after random label shuffling (null world)
- Red dashed line = our real observed difference (≈ 5.96)
- Dotted red line on left = opposite extreme for two-sided test

**What to write in report:**  
The null distribution is centered near 0. The observed difference lies far in the right tail. Very few (almost none) of the 10,000 random permutations reached a difference as large as 5.96 mmHg. This produces a very small p-value (0.0001).

## 7.6 Interpretation

- p-value = **0.0001**
- α = **0.05**
- Since \(0.0001 < 0.05\), we **reject the null hypothesis**.

### Final statistical conclusion
In this simulated trial, the difference in mean systolic BP reduction between Drug and Placebo is **statistically significant**.  
It is extremely unlikely that a difference of about 6 mmHg would appear only by random labeling.

### Careful wording (good for viva/report)
This does **not** prove the drug works in real hospitals forever.  
It shows that **in this simulated dataset**, the observed effect is unlikely under H₀.

---

# 8. Conclusion

This project successfully implemented a complete **Permutation Test Simulation** as required by SML AAT 2:

1. Generated two treatment groups (Placebo vs Drug)
2. Calculated observed mean difference (~6 mmHg)
3. Built a null distribution using 10,000 label permutations
4. Plotted the permutation distribution
5. Estimated p-value ≈ 0.0001 and rejected H₀

### Key learning
A permutation test gives an intuitive meaning of p-value:

> “If treatment labels did not matter, how often would we still see a difference this large?”

In our case, almost never. So we conclude the Drug arm shows a significantly larger BP reduction than Placebo in this simulation.

---

# Appendix A — How to explain this to someone writing the report

### One-paragraph summary (can paste/adapt)
> This project applies a permutation test to a simulated drug trial comparing placebo and an antihypertensive drug. Sixty patient outcomes (systolic BP reduction in mmHg) were generated, thirty per arm. The observed mean difference (Drug − Placebo) was approximately 5.96 mmHg. Treatment labels were randomly shuffled 10,000 times to construct the null distribution of mean differences. The two-sided p-value was 0.0001, which is less than 0.05, so the null hypothesis of no difference was rejected. Graphs of group distributions and the permutation null distribution support this conclusion.

### Report checklist (AAT format)
- [x] Title  
- [x] Problem Statement  
- [x] Objectives  
- [x] Dataset / Data Generation  
- [x] Methodology / Algorithm  
- [x] Implementation  
- [x] Results (tables + graphs + values)  
- [x] Conclusion  

### Files to attach / show during demo
1. `permutation_test.py` (code)
2. Terminal output
3. `group_comparison.png`
4. `permutation_distribution.png`
5. `results.csv`

---

# Appendix B — Viva preparation (short Q&A)

| Likely question | Good short answer |
|-----------------|-------------------|
| What is your project? | Permutation test on placebo vs drug BP reduction. |
| What is H₀? | No mean difference between placebo and drug. |
| What is observed delta? | Mean(Drug) − Mean(Placebo) ≈ 5.96 mmHg. |
| Why shuffle labels? | Under H₀ labels are exchangeable; shuffling builds chance distribution. |
| What is p-value here? | Proportion of shuffled \|Δ\| ≥ observed \|Δ\|. |
| Why two-sided? | Drug could be better or worse; we test for any difference. |
| Why reject H₀? | p = 0.0001 < 0.05. |
| Why not only t-test? | Permutation test needs fewer distributional assumptions and is easy to explain. |
| What does null mean ≈ 0 show? | Sanity check that random shuffles produce no average effect. |
| Real data or simulated? | Simulated dataset, allowed by AAT. |

---

# Appendix C — Exact numbers for tables (copy-ready)

| Metric | Value |
|--------|-------|
| n_Placebo | 30 |
| n_Drug | 30 |
| Mean Placebo | 4.109 mmHg |
| Mean Drug | 10.069 mmHg |
| Observed Δ | 5.960 mmHg |
| Permutations | 10,000 |
| Null mean | -0.010 |
| Null SD | 1.275 |
| p-value (two-sided) | 0.0001 |
| α | 0.05 |
| Decision | Reject H₀ |
