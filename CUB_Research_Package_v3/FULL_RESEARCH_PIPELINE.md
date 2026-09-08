# Research pipeline: visible evidence and relational bird parts

Prepared for the supplied executed `dino-v2-v2.ipynb`. The companion `CUB_Full_Research_Suite_v3_Kaggle.ipynb` is the executable next batch. New experimental performance is unknown until it runs on Kaggle.

## 1. What the existing results actually say

These are observed v2 results, one seed, not v3 predictions. Accuracy and PCK below are percentages.

| v2 model | Raw accuracy | Raw PCK | Visibility AUROC | Visibility Brier ↓ |
|---|---:|---:|---:|---:|
| Independent parts | 79.63 | 86.82 | 0.8926 | 0.1013 |
| Joint visibility gate | 79.32 | 86.63 | 0.8677 | 0.1658 |
| Detached gate, no graph | 78.22 | 86.08 | 0.8579 | 0.1167 |
| Detached gate, graph | 78.39 | 86.95 | 0.8597 | 0.1163 |

The graph improved localization over the matched detached model by **0.87 percentage point**, but classification by only **0.17 point**. Detachment improved Brier relative to the joint gate, while the independent model's auxiliary visibility head still performed best. We have not demonstrated a better overall classifier.

At the validation-calibrated 90% visible-recall operating point, detached graph accuracy was **75.82%** and invisible-part emission remained **69.12%**. Filtering removed evidence, including evidence the classifier relied on. The representation-training curves also showed early best classification checkpoints while localization continued improving. This motivates staged training, rather than simply adding more losses or epochs.

The old targeted-occlusion test covered about four annotated landmarks per square on average. It does not isolate the causal importance of a single part. The new suite measures how many landmarks each square covers.

## 2. The objective we can defend

**Learn an interpretable bird classifier that localizes semantic parts, uses their relationships, and reduces reliance on parts that are not visible, while preserving classification performance.**

Images and 2D part annotations are enough to test this. A head can appear in different locations and poses; learned spatial mixtures allow several plausible relations instead of one rigid template. A part's anatomical existence is different from its visibility: an occluded wing exists, but should not be presented as observed evidence.

The current approach uses CUB's 15 named part locations and visibility labels during training. Call it **part-supervised**, not unsupervised part discovery. Ground-truth parts are not supplied to model inference. Geometry priors are fitted from training images; the graph topology is manually specified anatomical knowledge.

Spatially flexible prototypes already exist: [Deformable ProtoPNet](https://arxiv.org/abs/2111.15000) explicitly models changing relative part positions. Therefore, “we added a graph” or “parts can change pose” alone is not an established novelty claim. The potential contribution is a carefully demonstrated interaction between relational localization, visibility, and faithful class evidence. A literature comparison must still establish what is new.

## 3. Run this complete batch now

The notebook defaults to **200 classes, three seeds (42, 123, 2026), 11 variants**. It trains frozen-feature models, not DINOv2 itself. Same-geometry variants share exactly the same representation checkpoint; different geometries get their own matched representation training.

| Variant | What changes | Main question |
|---|---|---|
| `mean_linear` | Linear classifier on masked mean DINO patch features; only class supervision | Is the interpretable approach sacrificing too much ordinary classification accuracy? |
| `independent` | Ungated part evidence, supervised detector and visibility head | What is the matched part-supervised reference? |
| `posthoc_soft` | Gate the final independent model without refitting | How much damage comes from changing evidence at inference? |
| `staged_soft` | Freeze representation; refit class weights and temperature with gates | Does adapting the classifier recover that damage? |
| `staged_normalized` | Normalize the available class-weighted evidence | Is reduced evidence scale a major cause of failure? |
| `graph_ungated` | Anatomical Gaussian-mixture graph, no classification gate | Does the graph help without the gating complication? |
| `graph_normalized` | Graph plus staged normalized scoring | Does the combined method improve the actual trade-off? |
| `single_graph_normalized` | One Gaussian per relation | Are multiple relation modes necessary? |
| `permuted_graph_normalized` | Assign paired relation priors to wrong edges | Does correct anatomical correspondence matter? |
| `distance_graph_normalized` | Generic isotropic distance kernel | Is generic spatial smoothing enough? |
| `entropy_graph_normalized` | Discount diffuse neighbor heatmaps in graph messages | Does map confidence improve relation propagation? |

The permuted control preserves graph connectivity and reverse-edge pairing. It tests **edge-prior specificity**, not random topology. The distance kernel is a fixed control, not an equally tuned competing model. Mean-linear is neither a CLS-token baseline nor an exact replication of the supplied paper.

There are 33 model/seed configurations. Shared representation training avoids doing the same work repeatedly. Default representation training is 12 epochs, classifier refitting 15, linear training 30; selection accuracy chooses the best checkpoint. These budgets are experimental settings, not a claim of equal training FLOPs across architectures.

## 4. Mathematics implemented in v3

Let normalized DINO patch features be f_t; q_p(t) is the probability that patch t contains part p. A train-fitted class/part prototype bank provides evidence e_cp. The visibility head predicts v_p in [0,1]. Class-specific nonnegative part weights w_cp sum to one over supported parts.

### Relational refinement

For directed edge p→j, training part displacements fit a mixture:

\[
K_{pj}(t,u)\propto\sum_k\pi_{pjk}\,\mathcal N(x_u-x_t;\mu_{pjk},\Sigma_{pjk}).
\]

Messages combine neighbor probability with this compatibility:

\[
m_p(t)=\frac{1}{|N(p)|}\sum_{j\in N(p)}v_j\log\left(\epsilon+\sum_uK_{pj}(t,u)q_j(u)\right),
\qquad q_p\leftarrow\operatorname{softmax}(a_p+\alpha m_p).
\]

The code centers messages over valid patches and bounds relation strength. Padding is excluded. Coordinates are in the letterboxed image frame; this is not a rotation-invariant or 3D model. In the entropy variant, neighbor confidence becomes v_j(1−H(q_j)/log T_valid). This is a heuristic: a confidently incorrect heatmap can still mislead the graph.

### Separate representation learning from classifier adaptation

Phase 1 uses ungated classification with localization and visibility supervision:

\[
\mathcal L=\mathcal L_{CE}+\lambda_{loc}\mathcal L_{heatmap}+\lambda_{vis}\mathcal L_{BCE}.
\]

The optional edge-loss coefficient remains at its configured default zero. Location loss uses visible annotated parts only. Phase 2 freezes the detector, visibility predictor, prototypes, and graph; only class part-weights and temperature are optimized using classification loss. This tests whether classifier adaptation can use a stable visibility signal.

### Evidence scoring

Soft scoring:

\[
z_c=\tau\sum_p w_{cp}v_pe_{cp}.
\]

Normalized scoring:

\[
z_c=\tau\frac{\sum_pw_{cp}v_pe_{cp}}{\epsilon+\sum_pw_{cp}v_p}.
\]

Implementation clamps the denominator rather than adding epsilon everywhere. Normalization removes sensitivity to a common positive rescaling of gates away from the clamp. It is **not guaranteed to improve accuracy**: the denominator differs by class and can amplify sparse, incorrect evidence. That is why both normalized and unnormalized controls are included.

For a calibrated threshold θ_p, hard filtering replaces v_p with v_p·1[v_p≥θ_p]. Every contribution uses the same denominator as its class score, so contributions sum exactly to the logit. If all parts are rejected, the system records abstention and prediction coverage; it must not describe the arbitrary argmax of all-zero logits as a meaningful prediction.

A calibration-only diagnostic substitutes annotation visibility into scoring to investigate detector error. This is an oracle diagnostic, not a deployable model or a guaranteed upper bound.

## 5. Evaluation protocol and figures

The official CUB training partition provides 5,094 fitting images, 450 checkpoint-selection images, and 450 threshold-calibration images. The official 5,794 test images are reported as a **reused development test** because earlier test results informed this design. Three seeds quantify training variation; they do not restore an untouched test set.

Training fits feature banks and geometry. Selection chooses checkpoints. Calibration alone fits thresholds at target visible recalls 80%, 90%, 95%, and 98%. The predeclared primary threshold is 90%. Test results must not choose a threshold. Validation targets do not guarantee the same recall on test data.

The notebook generates:

1. Accuracy and localization summaries with seed mean and standard deviation.
2. Paired image-bootstrap intervals for planned model comparisons, separately per seed. These are exploratory intervals, not multiplicity-adjusted confirmation.
3. Accuracy–missing-emission and visible-recall–missing-emission curves.
4. Visibility AUROC, Brier, reliability diagrams, and per-part diagnostics.
5. Per-part PCK heatmaps. PCK uses the configured 0.10 bounding-box-diagonal tolerance on visible ground-truth parts; record that convention when comparing papers.
6. Representation-selection trajectories for accuracy, PCK, and Brier.
7. Visual comparisons of anatomical, single-mode, permuted, and distance priors.
8. Additive class-margin explanations, predicted parts, visibility, and nearest training exemplars.
9. Performance slices from training-fitted 2D layout clusters, high layout distance, and low annotated visibility.
10. Targeted versus random occlusion at square sides 8%, 14%, and 20% of image size, plus global blur. These are image-space corruptions re-encoded with DINO, not feature-token deletion.
11. Occlusion severity plots and example images; accuracy, prediction flips, coverage, covered-part emission, surviving-part recall, and number of covered landmarks.

Robustness uses the same deterministic 256 report images for all variants and seeds. A square can cover several parts or background; the report makes this visible. These are stress tests, not realistic natural-occlusion ground truth or isolated causal part interventions. Report sample counts and avoid extrapolating small differences from this subset.

Layout clusters are diagnostic proxies, not flight-state, viewpoint, or novel-pose labels. The current experiment cannot prove that a model understands flying birds in 3D.

## 6. Decisions after this one batch

Read the planned comparisons before looking for the largest test score:

| Comparison | Interpretation if supported across seeds |
|---|---|
| Staged soft vs posthoc soft | Evidence mismatch can be reduced by classifier adaptation |
| Normalized vs staged soft | Available-evidence scaling helps |
| Graph ungated vs independent | Relations have value without visibility filtering |
| Graph normalized vs staged normalized | Relations add value to the chosen evidence mechanism |
| Graph vs permuted and distance controls | Specific anatomical relations add more than generic spatial structure |
| Graph mixture vs single Gaussian | Multimodal relation modeling is useful |
| Entropy variant vs ordinary graph | Confidence-weighted messages justify their extra mechanism |

Provisional engineering stop/go criteria, declared before v3 results: seek a meaningful missing-emission reduction at the 90% recall policy while losing no more than approximately one accuracy percentage point against the matched ungated reference. For the graph, a repeatable localization gain around 0.5 point or larger over matched no-graph, with superiority to wrong-prior/distance controls, is worth further study. These are practical targets, not significance cutoffs or publication guarantees. Examine uncertainty, absolute error counts, abstention, and failure cases together.

If normalization helps only one seed, do not lock it as a contribution. If graph gains disappear against distance/permuted controls, reduce the anatomical claim. If visibility filtering still causes large accuracy loss, retain the negative result and investigate calibration or classifier objectives before adding diffusion. If even mean-linear clearly dominates, quantify the interpretability cost explicitly.

Do not keep adding losses just because the code allows them. Keep the smallest mechanism supported by the controls. Archive failed variants and their actual results.

## 7. Full research pipeline beyond the executable batch

The following stages are necessary for a strong paper but are **not secretly implemented or executed by this notebook**.

### A. Audit and lock the method

Inspect annotation alignment figures, train-only prototype provenance, class balance, missing-part denominators, and the all-rejected case. Audit lateral part naming before introducing flips. Check whether an alleged absent part is merely unlabeled or occluded. Freeze the final architecture, thresholds, epoch-selection rule, and primary comparison after development.

### B. Reproduce published comparisons fairly on Kaggle

Reproduce the supplied non-parametric prototype paper with its official code/settings. Our supervised frozen-DINO models are not its exact reproduction. Add appropriate interpretable baselines, including a spatially flexible prototype method. Separate original published supervision from matched extra-annotation controls; report backbone, resolution, prototype count, trainable parameters, preprocessing, augmentation, and compute. A method receiving keypoint labels has extra information, so an accuracy difference alone does not establish a superior learning algorithm.

Include a strong ordinary classifier reference if claiming competitive classification. The current mean-patch linear reference is useful but insufficient for that claim. Fix tuning budgets before final comparisons.

### C. Confirm beyond the reused CUB test

Use a separately sourced, licensed dataset with compatible semantic-part annotations, or explicitly develop and validate the required annotation mapping. Do not assume another bird dataset has the same 15 CUB parts. Fit all banks and relation statistics using that dataset's training partition. Keep its evaluation untouched until the recipe is locked.

For a viewpoint/correspondence claim, [SPair-71k](https://cvlab.postech.ac.kr/research/SPair-71k/) supplies paired images and viewpoint/scale/occlusion variation annotations. It evaluates semantic correspondence, not CUB species classification. Implement its own part mapping, splits, and evaluation protocol; report it as a separate task. It can support a 2D viewpoint robustness claim without requiring 3D training data.

### D. Test supervision efficiency and actual generalization

After selecting the core method, use 25%, 50%, and 100% training part annotations with the same class-labeled images. Mask training annotations consistently across bank construction, geometry fitting, and losses. Ensure each class/part has sufficient support or document the fallback. Validation/test annotation use must remain evaluation-only except the predefined calibration split. This distinguishes annotation efficiency from simply training on fewer images.

Add natural occlusion subsets with trustworthy labels, and background interventions only if segmentation annotations support them. Maintain matched random interventions and report collateral part coverage. Do not call synthetic squares a causal proof.

### E. Consider diffusion only if a measured bottleneck warrants it

If local correspondence is the limiting error after the controls, test frozen diffusion features as an optional teacher and distill into the same small student. First compare an equally supervised DINO teacher/student with matched representation size and compute. Do not train a diffusion generator from scratch for this Kaggle project. Disclose teacher extraction cost and any extra pretrained data. A feature concatenation alone is not a demonstrated research contribution.

### F. Final paper package

Provide a locked protocol, three or more seeds where affordable, uncertainty, per-part error analysis, full ablations, failure cases, exact evidence decomposition, calibration trade-offs, external confirmation, official baseline reproduction, and an efficiency table. State annotation and pretraining assumptions. Release one reproducible Kaggle notebook plus configurations and saved metrics. Acceptance at an A* conference or Q1 journal depends on novelty and the eventual evidence; current results do not establish that level yet.

## 8. Kaggle operation and what to return

1. Upload the companion notebook and attach `/kaggle/input/datasets/wenewone/cub2002011`. Select **T4**, the configuration that worked for your prior run. Keep `REPAIR_LEGACY_CUDA=False`.
2. Use Internet for the pinned official DINO loader, or attach official source and weights and fill both local settings. No random-weight fallback exists.
3. Keep `PROFILE="full"` to run the complete batch. `smoke` is only a quick end-to-end check and cannot supply paper results.
4. Reuse the previous matching feature cache by setting `CACHE_INPUT` to its attached `cache` directory. A downloaded notebook file alone does not contain that cache. Default full feature storage is about 2.16 GiB, plus checkpoints, reports, and corruption caches.
5. Run All. Training histories record timing and peak GPU memory. Use observed timings to estimate remaining runtime; the full batch is not guaranteed to finish within one Kaggle session.
6. For an interrupted session, preserve the complete output. In a fresh session copy the saved suite directory back to `/kaggle/working/relational_cub_suite_v3` before Run All, preserving subdirectories and configuration. Model resume requires the saved `.pt` files, not only `decision_bundle.zip`. Do not mix a smoke cache/checkpoint with full-run artifacts.
7. Return **`/kaggle/working/relational_cub_suite_v3/decision_bundle.zip`**. It contains summary tables, per-seed metrics, paired intervals, calibration thresholds, compact predictions, robustness measurements, figures, split/configuration provenance, and environment information. Save the entire notebook Output separately to retain models and large caches.

The batch supplies enough measurements to make a focused next decision; further GPU experiments have not been run locally. The notebook has been validated end to end on a synthetic four-class fixture with a deterministic mock backbone, including all variants, calibration boundaries, exact contribution sums, freezing, and interrupted resume. Real CUB/DINO/T4 execution remains the user's Kaggle run.
