# RDYOLO manuscript: revision report

Project: `main.tex`, `references.bib`, `figures/`. Build: `pdflatex` → `bibtex` → `pdflatex` ×2. Tested: it compiles to 21 pages with no errors, no undefined references or citations, and no BibTeX warnings. The only remaining log messages are four cosmetic underfull-box notes.

Nature of the paper: a design and evaluation-protocol manuscript. It contains **no trained-model results**, and I did not add any.

## A. Grammar and academic language
- The source had many sentences with missing words. Examples: "three- asphalt surfaces", "anchor- detection", "-scale damage localisation", "direction- road-damage detector", "a priority than", "an testable", "At three training seeds", "answer questions … carry information..". All were restored to the evident meaning.
- A duplicated sentence (YOLOv8-PD / SEA-YOLO v8) was removed.
- Garbled passages were rewritten: Introduction paragraphs 1–2, 5–7; Related work §2.3 (Maeda, RDD2020, "Kaggle split"); CarNet; the Dataset-audit and Training-controls paragraphs; the bootstrap-interval sentence.
- A first-person aside ("I think that would have to count as a Kaggle split") was replaced by a neutral protocol statement.
- Spelling is now consistently British ("analysed", "labelled").
- Terminology is consistent: mAP$_{50}$ / mAP$_{50:95}$ throughout, and "Section" instead of "Sec.".
- Named-author citations now use `\citet`, so author names and numbers cannot drift apart.

## B. Technical and functional corrections
- **Unsupported claim removed.** The Introduction said RDYOLO "demonstrates this", although nothing is evaluated yet. It now says RDYOLO "tests this". The contribution list was likewise changed from "presents an analysis" to "specifies an analysis".
- **Abstract.** "Audited partition" now reads "partition … audited before use", since the audit is still to be done. "On synthetic inputs" was added to the checks.
- **Pothole description.** The Introduction said a pothole "may be small, but be very wide", which contradicted the abstract. It was harmonised to the abstract's wording (variable apparent size with distance and lighting). Please confirm this is what you mean.
- **Notation clashes.** The head $H$ conflicted with image height, so it is now $\mathcal{H}$. The parameter count $P$ conflicted with the pyramid features and precision, so it is now $N_{\rm par}$. Throughput time $T$ conflicted with $T_3,T_4$ and the IoU set $\mathcal{T}$, so it is now $t$. $C_+=|\mathcal{C}_+|$ is now defined.
- **CIoU symbols** ($\boldsymbol m$, $w,h$, $\widehat w,\widehat h$) were undefined and are now defined.
- **Figure 2 caption.** "3 heads for box prediction and four heads for damage-class prediction" was wrong. It now reads: three scale-specific heads, each with a box branch and a four-class branch. The stray "keypoint branch" mention was removed.
- **Figure 1 caption.** The undefined asterisk is now explained.
- **Confusion-matrix wording.** It now reads "5×5 covers four classes plus background".
- **Unreferenced tables.** Tables 1 and 4 were never cited in the text and now are.
- **Verified by recomputation.** I rebuilt the parameter counts and conv-MACs for A0–A3 analytically from the manuscript's own architecture description. Every Table 3 value matches exactly (3,011,612 / 2,610,376 / 3,015,357 / 2,614,121; 4.042 / 3.634 / 4.084 / 3.676 GMACs). The $2h^2+23h$ vs $18h^2$ weight claim, the gate parameter count (129), the detail-branch count (3,616), the 8,400 locations and the 68-channel head are all consistent.
- Loss constants, TAL (α=0.5, β=6, top-10), DFL clipping at 14.99, NMS 0.7 / 300, and the COCO maxDets and 101-point conventions match the Ultralytics v8.2.0 and COCO conventions as I know them. I could not re-run the Ultralytics code here.

## C. Redundant or unnecessary content removed
- The duplicated YOLOv8-PD / SEA-YOLO v8 sentence.
- Filler phrasing and restated points in the Introduction and Related work.
- The "keypoint branch" caption fragment.
- No scientific content was removed.

## D. Formatting
- Section headings are in sentence case ("Related work").
- No bold or italic emphasis occurs in running text. Bold remains only in table headers, and the italic remains only in the longtable continuation lines.
- `hyperref` now loads last in the preamble.
- Template, document class, labels, citation keys and file structure are unchanged.

## E. Structural and logical improvements
- **Added §4 "Limitations and conclusion".** The paper had no conclusion. The new section uses only statements already in your manuscript: what the counts and checks do and do not show, the limitations, and what the contribution rests on. Please review it; delete it if you prefer to hold the conclusion for the results paper.
- The roadmap paragraph now includes the new section.

## F. Citation and reference issues
- **All 32 DOI-bearing records resolve in Crossref.** Title, year, journal, volume and pages match `references.bib`. Four entries have no DOI (Strip Pooling, GFL, Ultralytics, COCO API), so I could not check them this way.
- Abstracts support your descriptions of RDD-YOLO, ACD-YOLOv8, YOLO-RD, YOLOv11-ATL, YOLOv8-PD, SEA-YOLO v8, YOLO-LRDD (RDD2020 plus Chinese imagery), Jiang 2024 (Deformable Attention Transformer, GSConv, MPDIoU), LE-YOLOv5 and LBN-YOLO.
- All 36 `\cite` keys exist in the `.bib`, and there are no uncited entries.
- Bib fixes: "RDD2022 :" spacing; the `number` field was removed from `zheng2020` (BibTeX warned that volume and number conflict).
- Intro sentence on YOLO-LRDD was made more precise: "RDD2020 augmented with Chinese road imagery".

## G. Remaining concerns for author verification (not changed)
1. **Reference code not in the ZIP.** The text repeatedly cites the "supplied architecture code" and a "reference implementation". My recomputation confirms the counts are consistent with the described architecture, but not with your actual code. Include the code, or a supplement, for reproducibility.
2. **Learning-rate schedule.** "Cosine decay to 0.001" is ambiguous. In Ultralytics, `lrf` is a final *ratio* (default 0.01, giving 1e-4). State whether the final LR is 0.001 or the ratio is 0.001.
3. **Batch size and accumulation.** With Ultralytics defaults (`nbs=64`), batch 8 accumulates gradients 8×. The claim "no hidden gradient accumulation" requires `nbs=8`. Weight decay is also scaled by `nbs`.
4. **Augmentation values.** Scale ±0.2 and HSV (0.015, 0.4, 0.2) differ from Ultralytics defaults. That is fine if intentional, but confirm they match the configuration used.
5. **Unreported costs.** The text says the cost of A4 is reported, but no A4 or A5 counts appear in Table 3 (and no counts for the P2-head baseline). Add them or soften the sentence.
6. **Unverified descriptions.** Metadata checks could not confirm the summaries of Silva, Li (CarNet), Li (YOLOX-RDD), Roy, Dong, Wang (GSBYOLO), Wu, Zhong and Tang, because Crossref has no abstracts for them. Examples are "Rain365 and Sun520" and "UAV-PDD2023". Confirm these against the papers.
7. **Wang et al. (2026).** The abstract describes strip convolution and feature equalisation on an "RDD2022-based dataset" but does not say "neck" or "YOLOv8n". Your claim that this is "directional filtering in a YOLOv8n neck" needs confirming in the full text, since it carries much of the novelty argument.
8. **Verification claim.** The statement that bibliographic records and summaries were checked against publisher text is yours. I checked the DOI metadata only.
9. **COCO API reference.** `cocoeval` is dated 2026 (it looks like an access year). Consider citing Lin et al., *Microsoft COCO* (ECCV 2014) and the repository version used.
10. **Kaggle path.** The data source is a Kaggle path only. Add the dataset URL, version and licence.
11. **Figure 1 photograph.** Confirm its source and licence.
12. **Run-time claims.** "Lightweight" in the title rests on parameter count and conv-FLOPs only.

## H. Critical issues
**Confirmed:**
- The paper has no trained results. It cannot support any accuracy, generalisation or speed claim, and it does not make any.
- The conclusion was missing (now added).
- The garbled sentences listed in A (fixed).

**Suspected:** items G2, G3 and G7.

**Unresolved:** items G1, G5 and G6.

## I. Readiness assessment
| Aspect | Status |
|---|---|
| Language quality | Good after revision; a final read by you is advised. |
| Technical consistency | Internally consistent; counts reproduced. Training details need G2–G4 settled. |
| Methodological clarity | Clear and detailed. A4/A5 costs are missing. |
| Citation integrity | DOIs verified. Several content summaries are unconfirmed (G6, G7). |
| Formatting | Clean generic article; **no journal template or limits have been checked**. |
| Journal readiness | **Not ready for a results-based journal.** The paper is a protocol and design paper. It becomes submittable once trained results are added, or if a venue accepting protocol or registered-report papers is chosen. Make the title and abstract state this status if submitted as is. |

Acceptance is not guaranteed and no journal's requirements have been checked.
