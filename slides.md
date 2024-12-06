---
title: traffic series \#2
subtitle: Training a Machine Learning Model to Detect Holding Patterns from ADS-B data
author:
  - Xavier Olive
  - Luis Basora
  - Junzi Sun
  - Enrico Spinielli
institute:
  - ONERA \quad TU Delft \quad EUROCONTROL
date: OpenSky Symposium 2024
aspectratio: 169
autopandoc: pandoc slides.md -t beamer -o slides.pdf --pdf-engine=xelatex
autopandoc-onsave: true
comment: weird to add the theme again here, but there's a bug in pandoc 3.5
header-includes:
  - \usetheme{metropolis}
  - \metroset{numbering=fraction}
  - \setmonofont[Scale=0.85]{Inconsolata}
  - \setmathfont[Scale=0.95]{Fira Math Light}
---

## Introduction

Focus on a specific method in the `traffic` library:

```python
>>> flight.holding_pattern()
>>> flight.next("holding_pattern")
>>> flight.label("holding_pattern")
```

Contributions:

- a labelled dataset: [[doi:10.4121/20411868]{.underline}](https://data.4tu.nl/articles/_/20411868)  
  (2 months of data at `EGLC`, `EGLL`, `EHAM`, `EIDW`, `LFPG`, `LSZH`)
- a ML model (`.onnx` files: _scaler_ and _classifier_)
- an implementation in `traffic`

## Definition

**holding pattern** _(noun)_: flight manoeuvre used to keep an aircraft while awaiting further instructions. Holding patterns are typically used when air traffic congestion or adverse weather conditions prevent an aircraft from landing immediately or progressing along its intended route.

An aircraft usually follows a \alert{racetrack-like loop}, typically flying a **straight path** for a set distance, then making a **turn to complete the loop**.

Rule-based method? ML-based method?

## Illustration

![ ](figures/holding_patterns_transparent.png)

## Bestiary (page 1)

![ ](figures/holding_01.png){ width=45% }
![ ](figures/holding_02.png){ width=45% }

## Bestiary (page 2)

![ ](figures/holding_04.png){ width=45% }
![ ](figures/holding_06.png){ width=45% }

## Bestiary (page 3)

![ ](figures/holding_03.png){ width=45% }
![ ](figures/holding_05.png){ width=45% }

## Approach

\newfontfamily\symbolfont{Noto Sans Symbols}
\def\arrow{{\symbolfont ➤}}

- We do not have a labelled dataset
- We want more than a yes/no answer: classification \arrow detection
- We know few things about _deep clustering:_ build a latent space so that  
  two resembling trajectories will be neighbours on a low dimension space

**Step 1:** Label a dataset (_unsupervised_ learning, then manual labour)  
**Step 2:** Train a model (_supervised_ learning)

# Step 1: Label a dataset

## Sliding windows

We need to identify parts of trajectories, so we will proceed with _sliding windows:_  
(classification vs. detection)
\vspace{-.5em}

```python
flight.sliding_windows("6 min", "2 min")
```

![ ](figures/sliding_windows_transparent.png){ width=30% }
![ ](figures/sliding_windows_00.png){ width=30% }

## Structuration of the latent space

After training a basic autoencoder `30-8-2-8-30` to create a latent space, we note that holding patterns tend to cluster:

![ ](figures/latent_space.png){ width=70% }

## Trajectory in the latent space

We can identify the holding pattern part of the trajectory in the latent space:

![ ](figures/latent_trajectory.png){ width=70% }

## Clustering of the latent space

Now, we need to identify those "holding pattern" areas for a pre-labelling:  
**clustering the latent space** (Gaussian Mixture for instance)

![ ](figures/latent_clustering.png){ width=40% }

## From a pre-labelling to a labelling

- Now we can mark all segments in the proper cluster as _"holding pattern"_
- But in fact, it was quite average (imagine _Google Translate level_)

We adjusted all the labels manually with a visual tool:

- left plot (_"holding patterns"_):  
  click to mark as "**not** holding pattern"
- right plot ("**not** holding pattern"):  
  click to mask as _"holding pattern"_

## Labelling the dataset

\newfontfamily\symbolfont{Noto Sans Symbols}
\def\checkmark{{\fontsize{40pt}{48pt} \symbolfont ☑ }}

\Huge

**GET THE SHIT**

\alert{\checkmark \textbf{DONE}}

\vspace{.5em}

\normalsize

97,700 flights \quad 784,177 segments (49,822 positive)

\pause

_The biggest, most horrendous, terribly monstrous part of the work—the absolute pinnacle of frustration and despair, the unparalleled abyss of toil and tedium, the most impossibly dreadful and gruelling segment of it all._

# Step 2: Train a classifier

## Supervised learning with a NN classifier

- `flight.sliding_window("6 min", "2 min") if segment.duration >= "5 min"`
- Feature: _unwrapped track angle_ | `MinMaxScaling()`
- **Convolutional architecture** (significantly better than flattened):

  ```text
     Linear(30, 8) | Reshape(-1, 1, 8) | Conv1d(1, 10, 3) | ReLU()
   | MaxPool1d(2) | Flatten() | Linear(30, 8) | ReLU()
   | Linear(8, 4) | ReLU() | Linear(4, 1) | Sigmoid()`
  ```

- train/test splitting (~ 80%)

- Metrics: accuracy, precision, recall, f1-score, IoU

> Best results with `EIDW` as testing set (precision: 0.85, recall: 0.75, f1: 0.76, IoU: ~0.6 but...)

# Conclusion

## Key take-aways

- Unsupervised learning helps to bootstrap a labelling
- Now, we have a labelled dataset for holding patterns (cost: _blood and tears_)
- Models exported in `.onnx` format: usable in any programming language

```python
from traffic.data.datasets import landing_heathrow_2019

subset = landing_heathrow_2019.has('holding_pattern').eval().sample(600)
subset.plot(color="#bab0ac")  # gray
for hp in subset.holding_pattern():
    hp.plot(color="#f58518")  # orange
```

## Visuals

\centering

![ ](figures/heathrow_transparent.png){ width=50% }
