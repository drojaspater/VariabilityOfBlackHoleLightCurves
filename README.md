# Light Propagation Prescriptions for Black Hole Movies

[![arXiv](https://img.shields.io/badge/arXiv-2605.12659-b31b1b.svg)](https://arxiv.org/abs/2605.12659v1)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code and analysis pipeline associated with the paper:

> **Light Propagation Prescriptions for Black Hole Movies**
> [arXiv:2605.12659](https://arxiv.org/abs/2605.12659v1)

The project extends the [AART](https://github.com/iAART/aart/) code to generate black hole movies in **fast-light** mode and introduces a new intermediate prescription called **brisk-light**, which is the central methodological contribution of this work.

---

## Table of Contents

- [Overview](#overview)
- [Physical Background](#physical-background)
  - [What is AART?](#what-is-aart)
  - [Lensing Bands and Photon Rings](#lensing-bands-and-photon-rings)
  - [Light Propagation Prescriptions](#light-propagation-prescriptions)
  - [The Brisk-Light Prescription](#the-brisk-light-prescription)
- [References](#references)

---

## Overview

When constructing a black hole movie — a time series of images — one must choose a **light-propagation prescription**: a rule that decides which source time each photon in an image is sampled from, given that photons travel different path lengths before reaching the observer.

This repository implements and studies three such prescriptions using the AART framework:

- **Slow-light**: the physically accurate prescription, where each photon in a snapshot is sampled at the source time corresponding to its individual geodesic travel time.
- **Fast-light**: a common approximation where all photons in a snapshot are sampled from the same single source emission time.
- **Brisk-light**: a new prescription introduced in this work, designed as a computationally efficient intermediate between slow and fast light.

The core contribution of this repository is the **slow-light movie generation pipeline** built on top of AART, and the **brisk-light implementation** as a novel prescription described in the associated paper.

---

## Physical Background

### What is AART?

**AART** (Adaptive Analytical Ray Tracing) is a numerical framework developed by Cárdenas-Avendaño, Lupsasca, & Zhu (2023) [[1]](#references) that exploits the integrability of the Kerr spacetime to compute high-resolution black hole images. Key features:

- Analytical ray-tracing on a **non-uniform adaptive grid** in the observer's image plane, specially designed to resolve photon rings.
- Computation of radio visibilities on long interferometric baselines, relevant for VLBI experiments such as the Event Horizon Telescope.
- Modular design that supports equatorial sources with complex emission profiles and time variability.
- Source variability modeled via **inoisy** (Lee & Gammie 2021), a generator of Gaussian random fields with tunable, position-dependent correlations.

This project builds directly on AART to extend its movie-generation capabilities under the fast-light prescription, and introduces brisk-light as a new mode.

### Lensing Bands and Photon Rings

In the Kerr spacetime, photons reaching a distant observer are classified by the number of half-orbits *n* they complete around the black hole before escaping:

| Order *n* | Description |
|---|---|
| n = 0 | Direct image — photons that reach the observer with no additional half-orbits |
| n = 1 | First-order lensed image — one additional half-orbit around the black hole |
| n = 2 | Second-order lensed image — two half-orbits; contributes to the photon ring |
| n → ∞ | Converges to the critical curve: the photon ring |

Each order occupies a distinct **lensing band** in the image plane. Higher-order images are narrower, more magnified, and — critically for this work — carry **longer and more spread-out geodesic time delays**. The distribution of these delays across lensing bands is the geometric structure that motivates the brisk-light prescription.

### Light Propagation Prescriptions

The spatiotemporal content of a black-hole movie is jointly determined by the intrinsic source variability and by the distribution of light-travel times across the image. When building a snapshot at a fixed observer time *t*, the key question is: *at what source time was each photon emitted?*

**Slow-light** answers this correctly: each photon carries a geodesic travel time τ, so it was emitted at source time *t* − τ. A single snapshot therefore samples the source at a spread of emission times, and two photons arriving at the same observer time may have left the source at very different moments.

**Fast-light** answers approximately: every photon in a snapshot is assigned the same emission time, ignoring the differences in travel time. This is the standard simplification used in many black hole movie pipelines and is valid only when the source variability timescale is much longer than the geodesic delay spread.

When the **variability timescale is comparable to or shorter than the delay spread** — which is increasingly relevant for rapidly varying sources — the mismatch between the two prescriptions becomes significant, particularly at high observer inclinations where the delay spread is largest.

### The Brisk-Light Prescription
 
The central methodological contribution of this work is **brisk-light**, a new intermediate prescription that sits between slow-light and fast-light in both physical accuracy and computational cost.
 
Rather than retaining the full screen-dependent delay field (slow-light) or collapsing the entire image to a single source time (fast-light), brisk-light **preserves the dominant temporal interval of each lensing band and clips only the low-density tails**. It is therefore a bandwise, geometry-guided support reduction of slow light.
 
The construction works as follows:
 
1. **Delay distribution per band.** For each lensing band *n*, the distribution of sampled emission times is estimated using a Gaussian kernel density estimate (KDE), built from a trimmed sample (central fraction *q* = 0.995) to avoid artificially broad supports caused by a small number of extreme tail pixels near the horizon or at large disk radii.
2. **Modal highest-density interval (HDI).** From the KDE, the modal time $\bar{t}_n$ and a connected interval $\mathcal{T}_{n,p}$ around it are identified, where *p* controls what fraction of the probability mass is enclosed. Lowering a density threshold around the modal peak until mass *p* is enclosed defines the interval.
3. **Clipping.** The slow-light emission time of each pixel is clipped to the nearest endpoint of $\mathcal{T}_{n,p}$ if it falls outside the interval; pixels already inside are left unchanged. Only the temporal argument of the source intensity is modified — geodesics, redshift factors, and source positions are untouched.
The parameter *p* controls the hierarchy of prescriptions: at *p* = 0 every pixel in a band evaluates the source at the single modal time $\bar{t}_n$; at *p* = 1 the interval covers the full support and brisk-light recovers slow-light exactly. This places brisk-light in a well-defined intermediate position: **fast-light retains one global source time, brisk-light retains a bandwise modal support, and slow-light retains the full screen-dependent time map.**
 
The computational saving comes not from reducing the number of ray-traced pixels, but from reducing the temporal support that must remain available during source interpolation or snapshot loading — a significant gain for stored source movies.

---

## References

1. Cárdenas-Avendaño, A., Lupsasca, A., & Zhu, H. (2023). **Adaptive Analytical Ray Tracing of Black Hole Photon Rings**. *Physical Review D*, 107, 043030. [arXiv:2211.07469](https://arxiv.org/abs/2211.07469)

2. Gralla, S. E., & Lupsasca, A. (2020). **Lensing by Kerr black holes**. *Physical Review D*, 101, 044031.

3. Gralla, S. E., & Lupsasca, A. (2020). **Null geodesics of the Kerr exterior**. *Physical Review D*, 101, 044032.

4. Gralla, S. E., Lupsasca, A., & Marrone, D. P. (2020). **The shape of the black hole photon ring: A precise test of strong-field general relativity**. *Physical Review D*, 102, 124004.

5. Lee, D., & Gammie, C. F. (2021). **inoisy: A Gaussian random field generator for black hole image variability modeling**.

---
