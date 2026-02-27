# Nuclear–Neural Hybrid Mass Model

This repository contains the implementation of a physics-informed sequential residual neural network framework for nuclear mass modeling.

## Overview

The model combines:
- A physics-based baseline (existing mass models acting as a major structural component)
- A neural network residual correction acting as a controlled minor adjustment

The framework is designed to:
- Preserve global physical trends
- Regulate smoothness through systematic regularization scheduling
- Prevent artificial high-frequency fluctuations
- Enable spectral analysis of post-correction residuals

## Methodology

The study involved a systematic series of approximately 250 sequential neural-network trainings, where:

- The residual from each stage is propagated to the next
- Regularization strength is gradually varied
- Smoothness and physically meaningful trends are isolated progressively

This structured residual strategy forms the basis of an upcoming manuscript currently in preparation.

## Contents

- Main training implementation-STEP_1 (.py files for FFNN, GNN(APPNP), MoE with the actual codes and .ipynb files to apply them on respective data files)
- Input data used(.xlsx files)
- Subsequent training implementation-STEP_2(.py files for FFNN, GNN(APPNP), MoE with the actual codes and .ipynb files to apply them on respective data files)

## Status

Research code accompanying a manuscript in preparation.
