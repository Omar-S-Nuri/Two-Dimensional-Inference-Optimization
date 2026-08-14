
# AIR — Adaptive Inference Routing

**Research Project**

AIR (Adaptive Inference Routing) is a research framework for
dynamic inference optimization in deep Transformer architectures.

The project investigates whether inference in autoregressive Transformer
models can be accelerated by dynamically adapting the computational path
to the characteristics of the current input.

The framework combines two complementary optimization dimensions:

- **Vertical inference optimization** through dynamic layer routing and
  post-training operator fusion.
- **Horizontal inference optimization** through Asynchronous Multi-Token
  Chaining (MTC) and numerical Token-ID prefix structures.

The framework additionally incorporates hidden-state-based validation and
an early-abort mechanism for detecting inputs that are unsuitable for
vertical compression.

---

## Scientific Paper

### Two-Dimensional Inference Optimization in Deep Transformer Architectures

**Dynamic-Adaptive Layer Routing, Post-Training Operator Fusion,
Asynchronous Multi-Token Chaining, and Evolutionary Inference Validation**

**Author:**  
Omar Nuri  
Independent Researcher  
Computer Science and Machine Learning Systems

**Date:** August 2026

**GitHub Repository:**  
https://github.com/Omar-S-Nuri/Two-Dimensional-Inference-Optimization

---

## Research Concept

Modern Transformer language models normally execute a largely fixed
sequence of computational layers for every generated token.

AIR investigates a different approach.

Instead of treating every token as requiring the same computational depth,
the system analyzes intermediate model representations and attempts to
identify computationally reusable paths.

The architecture is organized around two orthogonal dimensions.

```text
                         AIR
              Adaptive Inference Routing
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   VERTICAL DIMENSION           HORIZONTAL DIMENSION
          │                             │
   Layer Routing                 Multi-Token Chaining
          │                             │
   Operator Fusion                Token-ID Trie
          │                             │
   Shortcut Validation            In-Flight Learning
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
                 2D Inference Runtime
````

---

# 1. Vertical Inference Optimization

The vertical dimension operates along the depth of the Transformer.

The system captures intermediate activations and analyzes the evolution of
the model's output distributions across layers.

When two layers exhibit sufficiently high alignment, the corresponding
computational path can become a candidate for a shortcut.

The general workflow is:

```text
Input
  │
  ▼
Transformer Layers
  │
  ├── Layer 0
  ├── Layer 1
  ├── Layer 2
  ├── ...
  ├── Layer N
  │
  ▼
Activation / Logit Analysis
  │
  ▼
Layer Equivalence Detection
  │
  ▼
Candidate Shortcut
  │
  ▼
Shadow Validation
  │
  ▼
Operator Fusion
  │
  ▼
Validated Fast Path
```

The objective is to allow suitable inputs to bypass unnecessary
computational layers without fine-tuning the original model.

---

# 2. Dynamic-Adaptive Layer Routing

AIR uses activation information collected during model execution.

The system records intermediate representations through forward hooks and
uses these representations for subsequent analysis.

For supported architectures, the runtime captures:

* residual-stream activations
* MLP activations
* model logits
* token IDs
* layer-level information required for shortcut analysis

The current implementation provides model-specific infrastructure for:

* Meta Llama architectures
* GPT-2

The architecture-specific implementations are separated into dedicated
model-base modules.

---

# 3. Post-Training Operator Fusion

Candidate layer shortcuts can subsequently be processed by the operator
fusion subsystem.

The basic idea is to replace a sequence of individually executed
operations with a mathematically constructed fused operator.

Conceptually:

```text
Layer A
   │
   ▼
Layer B
   │
   ▼
Layer C
   │
   ▼
Layer D
```

may become:

```text
Layer A
   │
   ▼
Fused Operator
   │
   ▼
Layer D
```

The fusion process is therefore intended to reduce unnecessary intermediate
computation during inference.

AIR does not require modification of the original model parameters through
conventional fine-tuning for the shortcut-generation process.

---

# 4. Shortcut Validation

A candidate shortcut is not automatically considered valid.

AIR uses a validation stage to distinguish potential computational
shortcuts from routes that could negatively affect model behavior.

Candidate shortcuts can progress through states such as:

```text
DISCOVERED
    │
    ▼
PENDING
    │
    ▼
VALIDATION
    │
    ▼
VALIDATED
```

The validation infrastructure is designed to allow different validation
windows for research/training and production configurations.

---

# 5. Horizontal Inference Optimization

The second optimization dimension operates across the token sequence rather
than across model depth.

Autoregressive language models normally generate tokens sequentially:

```text
t1 → t2 → t3 → t4 → t5 → ...
```

AIR investigates whether highly predictable token sequences can be recognized
and reused as learned chains.

The horizontal component is called:

**Asynchronous Multi-Token Chaining (MTC)**

The system maintains a numerical representation of learned token
transitions using Token IDs rather than storing complete text phrases.

Conceptually:

```text
Token A
   │
   ▼
Token B
   │
   ▼
Token C
   │
   ▼
Token D
```

can be represented as a numerical prefix structure.

---

# 6. Token-ID Trie

The horizontal memory is implemented as a nested numerical structure.

A simplified representation is:

```text
prefix_token_1
    │
    └── prefix_token_2
            │
            ├── next_token_1
            ├── next_token_2
            └── next_token_3
```

The implementation stores numerical token IDs rather than textual phrases.

This keeps the runtime structure compact and avoids textual search during
inference.

The current implementation uses the final two observed token IDs as the
prefix context for horizontal chain registration.

---

# 7. In-Flight Learning

Horizontal chains can be learned during processing of a corpus or during
runtime operation.

The general process is:

```text
Input Text
    │
    ▼
Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Model Forward Pass
    │
    ▼
Logits
    │
    ▼
Top Token Probability
    │
    ▼
Confidence Threshold
    │
    ├── below threshold → no chain
    │
    └── above threshold → register chain
```

The resulting knowledge can be stored and reused by later inference runs.

---

# 8. Current Experimental Configuration

The central parameters are defined in `config.py`.

The current configuration is:

```python
GLOBAL_THRESHOLD = 0.92

EARLY_ABORT_THRESHOLD = 0.45

STABILITY_WINDOW_TRAINING = 1
STABILITY_WINDOW_PRODUCTION = 2
```

## Global Threshold

```python
GLOBAL_THRESHOLD = 0.92
```

The current implementation uses this value as the principal confidence /
similarity threshold for shortcut registration.

For vertical routing, it is used as the minimum cosine similarity between
layer-level probability distributions required to register a potential
layer shortcut.

For horizontal Multi-Token Chaining, it is used as the minimum top-token
probability required to register a token transition.

The value `0.92` represents the **current experimental configuration**.
It should not be interpreted as a universally established mathematical
optimum.

## Early-Abort Threshold

```python
EARLY_ABORT_THRESHOLD = 0.45
```

The Early-Abort mechanism is used during vertical shortcut analysis.

The system first compares the probability distributions of an early layer
and an intermediate layer.

If the measured similarity falls below the configured threshold, the
vertical layer-equivalence scan is aborted for the current input.

This is intended to avoid unnecessary analysis of inputs for which
layer-level compression is unlikely to be useful.

## Shadow Validation

```python
STABILITY_WINDOW_TRAINING = 1
STABILITY_WINDOW_PRODUCTION = 2
```

These parameters define the configured validation window for candidate
shortcuts.

The training configuration uses an accelerated validation setting, while
the production configuration is more conservative.

---

# 9. Two-Dimensional Runtime

The two optimization dimensions are intended to operate together.

```text
                         INPUT
                           │
                           ▼
                    Transformer Model
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      Vertical Analysis          Horizontal Analysis
             │                           │
       Layer Similarity              Token IDs
             │                           │
       Shortcut Candidate          Token Probability
             │                           │
       Shadow Validation            Token-ID Trie
             │                           │
       Operator Fusion             MTC Candidate
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                  Adaptive Runtime Path
```

The intended result is a runtime capable of selecting between conventional
execution and learned computational shortcuts.

---

# 10. Supported Model Infrastructure

The repository currently contains separate model adapters for different
Transformer architectures.

### Llama

The Llama implementation uses:

```text
AdaptiveLlamaBase
```

and supports the local loading and instrumentation of:

```text
Meta Llama 3.2 1B
```

The implementation uses forward hooks to capture intermediate layer
activations.

### GPT-2

The GPT-2 implementation uses:

```text
AdaptiveTransformerBase
```

and provides the corresponding activation-capture infrastructure for the
GPT-2 architecture.

---

# 11. Repository Structure

The repository is organized around the inference engine, model adapters,
analysis, routing, validation, and benchmarking components.

A typical project structure is:

```text
Two-Dimensional-Inference-Optimization/
│
├── config.py
│
├── shortcut_engine.py
│
├── inference_onthefly_v2.py
│
├── train_router_backbone_2_6_2.py
│
├── model_base.py
├── model_base_llama.py
│
├── analyzer.py
│
├── compiler.py
├── compiler_llama.py
│
├── shadow_validator.py
│
├── app_onthefly.py
│
├── run_benchmarks.py
│
├── training_corpus/
│
├── paper/
│
├── README.md
│
└── LICENSE
```

The exact repository contents may change as the research implementation
evolves.

---

# 12. Training / Research Pipeline

The experimental backbone-training process is designed to process corpus
sentences and incrementally build inference knowledge.

A simplified pipeline is:

```text
Training Corpus
      │
      ▼
Sentence Loader
      │
      ▼
Model Forward Pass
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
Vertical Analysis      Horizontal Analysis
      │                      │
      ▼                      ▼
Layer Similarity        Token Probability
      │                      │
      ▼                      ▼
Shortcut Candidate       Token Chain
      │                      │
      ▼                      ▼
Shadow Validation        Trie Storage
      │                      │
      └──────────┬───────────┘
                 ▼
          Persistent Storage
```

The training pipeline also maintains a model-specific processing history so
that previously processed corpus files can be skipped on subsequent runs.

---

# 13. Persistent Knowledge Storage

AIR stores the learned shortcut structures separately from the base model.

The current configuration defines model-specific storage paths:

```python
STORAGE_PATH_GPT2 = ...
STORAGE_PATH_LLAMA = ...
```

The shortcut engine stores both dimensions in a common payload structure:

```python
{
    "vertical": ...,
    "horizontal": ...
}
```

This allows vertical and horizontal inference knowledge to be loaded
incrementally when the runtime starts.

---

# 14. Experimental Results

The accompanying paper reports experimental measurements using
Meta Llama 3.2 1B on x86 CPU infrastructure.

The reported integrated 2D runtime achieved a measured latency reduction
relative to the reported baseline in the corresponding experiment.

The paper reports:

```text
Slow-Track baseline:       19,029.58 ms
Integrated 2D-Track:       17,575.56 ms

Reported latency change:   -7.64%
```

The repository should be considered a research implementation. Reported
measurements are experimental results under the configurations described
in the paper and should not be interpreted as universal performance
guarantees across hardware, models, datasets, or software environments.

---

# 15. GPU Projection

The paper also discusses a projected performance range for GPU
infrastructure such as NVIDIA H100-class systems.

The reported projection estimates a potential latency reduction in the range
of:

```text
30% – 50%
```

This is a projection rather than a measured result from the current
repository and should therefore be distinguished from the reported x86 CPU
measurements.

---

# 16. Research Status

AIR is an experimental research project.

The current work focuses on:

* dynamic inference routing
* post-training operator fusion
* shortcut discovery
* shortcut validation
* Token-ID based horizontal chaining
* in-flight learning
* CPU inference optimization
* two-dimensional inference architectures

The system is under active development.

Interfaces, algorithms, thresholds, validation mechanisms, and storage
formats may change as experimental results are obtained.

---

# 17. Reproducibility

The repository is intended to make the research implementation available
for inspection and experimentation.

To reproduce experiments, the following factors should be recorded:

* model architecture
* model version
* model weights
* tokenizer version
* Python version
* PyTorch version
* Transformers version
* hardware
* numerical precision
* corpus
* benchmark configuration
* AIR configuration parameters

Performance results can vary substantially depending on these parameters.

---

# 18. Important Experimental Note

AIR does not claim that every token or every input benefits from
compression.

The central hypothesis is that Transformer inference contains regions of
computational redundancy and highly predictable token transitions that can
potentially be exploited dynamically.

The system therefore attempts to identify suitable cases rather than
forcing every input through an optimized path.

The Early-Abort mechanism and validation infrastructure are important parts
of this design.

---

# 19. Research Hypothesis

The central research hypothesis can be summarized as:

> **Transformer inference can potentially be accelerated by dynamically
> adapting both computational depth and token-sequence execution to the
> observed predictability of the current input.**

The vertical dimension addresses computational depth.

The horizontal dimension addresses sequential token generation.

Together they form the two-dimensional inference optimization approach
investigated by AIR.

---

# 20. Citation

If you use the concepts, implementation, or experimental results from this
repository in academic work, please cite the accompanying paper.

```text
O. Nuri,
"Two-Dimensional Inference Optimization in Deep Transformer Architectures:
Dynamic-Adaptive Layer Routing, Post-Training Operator Fusion,
Asynchronous Multi-Token Chaining, and Evolutionary Inference Validation,"
2026.
```

---

# 21. License

This repository is distributed under the license specified in:

```text
LICENSE
```

Please review the license before using, modifying, or redistributing the
software.

---

# 22. Author

**Omar Nuri**

Independent Researcher
Computer Science and Machine Learning Systems

GitHub:

[https://github.com/Omar-S-Nuri/Two-Dimensional-Inference-Optimization](https://github.com/Omar-S-Nuri/Two-Dimensional-Inference-Optimization)

---

# 23. Disclaimer

This repository contains experimental research software.

The implementation is provided for research and evaluation purposes.
No guarantee is made that the optimization mechanisms preserve model
behavior or improve inference performance under all configurations.

Performance results reported in the accompanying paper are tied to the
specific experimental conditions under which they were obtained. PLEASE NOTE: Without any guarantee.

---

## Project

**AIR — Adaptive Inference Routing**

**Two-dimensional inference optimization for deep Transformer
architectures.**

```
