# Two-Dimensional Inference Optimization in Deep Transformer Architectures

## AIR — Adaptive Inference Routing

**Research Project by Omar Nuri**  
Independent Researcher — Computer Science and Machine Learning Systems

**Paper:**  
*Two-Dimensional Inference Optimization in Deep Transformer Architectures: Dynamic-Adaptive Layer Routing, Post-Training Operator Fusion, Asynchronous Multi-Token Chaining, and Evolutionary Inference Validation*

**GitHub Repository:**  
https://github.com/Omar-S-Nuri/Two-Dimensional-Inference-Optimization

---

## Abstract

Modern Large Language Models (LLMs) rely on homogeneous, deep-layer
architectures that incur computational costs for every input token,
regardless of semantic complexity or predictability.

This research project introduces a modular inference optimization
framework designed to reduce unnecessary computation in Transformer
decoder architectures along two orthogonal dimensions:

1. **Vertical inference optimization**
2. **Horizontal inference optimization**

The vertical dimension, referred to as **Adaptive Inference Routing
(AIR)**, analyzes intermediate Transformer representations and identifies
potential layer equivalences. Validated layer shortcuts can subsequently
be compiled through **Post-Training Operator Fusion**, allowing selected
inference paths to bypass multiple intermediate layers.

The horizontal dimension introduces **Asynchronous Multi-Token Chaining
(MTC)**. A numerical Token-ID prefix structure continuously records
high-confidence token transitions and can subsequently be used to
identify deterministic phrase patterns during inference.

The system additionally contains mechanisms for:

- Logit-Lens-based activation analysis
- Hidden-state and representation matching
- Mathematical early-abort filtering
- Shadow validation of candidate shortcuts
- Evolutionary / in-flight inference routing
- Model-specific persistent shortcut knowledge
- CPU-oriented inference benchmarking
- Token-identity verification

The current implementation provides experimental support for
**Meta Llama 3.2 1B** and **GPT-2**.

---

# 1. Research Objective

The central objective of this project is to investigate whether
Transformer inference can be dynamically optimized after model training
without modifying or fine-tuning the original model weights.

Instead of assuming that every input token requires the complete
computational depth of the model, AIR investigates whether portions of
the computation can be safely reused, bypassed, or compressed when
intermediate representations exhibit sufficient similarity.

The research therefore separates inference optimization into two
dimensions.

### Vertical Dimension

The vertical dimension operates across Transformer layers.

```text
Input
  │
  ▼
Layer 0
  │
  ▼
Layer 1
  │
  ▼
Layer 2
  │
  ▼
 ...
  │
  ▼
Layer N
````

AIR searches for situations where intermediate representations converge
sufficiently that parts of this path may potentially be skipped.

Conceptually:

```text
Layer 0
   │
   ├──────────────► Layer 8
   │
   └── Layer 1 ... Layer 7 ──► bypassed
```

Candidate shortcuts are subsequently subjected to validation before
being considered eligible for optimized inference.

---

# 2. Horizontal Dimension

The second dimension operates across the temporal sequence of tokens.

Standard autoregressive generation generally follows:

```text
Token 1
   │
   ▼
Token 2
   │
   ▼
Token 3
   │
   ▼
Token 4
   │
   ▼
...
```

The AIR/MTC concept investigates whether highly predictable token
sequences can be represented as reusable chains.

For example:

```text
Token A → Token B → Token C → Token D
```

can be represented through a numerical prefix structure.

The current implementation stores token IDs rather than complete text
strings.

Conceptually:

```text
prefix_token_1
    │
    └── prefix_token_2
            │
            ├── next_token_1
            ├── next_token_2
            └── next_token_3
```

This structure is referred to as the **Horizontal Token-ID Trie**.

---

# 3. Two-Dimensional AIR Architecture

The overall research architecture combines both dimensions:

```text
                    AIR Runtime
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
   Vertical Routing             Horizontal MTC
          │                           │
          ▼                           ▼
 Layer Equivalence             Token-ID Trie
          │                           │
          ▼                           ▼
 Operator Fusion               Phrase Chains
          │                           │
          └─────────────┬─────────────┘
                        │
                        ▼
              Evolutionary Runtime
                        │
                        ▼
                 Model Inference
```

The two mechanisms are complementary rather than mutually exclusive.

The vertical mechanism operates primarily across network depth, while
the horizontal mechanism operates across token sequence progression.

---

# 4. Main Components

The repository contains the following principal components.

## 4.1 Adaptive Inference Routing

AIR provides the infrastructure for identifying potential inference
shortcuts through analysis of intermediate model representations.

Candidate routes are stored in a persistent shortcut knowledge base.

A candidate route contains information such as:

```text
start layer
end layer
similarity / confidence
validation status
```

Candidate routes initially enter a pending state and can subsequently be
validated.

---

## 4.2 Logit-Lens Flow Analysis

The analyzer extracts information from intermediate Transformer
activations and evaluates their representational behavior.

The analysis is used to identify potential convergence between different
network depths.

The implementation is contained primarily in:

```text
analyzer.py
```

---

## 4.3 Post-Training Operator Fusion

Validated layer routes can be transformed into fused computational
operators.

The purpose is to avoid repeatedly executing a sequence of intermediate
operations when the learned AIR route determines that a shortcut is
eligible.

The repository contains separate implementations for the supported
architectures:

```text
compiler.py
compiler_llama.py
```

---

## 4.4 Horizontal Multi-Token Chaining

The horizontal component maintains a numerical Token-ID Trie.

A token sequence is analyzed during training / in-flight processing and
high-confidence transitions can be registered in the persistent
horizontal knowledge structure.

The relevant implementation is located in:

```text
shortcut_engine.py
```

The horizontal knowledge structure is persisted together with the
vertical shortcut map.

---

## 4.5 Shadow Validation

Candidate shortcuts are not automatically assumed to be correct.

The repository contains a shadow-validation mechanism that evaluates
candidate routes before they become fully validated.

Implementation:

```text
shadow_validator.py
```

The current configuration provides separate stability windows for
training and production:

```python
STABILITY_WINDOW_TRAINING = 1
STABILITY_WINDOW_PRODUCTION = 2
```

---

## 4.6 Evolutionary Inference Engine

The runtime component combines learned AIR knowledge with the inference
process.

Implementation:

```text
inference_onthefly_v2.py
```

The engine can select between optimized and standard inference paths
depending on the available shortcut knowledge and runtime conditions.

---

# 5. Current Configuration

Central parameters are defined in:

```text
config.py
```

The current experimental configuration is:

```python
GLOBAL_THRESHOLD = 0.92
EARLY_ABORT_THRESHOLD = 0.45

STABILITY_WINDOW_TRAINING = 1
STABILITY_WINDOW_PRODUCTION = 2
```

## GLOBAL_THRESHOLD

```python
GLOBAL_THRESHOLD = 0.92
```

This is the central convergence / confidence threshold used by the
current AIR implementation.

It is used for determining whether a representation similarity or
token-level confidence satisfies the configured threshold for the
corresponding mechanism.

## EARLY_ABORT_THRESHOLD

```python
EARLY_ABORT_THRESHOLD = 0.45
```

This parameter controls the early-abort mechanism used during vertical
layer analysis.

If the initial representational similarity falls below the configured
threshold, the vertical search can be terminated early.

This is intended to avoid unnecessary analysis of inference paths that
already exhibit insufficient convergence.

## Stability Windows

Training:

```python
STABILITY_WINDOW_TRAINING = 1
```

Production:

```python
STABILITY_WINDOW_PRODUCTION = 2
```

These parameters control the number of successful shadow-validation
observations required by the respective runtime configuration.

---

# 6. Supported Models

The current repository contains model backends for:

## Meta Llama 3.2 1B

Backend:

```text
model_base_llama.py
```

The implementation uses the Hugging Face Transformers architecture and
supports loading the model either from a local directory or through the
configured model identifier.

The current default model identifier is:

```text
unsloth/Llama-3.2-1B
```

---

## GPT-2

Backend:

```text
model_base.py
```

GPT-2 is included as an additional experimental backend and provides a
smaller environment for testing the AIR mechanisms.

---

# 7. Persistent Knowledge Base

AIR maintains a model-specific persistent knowledge base.

The current configuration defines:

```python
STORAGE_PATH_GPT2
STORAGE_PATH_LLAMA
```

These paths are generated dynamically from the location of `config.py`.

The stored structure contains two principal components:

```text
vertical
horizontal
```

Conceptually:

```python
{
    "vertical": {
        ...
    },

    "horizontal": {
        ...
    }
}
```

The vertical component contains learned layer shortcuts.

The horizontal component contains learned Token-ID prefix chains.

The knowledge base is stored using PyTorch serialization.

---

# 8. Training / Backbone Learning

The main backbone training script is:

```text
train_router_backbone_2_6.py
```

The training system maintains model-specific processing histories.

For example:

```text
processed_files_llama3.txt
processed_files_gpt2.txt
```

This allows the training process to distinguish the learning history of
different model backends.

The training process can operate in three modes:

```text
v1
v2
both
```

### v1

Focuses on the vertical inference optimization path:

```text
Activation Capture
       ↓
Flow Analysis
       ↓
Layer Equivalence
       ↓
Candidate Shortcuts
       ↓
Validation / Compilation
```

### v2

Focuses on the horizontal Token-ID chain mechanism:

```text
Input Text
    ↓
Tokenization
    ↓
Logit Analysis
    ↓
Confidence Evaluation
    ↓
Horizontal Trie Registration
```

### both

Combines the two mechanisms within the experimental AIR runtime.

---

# 9. Benchmarking

The repository includes dedicated benchmarking and accuracy-evaluation
scripts.

These experiments have different purposes and should not be interpreted
as identical measurements.

---

## 9.1 Llama / GPT2 Production Benchmark

Script:

```text
run_benchmarks_llama.py

and correspondingly for GPT-2: 

run_benchmarks.py

```

This benchmark compares two runtime configurations:

```text
SLOW-TRACK
```

against:

```text
AIR 2D-TRACK
```

The benchmark uses a standardized corpus of 50 test prompts.

The SLOW-TRACK disables the learned shortcut structures and measures the
baseline runtime.

The AIR 2D-TRACK restores the learned shortcut structures and evaluates
the optimized runtime.

The benchmark reports:

```text
Total Latency SLOW-TRACK
Total Latency AIR 2D-TRACK
Net Efficiency Increase
```

The purpose of this benchmark is to evaluate runtime performance under
the experimental configuration.

---

# 10. Accuracy and Token Identity Evaluation

Two additional scripts are provided.

## 10.1 Accuracy Benchmark

```text
run_accuracy_benchmark.py
```

This script compares the next-token prediction of:

```text
Standard / SLOW inference
```

against:

```text
AIR optimized inference
```

The benchmark measures whether the optimized path produces the same
predicted token as the reference path.

The primary reported metric is:

```text
Token Identity
```

This experiment is intended to evaluate whether shortcut activation
changes the model's predicted next token.

---

## 10.2 Pure AIR Knowledge Check

```text
run_accuracy_check.py
```

This script performs a more selective evaluation.

Only prompts for which the AIR Fast-Track is actually activated are
included in the fidelity calculation.

Standard SLOW-TRACK fallbacks are excluded.

The purpose is therefore different from the general accuracy benchmark.

It measures the fidelity of the shortcuts that actually activated rather
than the overall activation rate.

---

# 11. Experimental Results

The accompanying research paper reports an experimental evaluation of
the AIR framework using Meta Llama 3.2 1B on an x86 CPU environment.

The paper reports the following experimental comparison:

```text
Slow-Track baseline:       19,029.58 ms
Pure Vertical Compression: 19,042.45 ms
Integrated 2D-Track:       17,575.56 ms
```

The reported integrated 2D result corresponds to an experimental latency
reduction of approximately:

```text
-7.64%
```

These numbers are experimental observations from the research setup and
should not be interpreted as universal performance guarantees.

Actual results can vary depending on:

* CPU architecture
* memory bandwidth
* operating system
* Python version
* PyTorch version
* Transformers version
* model precision
* thermal conditions
* background processes
* model loading state
* local hardware configuration

For this reason, the benchmark scripts are included so that independent
experiments can be performed.

---

# 12. Research Paper

The scientific description of the architecture is provided in the
accompanying paper:

> **Two-Dimensional Inference Optimization in Deep Transformer
> Architectures: Dynamic-Adaptive Layer Routing, Post-Training Operator
> Fusion, Asynchronous Multi-Token Chaining, and Evolutionary Inference
> Validation**

The paper describes:

* the theoretical motivation
* the vertical routing mechanism
* post-training operator fusion
* asynchronous Multi-Token Chaining
* Token-ID Trie structures
* hidden-state matching
* mathematical early-abort filtering
* evolutionary inference validation
* CPU benchmarking
* experimental limitations
* potential future GPU investigations

The paper should be considered the primary scientific description of the
research architecture.

---

# 13. Repository Structure

The recommended repository structure is:

```text
Two-Dimensional-Inference-Optimization/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── paper/
│   └── two_dimensional_inference_optimization.pdf
│
├── config.py
├── shortcut_engine.py
├── analyzer.py
├── inference_onthefly_v2.py
├── model_base.py
├── model_base_llama.py
├── compiler.py
├── compiler_llama.py
|── shadow_validator.py
│
├── train_router_backbone_2_6_2.py
|── generate_local_training_data.py
│
├── run_benchmarks.py
├── run_benchmarks_llama.py
├── run_accuracy_benchmark.py
|── run_accuracy_check.py
│
|── app_onthefly.py
```

The exact organization may differ depending on the local development
environment.

---

# 14. Installation

Clone the repository:

```bash
git clone https://github.com/Omar-S-Nuri/Two-Dimensional-Inference-Optimization.git
```

Enter the repository:

```bash
cd Two-Dimensional-Inference-Optimization
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 15. Model Preparation

The model backend can load models from a local directory.

For Llama, the current implementation expects:

```text
./local_llama_free
```

if the model has already been downloaded.

If the directory is not available, the model backend can load the
configured model through Hugging Face Transformers and subsequently save
the model locally.

Model weights are intentionally **not included in this repository**.

---

# 16. Running the Training Pipeline

The backbone training system can be executed using:

```bash
python train_router_backbone_2_6_2.py both llama3 1000
```

The arguments are:

```text
argument 1 = mode
argument 2 = model
argument 3 = number of sentences
```

Examples:

```bash
python train_router_backbone_2_6_2.py v1 llama3 100
```

```bash
python train_router_backbone_2_6_2.py v2 llama3 100
```

```bash
python train_router_backbone_2_6_2.py both llama3 1000
```

For GPT-2:

```bash
python train_router_backbone_2_6_2.py both gpt2 1000
```

The actual available command depends on the repository layout and the
location of the training script.

---

# 17. Running the Llama Benchmark

After a knowledge base has been generated:

```bash
python run_benchmarks_llama.py
```

OR 

```bash
python run_benchmarks.py
```


The benchmark performs:

```text
1. Model initialization
2. AIR knowledge loading
3. Baseline measurement
4. AIR 2D measurement
5. Latency comparison
6. Efficiency calculation
```

---

# 18. Running the Accuracy Benchmark

Run:

```bash
python run_accuracy_benchmark.py
```

This evaluates next-token identity between the reference and optimized
inference paths.

---

# 19. Running the Pure Knowledge Check

Run:

```bash
python run_accuracy_check.py
```

This evaluates only prompts for which an AIR Fast-Track was activated.

---

# 20. Important Reproducibility Notes

The AIR project is an experimental research implementation.

The reported benchmark numbers are dependent on the exact experimental
environment.

For meaningful comparisons, record at least:

```text
CPU
RAM
Operating System
Python version
PyTorch version
Transformers version
Model version
Model precision
GLOBAL_THRESHOLD
EARLY_ABORT_THRESHOLD
Number of prompts
```

Benchmark runs should ideally be repeated multiple times.

A single timing run should not be interpreted as a statistically
complete performance characterization.

---

# 21. Generated Files

The following files may be generated during execution:

```text
shortcuts_llama.pt
shortcuts_gpt2.pt
processed_files_llama3.txt
processed_files_gpt2.txt
```

These files represent local experimental state.

They should generally **not** be committed to the public repository
unless a specific experiment requires them.

Large model directories should also not be committed.

For example:

```text
local_llama_free/
```

should remain outside the Git repository.

---

# 22. Security and Privacy

The runtime and training components can process user-provided text.

If the system is connected to a live application, users should ensure
that private or personally identifiable information is not unintentionally
included in experimental training corpora or persistent inference
knowledge bases.

The repository itself does not require user credentials or API keys for
the core experimental architecture.

Do not commit:

```text
API keys
passwords
access tokens
private datasets
personal user logs
private model credentials
local absolute paths
```

---

# 23. Scientific Scope

This repository is intended as a research prototype.

The project investigates whether learned inference shortcuts can reduce
unnecessary computation while preserving the output behavior of the
underlying model.

It does **not** claim that every Transformer architecture can achieve
the same performance improvement.

Likewise, the reported CPU measurements should not be interpreted as a
general guarantee of speedup across all hardware or model sizes.

The purpose of the repository is to provide an implementation through
which the proposed mechanisms can be inspected, tested, benchmarked and
further developed.

---

# 24. Limitations

The current implementation has several experimental limitations.

### Model Coverage

The current code primarily targets:

* Meta Llama 3.2 1B
* GPT-2

Additional Transformer architectures may require architecture-specific
backend and compiler implementations.

### Hardware Dependence

Performance is highly dependent on the underlying hardware and software
stack.

### Threshold Dependence

The behavior of shortcut discovery and activation depends on parameters
such as:

```python
GLOBAL_THRESHOLD
EARLY_ABORT_THRESHOLD
```

Changing these values can substantially alter the number of candidate
shortcuts and their activation behavior.

### Benchmark Size

The current benchmark uses a standardized 50-prompt evaluation corpus.

This is useful for controlled experiments but is not equivalent to a
large-scale production workload.

### Validation

The current validation system is designed as an experimental mechanism
and should not be interpreted as a formal proof of mathematical
equivalence between the original and optimized networks.

---

# 25. Future Research

Potential future directions include:

* larger language models
* broader Transformer architectures
* longer horizontal token chains
* adaptive confidence thresholds
* improved entropy estimation
* more extensive hidden-state matching
* larger evaluation corpora
* statistical benchmark analysis
* GPU evaluation
* HBM-oriented optimization
* kernel-level optimization
* multi-token speculative verification
* more sophisticated shortcut invalidation
* distributed inference experiments

The GPU performance values discussed in the paper are projections /
research hypotheses and require dedicated hardware experiments for
empirical confirmation.

---

# 26. Terminology

| Term               | Meaning                                                               |
| ------------------ | --------------------------------------------------------------------- |
| AIR                | Adaptive Inference Routing                                            |
| 2D Runtime         | Combined vertical and horizontal inference optimization               |
| Vertical Routing   | Optimization across Transformer depth                                 |
| Horizontal MTC     | Optimization across token sequence progression                        |
| MTC                | Asynchronous Multi-Token Chaining                                     |
| Token-ID Trie      | Numerical prefix structure for token sequences                        |
| Operator Fusion    | Combination of computational operators into fused structures          |
| Shadow Validation  | Validation of candidate shortcuts without immediately relying on them |
| Early-Abort Filter | Mechanism for terminating unsuitable shortcut searches                |
| Fast-Track         | Optimized inference route                                             |
| Slow-Track         | Reference / standard inference route                                  |
| Token Fidelity     | Agreement between optimized and reference token prediction            |

---

# 27. Author

**Omar Nuri**

Independent Researcher
Computer Science and Machine Learning Systems

GitHub:

[https://github.com/Omar-S-Nuri/Two-Dimensional-Inference-Optimization](https://github.com/Omar-S-Nuri/Two-Dimensional-Inference-Optimization)

---

# 28. License

This repository is distributed under the **MIT License**.

See:

```text
LICENSE
```

for the complete license text.

The license applies to the original source code contained in this
repository.

Third-party software, pretrained models, datasets and dependencies remain
subject to their respective licenses and terms.

---

# 29. Research Status

**Status: Experimental Research Prototype**

The implementation is actively developed and may change as the
underlying research evolves.

The repository is provided for:

* research
* experimentation
* reproducibility
* benchmarking
* inspection
* further development

rather than as a production-ready inference framework.


# 30. Citation

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

## 31. Disclaimer

This repository contains experimental research software.

The implementation is provided for research and evaluation purposes.
No guarantee is made that the optimization mechanisms preserve model
behavior or improve inference performance under all configurations.

Performance results reported in the accompanying paper are tied to the
specific experimental conditions under which they were obtained. PLEASE NOTE: Without any guarantee.

---

## Repository

**Two-Dimensional Inference Optimization**

[https://github.com/Omar-S-Nuri/Two-Dimensional-Inference-Optimization](https://github.com/Omar-S-Nuri/Two-Dimensional-Inference-Optimization)

