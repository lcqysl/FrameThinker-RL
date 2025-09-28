# FrameThinker

This is the official repository for the core code of the paper: **FrameThinker: Learning to Think with Long Videos via Multi-Turn Frame Spotlighting**.

## 📖 About The Project

FrameThinker is a novel framework for long-video reasoning that challenges the inefficient, passive methods of traditional models. Instead of processing a fixed set of pre-sampled frames, FrameThinker **actively** interrogates video content through a **multi-turn, iterative process**. It intelligently spotlights relevant frame sequences to gather evidence, guided by a Cognitive Consistency Verification (CCV) module that ensures its reasoning is logical and interpretable. Across six challenging benchmarks, FrameThinker achieves an **average +10.4% accuracy improvement** over the baseline. As a highlight, it surpasses the strong LongVILA-R1 to set a new state-of-the-art on the LongVideo-Reason benchmark, using just 20.6 frames on average.

<p align="center">
  <img src="assets/Flow Chart.png" width="800" alt="FrameThinker Framework">
  <br>
  <em>An illustration of the FrameThinker framework.</em>
</p>

## 🚀 Getting Started

### Prerequisites

*   Python==3.10
*   vllm==0.9.1
*   transformers==4.52.4
*   Other dependencies listed in `requirements.txt`

### ⚙️ Training

```bash
bash examples/agent/train_frame_thinker.sh
```

### 🚀 Inference & Evaluation

```bash
python examples/agent/infer.py 
```

### 🙏 Acknowledgements

We would like to express our sincere gratitude to the open-source community and the creators of the foundational projects that made this work possible.

Our implementation is built upon the excellent codebases of **[verl](https://github.com/volcengine/verl)** and **[DeepEyes](https://github.com/Visual-Agent/DeepEyes)**. Their work provided a strong foundation and significantly accelerated our research. We highly recommend checking out their projects.