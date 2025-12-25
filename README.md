<div align="center">

<h1>UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement</h1>

<a href="https://arxiv.org/pdf/2512.21185"><img src="https://img.shields.io/badge/arXiv-2512.21185-b31b1b.svg?style=flat-square" alt="arXiv"></a>
<a href="https://pku-yuangroup.github.io/UltraShape-1.0/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page"></a>
<!-- <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=flat-square" alt="HuggingFace Models"></a> -->
<!-- <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a> -->

</div>

<br/>

<div align="center">
  <img src="assets/images/teaser.png" width="100%" alt="UltraShape 1.0 Teaser" />
</div>

<br/>

## 📖 Abstract

In this report, we introduce **UltraShape 1.0**, a scalable 3D diffusion framework for high-fidelity 3D geometry generation. The proposed approach adopts a **two-stage generation pipeline**: a coarse global structure is first synthesized and then refined to produce detailed, high-quality geometry.

To support reliable 3D generation, we develop a comprehensive data processing pipeline that includes a novel **watertight processing method** and **high-quality data filtering**. This pipeline improves the geometric quality of publicly available 3D datasets by removing low-quality samples, filling holes, and thickening thin structures, while preserving fine-grained geometric details. 

To enable fine-grained geometry refinement, we decouple spatial localization from geometric detail synthesis in the diffusion process. We achieve this by performing **voxel-based refinement** at fixed spatial locations, where voxel queries derived from coarse geometry provide explicit positional anchors encoded via **RoPE**, allowing the diffusion model to focus on synthesizing local geometric details within a reduced, structured solution space.

Extensive evaluations demonstrate that UltraShape 1.0 performs competitively with existing open-source methods in both data processing quality and geometry generation.

## 🔥 News
* **[2025-12-25]** 📄 We released the technical report of **UltraShape 1.0** on arXiv.
* **[Coming Soon]** 🚀 Training code and pre-trained models will be released soon. Stay tuned!

## 💡 Key Features

### 1. Scalable Two-Stage Generation
We adopt a coarse-to-fine strategy. The model first synthesizes a global structure and then performs voxel-based refinement to hallucinate high-frequency geometric details.

### 2. Robust Data Processing Pipeline
We train exclusively on publicly available 3D datasets. Our novel pipeline includes:
* **Watertight Processing:** Effectively fills holes and repairs non-manifold geometry.
* **Quality Filtering:** Automatically removes low-quality samples to ensure high training stability.

### 3. Decoupled Spatial Refinement
We introduce a mechanism to decouple spatial localization from detail synthesis. By using voxel queries with **RoPE (Rotary Positional Embeddings)** as explicit positional anchors, the model can focus purely on local geometric details within a structured solution space.

## 🗓️ To-Do List
- [ ] Release inference code
- [ ] Release pre-trained weights (Hugging Face)
- [ ] Release training code
- [ ] Release data processing scripts

## 🛠️ Installation & Usage

*(Code coming soon. The following is a placeholder for future updates)*

```bash
git clone https://github.com/PKU-YuanGroup/UltraShape-1.0.git
cd UltraShape
pip install -r requirements.txt
```

## Acknowledgements

- **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** (Open Source)
- **[Lattice3D](https://lattice3d.github.io/)**
