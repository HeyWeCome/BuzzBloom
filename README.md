### BuzzBloom 🌼🐝

BuzzBloom is the flexible and useful project for information diffusion prediction. I hope this project will help the community to harness the power of information diffusion and its potential to shape a more connected and engaging digital landscape. Join us in revolutionizing the way information spreads—we welcome passionate contributors to submit their PRs and help us grow this exciting journey! 🌟✨

**Currently integrated models**:

| **Model** | **Title**                                                                                                                            | **Link**                                                    | **Publication** |
|---------|--------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|-----------------|
| DyHGCN  | DyHGCN: A Dynamic Heterogeneous Graph Convolutional Network to Learn Users' Dynamic Preferences for Information Diffusion Prediction | [Paper](https://dl.acm.org/doi/10.1007/978-3-030-67664-3_21) | ECML PKDD 2020  |
| MS-HGAT | MS-HGAT: Memory-Enhanced Sequential Hypergraph Attention Network for Information Diffusion Prediction                                | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20334) | AAAI-22         |
| DisenIDP | Enhancing Information Diffusion Prediction with Self-Supervised Disentangled User and Cascade Representations                        | [Paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615230) | CIKM-23         |
| Buzz    | An Information Diffusion Prediction Model Aligning Multiple Propagation Intentions with Dynamic User Cognition |  | TCSS-25         |
| MIM     | Disentangling Inter- and Intra-Cascades Dynamics for Information Diffusion Prediction                                                | [Paper](https://doi.org/10.1109/TKDE.2025.3568289) | TKDE-25         |
| PMRCA   | A Pattern-Driven Information Diffusion Prediction Model Based on Multisource Resonance and Cognitive Adaptation                      | [Paper](https://doi.org/10.1145/3726302.3729883) | SIGIR-25        |


## ✨ Updates ✨

**🚀 Performance & Modernization Update (2025-07-01)**

This project has been supercharged with the latest features from **PyTorch 2.x** for a massive performance boost! ⚡ We've integrated `torch.compile` to JIT-compile the model, slashing Python overhead and accelerating GPU computations. Additionally, we've enabled **TensorFloat32 (TF32)** to leverage the power of modern NVIDIA GPUs, offering near-FP16 speed with FP32's stability. The data loading pipeline was also refactored using PyTorch's native `DataLoader`, enabling parallel data pre-fetching to keep your GPU fully utilized. Enjoy the new speed! 💨