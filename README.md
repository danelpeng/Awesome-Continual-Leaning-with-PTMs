# 🤖 Awesome-Continual-Learning-with-PTMs

> This is a curated list of "Continual Learning with Pretrained Models" research which is maintained by [danelpeng](https://github.com/danelpeng).

![](https://github.com/danelpeng/Awesome-Continual-Leaning-with-PTMs/blob/main/lifelong-learning.png)

## News 🔥

[2024/10/25] Updated with latest papers.

[2024/10/08] Created this repo.

## Table of Contents 📰

- [Survey](#Survey)
- [Prompt Based](#Prompt-Based)
- [Adapter Based ](#Adapter-Based)
- [LoRA Based ](#LoRA-Based)
- [MoE&Ensemble Based ](#MoE-&-Ensemble-Based)
- [VLM Based ](#VLM-Based)
- [Application ](#Application)


## Methods 🌟

> ### Survey


* [**Recent Advances of Multimodal Continual Learning: A Comprehensive Survey**](https://arxiv.org/pdf/2410.05352) [**Arxiv 2024.10**]
The Chinese University of Hong Kong, Tsinghua University, University of Illinois Chicago


* [**Continual Learning with Pre-Trained Models: A Survey**](https://arxiv.org/pdf/2401.16386) [**IJCAI 2024**]
Nanjing University


> ### Prompt Based 


* [**Replay-and-Forget-Free Graph Class-Incremental Learning: A Task Profiling and Prompting Approach**](https://arxiv.org/pdf/2410.10341) [**NeurIPS 2024**]
University of Technology Sydney, Singapore Management University, University of Illinois at Chicago


* [**ModalPrompt:Dual-Modality Guided Prompt for Continual Learning of Large Multimodal Models**](https://arxiv.org/pdf/2410.05849) [**Arxiv 2024.10**]
Institute of Automation, CAS


* [**Leveraging Hierarchical Taxonomies in Prompt-based Continual Learning**](https://arxiv.org/pdf/2410.04327) [**Arxiv 2024.10**]
VinAI Research, Monash University, Hanoi University of Science and Technolgy, Univesity of Oregon, The University of Texas at Austin


* [**LW2G: Learning Whether to Grow for Prompt-based Continual Learning**](https://arxiv.org/pdf/2409.18860) [**Arxiv 2024.09**]
Zhejiang University, Nanjing University


* [**Mind the Interference: Retaining Pre-trained Knowledge in Parameter Efficient Continual Learning of Vision-Language Models**](https://arxiv.org/pdf/2407.05342) [**ECCV 2024**]
Tsinghua University, SmartMore, CUHK, HIT(SZ), Meta Reality Labs, HKU


* [**Evolving Parameterized Prompt Memory for Continual Learning**](https://ojs.aaai.org/index.php/AAAI/article/view/29231/30323) [**AAAI 2024**]
Xi'an Jiaotong University


* [**Generating Prompts in Latent Space for Rehearsal-free Continual Learning**](https://openreview.net/pdf?id=6HT4jUkSRg) [**ACMMM 2024**]
East China Normal University


* [**Convolutional Prompting meets Language Models for Continual Learning**](https://openaccess.thecvf.com/content/CVPR2024/papers/Roy_Convolutional_Prompting_meets_Language_Models_for_Continual_Learning_CVPR_2024_paper.pdf) [**CVPR 2024**]
IIT Kharagpur, IML Amazon India


* [**Consistent Prompting for Rehearsal-Free Continual Learning**](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_Consistent_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2024_paper.pdf) [**CVPR 2024**]
Sun Yat-sen University, HKUST


* [**Steering Prototypes with Prompt-tuning for Rehearsal-free Continual Learning**](https://openaccess.thecvf.com/content/WACV2024/papers/Li_Steering_Prototypes_With_Prompt-Tuning_for_Rehearsal-Free_Continual_Learning_WACV_2024_paper.pdf) [**WACV 2024**]
  Rutgers University, Google Research, Google Cloud AI
  
* [**Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality**](https://proceedings.neurips.cc/paper_files/paper/2023/file/d9f8b5abc8e0926539ecbb492af7b2f1-Paper-Conference.pdf) [**NeurIPS 2023**]
Tsinghua-Bosch Joint Center for ML, Tsinghua University


* [**When Prompt-based Incremental Learning Does Not Meet Strong Pretraining**](https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_When_Prompt-based_Incremental_Learning_Does_Not_Meet_Strong_Pretraining_ICCV_2023_paper.pdf) [**ICCV 2023**]
Sun Yat-sen University, Peng Cheng Laboratory


* [**Introducing Language Guidance in Prompt-based Continual Learning**](https://openaccess.thecvf.com/content/ICCV2023/papers/Khan_Introducing_Language_Guidance_in_Prompt-based_Continual_Learning_ICCV_2023_paper.pdf) [**ICCV 2023**]
RPTU, DFKI, ETH Zurich, TUM, Google


* [**MoP-CLIP: A Mixture of Prompt-Tuned CLIP Models for Domain Incremental Learning**](https://arxiv.org/pdf/2307.05707) [**Arxiv 2023.07**]
ETS Montreal


* [**Progressive Prompts: Continual Learning for Language Models**](https://arxiv.org/pdf/2301.12314) [**ICLR 2023**]
University of Toronto & Vector Institute, Meta AI


* [**Online Class Incremental Learning on Stochastic Blurry Task Boundary via Mask and Visual Prompt Tuning**](https://openaccess.thecvf.com/content/ICCV2023/papers/Moon_Online_Class_Incremental_Learning_on_Stochastic_Blurry_Task_Boundary_via_ICCV_2023_paper.pdf) [**ICCV 2023**]
Kyung Hee University


* [**Self-regulating Prompts: Foundational Model Adaptation without Forgetting**](https://openaccess.thecvf.com/content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf) [**ICCV 2023**]
Mohamed bin Zayed University of AI, Australian National University, Linkoping University, University of California, Merced, Google Research


* [**Generating Instance-level Prompts for Rehearsal-free Continual Learning**](https://openaccess.thecvf.com/content/ICCV2023/html/Jung_Generating_Instance-level_Prompts_for_Rehearsal-free_Continual_Learning_ICCV_2023_paper.html) [**ICCV 2023 (oral)**]
Seoul National University, NAVER AI Lab, NAVER Cloud, AWS AI Labs


* [**CODA-Prompt: COntinual Decomposed Attention-Based Prompting for Rehearsal-Free Continual Learning**](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) [**CVPR 2023**]
Georgia Institute of Technology, MIT-IBM Watson AI Lab, Rice University, IBM Research


* [**S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning**](https://proceedings.neurips.cc/paper_files/paper/2022/file/25886d7a7cf4e33fd44072a0cd81bf30-Paper-Conference.pdf) [**NeurIPS 2022**]
Xi’an Jiaotong University


* [**DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning**](https://arxiv.org/pdf/2204.04799) [**ECCV 2022**]
Northeastern University, Google Cloud AI, Google Research


* [**Learning to Prompt for Continual Learning**](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf) [**CVPR 2022**]
Northeastern University, Google Cloud AI, Google Research


> ### Adapter Based 


* [**ATLAS: Adapter-Based Multi-Modal Continual Learning with a Two-Stage Learning Strategy**](https://arxiv.org/abs/2410.10923) [**Arxiv 2024.10**]
Shanghai Jiao Tong University,  ShanghaiTech University, Tsinghua University


* [**Adaptive Adapter Routing for Long-Tailed Class-Incremental Learning**](https://arxiv.org/pdf/2409.07446) [**Arxiv 2024.09**]
Nanjing University


* [**Learning to Route for Dynamic Adapter Composition in Continual Learning with Language Models**](https://arxiv.org/pdf/2408.09053) [**Arxiv 2024.08**]
KU Leuven


* [**Expand and Merge: Continual Learning with the Guidance of Fixed Text Embedding Space**](https://ieeexplore.ieee.org/abstract/document/10650910) [**IJCNN 2024**]
Sun Yat-sen University


* [**Beyond Prompt Learning: Continual Adapter for Efficient Rehearsal-Free Continual Learning**](https://arxiv.org/pdf/2407.10281) [**ECCV 2024**]
Xi’an Jiaotong University


* [**Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer**](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_Semantically-Shifted_Incremental_Adapter-Tuning_is_A_Continual_ViTransformer_CVPR_2024_paper.pdf) [**CVPR 2024**]
Huazhong University of Science and Tech., DAMO Academy, Alibaba Group


* [**Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning**](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Expandable_Subspace_Ensemble_for_Pre-Trained_Model-Based_Class-Incremental_Learning_CVPR_2024_paper.pdf) [**CVPR 2024**]
Nanjing University


> ### LoRA Based 


* [**InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning**](https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.pdf) [**CVPR 2024**]
Nanjing University


* [**Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation**](https://openreview.net/pdf?id=X7OKRr09OS) [**NeurIPSW 2024**]
University of Texas at Austin


* [**Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters**](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.pdf) [**CVPR 2024**]
Dalian University of Technology, UESTC, Tsinghua University


> ### MoE&Ensemble Based 


* [**Learning Attentional Mixture of LoRAs for Language Model Continual Learning**](https://arxiv.org/pdf/2409.19611) [**Arxiv 2024.09**]
Nankai University


* [**Theory on Mixture-of-Experts in Continual Learning**](https://arxiv.org/pdf/2406.16437) [**Arxiv 2024.10**]
Singapore University of Technology and Design, University of Houston, The Ohio State University


* [**Weighted Ensemble Models Are Strong Continual Learners**](https://arxiv.org/abs/2312.08977) [**ECCV 2024**]
Télécom-Paris, Institut Polytechnique de Paris


* [**LEMoE: Advanced Mixture of Experts Adaptor for Lifelong Model Editing of Large Language Models**](https://arxiv.org/pdf/2406.20030) [**Arxiv 2024.06**]
Nanjing University of Aeronautics and Astronautics


* [**Mixture of Experts Meets Prompt-Based Continual Learning**](https://arxiv.org/pdf/2405.14124) [**Arxiv 2024.05**]
The University of Texas at Austin, Hanoi University of Science and Technology, VinAI Research


* [**Learning More Generalized Experts by Merging Experts in Mixture-of-Experts**](https://arxiv.org/pdf/2405.11530) [**Arxiv 2024.05**]
KAIST


* [**MoRAL: MoE Augmented LoRA for LLMs’ Lifelong Learning**](https://arxiv.org/pdf/2402.11260) [**Arxiv 2024.02**]
Provable Responsible AI and Data Analytics (PRADA) Lab, KAUST, University of Macau


* [**Divide and not forget: Ensemble of selectively trained experts in Continual Learning**](https://arxiv.org/pdf/2401.10191) [**ICLR 2024**]
IDEAS-NCBR, Warsaw University of Technology


* [**Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters**](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.pdf) [**CVPR 2024**]
Dalian University of Technology, UESTC, Tsinghua University


* [**An Efficient General-Purpose Modular Vision Model via Multi-Task Heterogeneous Training**](https://arxiv.org/pdf/2306.17165) [**Arxiv 2023.06**]
University of Massachusetts Amherst, University of California Berkeley, MIT-IBM Watson AI Lab


* [**Lifelong Language Pretraining with Distribution-Specialized Experts**](https://proceedings.mlr.press/v202/chen23aq/chen23aq.pdf) [**ICML 2023**]
The University of Texas at Austin, Google


* [**Continual Learning Beyond a Single Model**](https://proceedings.mlr.press/v232/doan23a/doan23a.pdf) [**CoLLAs 2023**]
Bosch Center for Artificial Intelligence, Washington State University, Apple


* [**Mixture-of-Variational-Experts for Continual Learning**](https://arxiv.org/pdf/2009.04381) [**Arxiv 2022.03**]
Ulm University


* [**CoSCL: Cooperation of Small Continual Learners is Stronger Than a Big One**](https://arxiv.org/pdf/2207.06543) [**ECCV 2022**]
Tsinghua University


* [**Ex-Model: Continual Learning from a Stream of Trained Models**](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Carta_Ex-Model_Continual_Learning_From_a_Stream_of_Trained_Models_CVPRW_2022_paper.pdf) [**CVPRW 2022**]
University of Pisa


* [**Routing Networks with Co-training for Continual Learning**](https://arxiv.org/pdf/2009.04381) [**ICMLW 2020**]
Google AI, Zurich


* [**A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning**](https://arxiv.org/pdf/2001.00689) [**ICLR 2020**]
Seoul National University


> ### VLM Based 


* [**Continual learning with task specialist**](https://arxiv.org/pdf/2409.17806) [**Arxiv 2024.09**]
International Institute of Information Technology Bangalore, A*STAR


* [**A Practitioner’s Guide to Continual Multimodal Pretraining**](https://arxiv.org/abs/2408.14471) [**Arxiv 2024.08**]
University of T¨ubingen, Helmholtz Munich, Munich Center for ML, Google DeepMind


* [**CLIP with Generative Latent Replay: a Strong Baseline for Incremental Learning**](https://arxiv.org/abs/2407.15793) [**BMVC 2024**]
University of Modena and Reggio


* [**Mind the Interference: Retaining Pre-trained Knowledge in Parameter Efficient Continual Learning of Vision-Language Models**](https://arxiv.org/pdf/2407.05342) [**ECCV 2024**]
Tsinghua University, SmartMore, CUHK, HIT(SZ), Meta Reality Labs, HKU


* [**Anytime Continual Learning for Open Vocabulary Classification**](https://arxiv.org/abs/2409.08518) [**ECCV 2024 (oral)**]
University of Illinois at Urbana-Champaign


* [**Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models**](https://arxiv.org/pdf/2403.09296) [**ECCV 2024**]
National Taiwan University, NVIDIA


* [**Expand and Merge: Continual Learning with the Guidance of Fixed Text Embedding Space**](https://ieeexplore.ieee.org/abstract/document/10650910) [**IJCNN 2024**]
Sun Yat-sen University


* [**CoLeCLIP: Open-Domain Continual Learning via Joint Task Prompt and Vocabulary Learning**](https://arxiv.org/pdf/2403.10245) [**Arxiv 2024.05**]
Northwestern Polytechnical University, Singapore Management University, Zhejiang University, University of Adelaide


* [**TiC-CLIP: Continual Training of CLIP Models**](https://arxiv.org/pdf/2310.16226) [**ICLR 2024**]
Apple, Carnegie Mellon University


* [**Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters**](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.pdf) [**CVPR 2024**]
Dalian University of Technology, UESTC, Tsinghua University


* [**Pre-trained Vision and Language Transformers Are Few-Shot Incremental Learners**](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Pre-trained_Vision_and_Language_Transformers_Are_Few-Shot_Incremental_Learners_CVPR_2024_paper.pdf) [**CVPR 2024**]
Kyung Hee University, Yonsei University


* [**MoP-CLIP: A Mixture of Prompt-Tuned CLIP Models for Domain Incremental Learning**](https://arxiv.org/pdf/2307.05707) [**Arxiv 2023.07**]
ETS Montreal


* [**Learning without Forgetting for Vision-Language Models**](https://arxiv.org/pdf/2305.19270) [**Arxiv 2023.05**]
Nanjing University, Nanyang Technological University


* [**Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models**](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Preventing_Zero-Shot_Transfer_Degradation_in_Continual_Learning_of_Vision-Language_Models_ICCV_2023_paper.pdf) [**ICCV 2023**]
National University of Singapore, UC Berkeley, The Chinese University of Hong Kong


* [**Self-regulating Prompts: Foundational Model Adaptation without Forgetting**](https://openaccess.thecvf.com/content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf) [**ICCV 2023**]
Mohamed bin Zayed University of AI, Australian National University, Linkoping University, University of California, Merced, Google Research


* [**Continual Vision-Language Representation Learning with Off-Diagonal Information**](https://proceedings.mlr.press/v202/ni23c/ni23c.pdf) [**ICML 2023**]
Zhejiang University, Huawei Cloud


* [**CLIP model is an Efficient Continual Learner**](https://arxiv.org/pdf/2210.03114) [**Arxiv 2022.10**]
Mohamed bin Zayed University of Artificial Intelligence, Australian National University, Monash University, Linkoping University


* [**Don’t Stop Learning: Towards Continual Learning for the CLIP Model**](https://arxiv.org/pdf/2207.09248) [**Arxiv 2022.07**]
Xidian University, University of Adelaide


* [**CLiMB: A Continual Learning Benchmark for Vision-and-Language Tasks**](https://proceedings.neurips.cc/paper_files/paper/2022/file/bd3611971089d466ab4ca96a20f7ab13-Paper-Datasets_and_Benchmarks.pdf) [**NeurIPS 2022**]
University of Southern California,


* [**S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning**](https://proceedings.neurips.cc/paper_files/paper/2022/file/25886d7a7cf4e33fd44072a0cd81bf30-Paper-Conference.pdf) [**NeurIPS 2022**]
Xi’an Jiaotong University


* [**Robust Fine-Tuning of Zero-Shot Models**](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf) [**CVPR 2022**]
University of Washington, OpenAI, Columbia University, Google Research, Brain Team


> ### Diffusion Based 


* [**Continual learning with task specialist**](https://arxiv.org/pdf/2409.17806) [**Arxiv 2024.09**]
International Institute of Information Technology Bangalore, A*STAR


* [**Diffusion Model Meets Non-Exemplar Class-Incremental Learning and Beyond**](https://arxiv.org/pdf/2409.01128) [**Arxiv 2024.08**]
BNRist, Tsinghua University


* [**Class-Prototype Conditional Diffusion Model with Gradient Projection for Continual Learning**](https://arxiv.org/pdf/2312.06710) [**Arxiv 2024.03**]
VinAI Research, Monash University


* [**Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning**](https://arxiv.org/abs/2408.02983) [**ECCV 2024 (oral)**]
South China University of Technology, HKUST, China University of Petroleum, WeBank, Pazhou Laboratory


* [**DiffClass: Diffusion-Based Class Incremental Learning**](https://arxiv.org/pdf/2403.05016) [**ECCV 2024**]
Northeastern University, ETH Zürich


* [**GUIDE: Guidance-based Incremental Learning with Diffusion Models**](https://arxiv.org/pdf/2403.03938) [**Arxiv 2024.03**]
Warsaw University of Technology


* [**SDDGR: Stable Diffusion-based Deep Generative Replay for Class Incremental Object Detection**](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_SDDGR_Stable_Diffusion-based_Deep_Generative_Replay_for_Class_Incremental_Object_CVPR_2024_paper.pdf) [**CVPR 2024**]
UNIST, LG Electronics, KETI


* [**Class-Incremental Learning using Diffusion Model for Distillation and Replay**](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Jodelet_Class-Incremental_Learning_Using_Diffusion_Model_for_Distillation_and_Replay_ICCVW_2023_paper.pdf) [**ICCVW 2023**]
Tokyo Institute of Technology, Artificial Intelligence Research Center


* [**DDGR: Continual Learning with Deep Diffusion-based Generative Replay**](https://proceedings.mlr.press/v202/gao23e/gao23e.pdf) [**ICML 2023**]
Wuhan University


> ### Representation Based


* [**Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning**](https://arxiv.org/pdf/2410.00911) [**Arxiv 2024.10**]
Nanjing University


> ### Application


> #### Embodied AI


* [**Incremental Learning for Robot Shared Autonomy**](https://arxiv.org/pdf/2410.06315) [**Arxiv 2024.10**]
Robotics Institute, Carnegie Mellon University


* [**Task-unaware Lifelong Robot Learning with Retrieval-based Weighted Local Adaptation**](https://arxiv.org/pdf/2410.02995) [**Arxiv 2024.10**]
TU Delft, Booking.com, UCSD


* [**Vision-Language Navigation with Continual Learning**](https://arxiv.org/pdf/2409.02561) [**Arxiv 2024.09**]
Institute of Automation, Chinese Academy of Science


* [**Continual Vision-and-Language Navigation**](https://arxiv.org/pdf/2403.15049) [**Arxiv 2024.03**]
Seoul National University


* [**Online Continual Learning For Interactive Instruction Following Agents**](https://arxiv.org/pdf/2403.07548) [**ICLR 2024**]
Yonsei University, Seoul National University


> #### LLM


* [**LLaCA: Multimodal Large Language Continual Assistant**](https://arxiv.org/pdf/2410.10868) [**Arxiv 2024.10**]
East China Normal University, Xiamen University, Tencent YouTu Lab


* [**Is Parameter Collision Hindering Continual Learning in LLMs?**](https://arxiv.org/pdf/2410.10179) [**Arxiv 2024.10**]
Peking University, DAMO Academy


* [**ModalPrompt:Dual-Modality Guided Prompt for Continual Learning of Large Multimodal Models**](https://arxiv.org/pdf/2410.05849) [**Arxiv 2024.10**]
Institute of Automation, CAS


* [**Learning Attentional Mixture of LoRAs for Language Model Continual Learning**](https://arxiv.org/pdf/2409.19611) [**Arxiv 2024.09**]
Nankai University


* [**Empowering Large Language Model for Continual Video Question Answering with Collaborative Prompting**](https://arxiv.org/pdf/2410.00771) [**EMNLP 2024**]
Nanyang Technological University


> #### Diffusion


* [**Low-Rank Continual Personalization of Diffusion Models**](https://arxiv.org/pdf/2410.04891) [**Arxiv 2024.10**]
Warsaw University of Technology


* [**Continual Diffusion with STAMINA: STack-And-Mask INcremental Adapters**](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/papers/Smith_Continual_Diffusion_with_STAMINA_STack-And-Mask_INcremental_Adapters_CVPRW_2024_paper.pdf) [**CVPRW 2024**]
Samsung Research America, Georgia Institute of Technology


* [**Continual Diffusion: Continual Customization of Text-to-Image Diffusion with C-LoRA**](https://arxiv.org/pdf/2304.06027) [**TMLR 2024**]
Samsung Research America, Georgia Institute of Technology


* [**Continual Learning of Diffusion Models with Generative Distillation**](https://arxiv.org/pdf/2311.14028) [**CoLLAs 2024**]
Master in Computer Vision (Barcelona), Apple, KU Leuven


> #### DeepFake


* [**Conditioned Prompt-Optimization for Continual Deepfake Detection**](https://arxiv.org/pdf/2407.21554) [**ICPR 2024**]
University of Trento, Fondazione Bruno Kessler


* [**A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials**](https://openaccess.thecvf.com/content/WACV2023/papers/Li_A_Continual_Deepfake_Detection_Benchmark_Dataset_Methods_and_Essentials_WACV_2023_paper.pdf) [**WACV 2023**]
ETH Zurich, Singapore Management University, Xi’an Jiaotong University, Harbin Institute of Technology, KU Leuven



## Acknowledge 👨‍🏫
* Figure from this URL: [Lifelong learning? Part-time undergraduate provision is in crisis](https://world.edu/lifelong-learning-part-time-undergraduate-provision-crisis/).
