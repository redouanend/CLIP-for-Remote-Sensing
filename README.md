# 🛰️ CLIP for Remote Sensing

**Adapting a vision-language model to satellite imagery: zero-shot vs linear probe vs LoRA fine-tuning.**

> 🚧 Work in progress. This README is updated at each step of the project.

## Motivation

Vision-language models like [CLIP](https://openai.com/index/clip/) are trained on hundreds of millions of image-text pairs from the internet. Those images are mostly ground-level photos: people, animals, objects.

Satellite images are very different: they are seen from above, have low resolution, and are defined by textures and colors rather than objects. This **domain shift** makes general models perform poorly on Earth observation tasks.

In practice, nobody trains a large model from scratch for each domain. We **adapt** an existing foundation model. This project answers a practical question:

> **Which adaptation method gives the best trade-off between accuracy, number of trainable parameters, and amount of labeled data?**

## Methods compared

| Method | What is trained | Idea |
|---|---|---|
| **Zero-shot** | Nothing | Classify images by comparing them with text prompts ("a satellite photo of a forest") |
| **Linear probe** | A single linear layer | Freeze CLIP, learn a classifier on top of its image features |
| **LoRA** | Small low-rank matrices inside the attention layers | Adapt the model while training ~1% of the parameters |
| **Full fine-tuning** | All the weights | Upper bound, but expensive |

## Datasets

- **[EuroSAT (RGB)](https://github.com/phelber/EuroSAT)**: 27,000 Sentinel-2 images (64×64 px), 10 land-use classes. Used for classification.
- **[RSICD](https://github.com/201528014227051/RSICD_optimal)**: ~10,900 aerial images with 5 captions each. Used for text→image retrieval.

## Roadmap

- [ ] Data exploration and fixed train / val / test split
- [ ] Zero-shot baseline (naive prompts vs remote-sensing prompts)
- [ ] Linear probe + few-shot curve (1, 5, 10, 50 images per class, then all)
- [ ] LoRA vs full fine-tuning
- [ ] Text→image retrieval on RSICD (Recall@1/5/10)
- [ ] Error analysis and ablations (LoRA rank, target layers, data size)
- [ ] Demo

## Results

*Coming soon.* All methods are evaluated on the same held-out test set.

| Method | Trainable params | Test accuracy (EuroSAT) |
|---|---|---|
| Zero-shot (naive prompts) | 0 | – |
| Zero-shot (remote-sensing prompts) | 0 | – |
| Linear probe | – | – |
| LoRA | – | – |
| Full fine-tuning | – | – |

## Repository structure

```
clip-remote-sensing/
├── README.md
├── requirements.txt
├── notebooks/        # one notebook per step
├── src/              # reusable code (data loading, utils)
├── results/          # figures, confusion matrices, tables
└── splits.json       # fixed train / val / test split
```

## Stack

PyTorch · Hugging Face Transformers · PEFT (LoRA) · scikit-learn · Weights & Biases · Google Colab (GPU)

## References

- Radford et al., 2021. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- Dosovitskiy et al., 2021. *An Image is Worth 16x16 Words* (ViT).
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models*.
- Helber et al., 2019. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*.
