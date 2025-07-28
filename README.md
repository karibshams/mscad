📚 References & Inspirations
This project draws inspiration from several excellent implementations and experiments shared by the Kaggle community. A heartfelt thank you to the authors of the following notebooks for their contributions and insights:

Mean Teacher with Swin Transformer
https://www.kaggle.com/code/aritra5101999/mean-teacher-with-swin-backbone/comments?scriptVersionId=252458326

A great implementation of the Mean Teacher semi-supervised learning method using a Swin Transformer backbone.

FixMatch with Swin Transformer
https://www.kaggle.com/code/ari7889/fixmatch-with-swin-transformer

This notebook inspired the use of FixMatch with modern transformer-based vision backbones.

Swin Transformer Supervised
https://www.kaggle.com/code/ari7889/swin-transformer-supervised/comments?scriptVersionId=252426681
A clean and simple supervised training pipeline using the Swin Transformer architecture.

Swin DINOv2 Final
https://www.kaggle.com/code/shamskarib/swin-dinov2final?scriptVersionId=252494025


Insightful use of Swin as a backbone with DINOv2-style self-supervised learning.

ViT DINOv2 Final
https://www.kaggle.com/code/karib123/vit-dinov2last?scriptVersionId=252494173

A strong baseline using Vision Transformer and DINOv2 for representation learning.

These notebooks provided valuable guidance and ideas that helped shape the direction of this project. Please check them out and support the original authors.

🧠 Swin Transformer + Self-Supervised Learning (SSL) for Medical Imaging
Course: CSE521
Notebook: Swin_SSL_CSE521.ipynb

📌 Objective
This project benchmarks two pipelines on a medical imaging dataset to evaluate the impact of Self-Supervised Learning (SSL) on Swin Transformer performance:

Supervised Baseline: Fine-tuned Swin Transformer.

SSL Variant: Pre-trained Swin backbone using the Mean Teacher SSL technique, followed by fine-tuning or linear probing.

🗂️ Pipeline Overview
1️⃣ Data Pre-processing
Techniques: Resizing, normalization, and class balancing.

Optional Augmentations: Adaptive histogram equalization to enhance local contrast, especially useful for X-ray imagery.

2️⃣ Supervised Baseline
Model: swin_tiny_patch4_window7_224 (via timm)

Training: Mixed-precision training with torch.cuda.amp

Metrics: Training & validation loss, precision, recall, F1-score

3️⃣ SSL Variant
Method Used: Mean Teacher

Strategy:

Use a student-teacher architecture, where both networks share the same architecture (Swin Transformer).

The teacher model is an exponential moving average (EMA) of the student model.

The student learns from both labeled and unlabeled images, using a consistency loss between the outputs of student and teacher on augmented views.

Comparison: Final performance is benchmarked against the fully supervised Swin model.

📊 Visualisation & Evaluation
📈 Loss Curves: Single plot comparing supervised and SSL training

📊 Table / Bar Chart: Summarized metrics (Precision, Recall, F1)

🔍 Bonus (Optional): Grad-CAM/Attention maps to visualize what the model focuses on

🧪 Results
Model	Precision	Recall	F1-Score
Swin (Supervised)	0.85	0.83	0.84
Swin + SSL (Mean Teacher)	0.88	0.87	0.87

(Replace these values with actual experiment outputs)

▶️ How to Reproduce
Open Swin_SSL_CSE521.ipynb in Google Colab.

Mount your dataset (or upload directly).

Set runtime to GPU via Runtime > Change Runtime Type.

Run all cells — both supervised and SSL (Mean Teacher) pipelines are implemented.

🧠 SSL Method Description
Mean Teacher is a semi-supervised learning method that uses two models: a student and a teacher. Both share the same architecture (in this case, a Swin Transformer), but the teacher’s weights are an exponential moving average (EMA) of the student’s.

During training:

The student is updated via backpropagation using both labeled and unlabeled data.

The teacher generates consistent targets for the student by using different data augmentations.

The student learns not only from true labels but also by matching the teacher’s predictions — encouraging temporal consistency and improved generalization.

This method is particularly useful in medical domains where labeled data is limited.

📦 Submission Files
✅ Swin_SSL_CSE521.ipynb

✅ Checkpoints:

swin_supervised_best.pth

swin_ssl_meanteacher_best.pth

✅ README.md

📁 Final Zip: Group_ID_SwinSSL.zip
