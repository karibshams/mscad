1. Objective
Implement and benchmark two pipelines on your assigned medical-imaging dataset:
1. A supervised Swin Transformer baseline.

2. A Swin Transformer + Self-Supervised Learning (SSL) variant using any well-
motivated SSL technique (e.g. SimCLR, BYOL, MoCo v3, DINO, Mean Teacher, Mix-
Match, FixMatch, . . . ).


3. Tasks & Notebook Requirements
1. Data Pre-processing
• Resizing, normalisation, class-balancing techniques.
• Optionally include domain-specific augmentations (e.g. adaptive histogram equalisation
for X-Ray).
2. Baseline: Swin Transformer (Supervised)
• Use any timm Swin variant (swin_tiny_patch4_window7_224 or larger).
• Fine-tune with mixed-precision (torch.cuda.amp).
• Log train & validation loss, precision, recall, F1.
3. Swin Transformer + SSL
• Pre-train the backbone with one SSL algorithm of your choice on the unlabelled portion
(or on labelled data without labels).

• Either (a) frozen-backbone linear probe or (b) fine-tune the SSL-initialised model end-
to-end.

• Compare metrics against the supervised baseline.
