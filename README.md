🩻 Insight-CXR
Multimodal Vision-Language Model for Chest X-Ray Question Answering and Visual Grounding

Insight-CXR is an ongoing capstone project that leverages a fine-tuned Qwen2.5-VL-3B Vision-Language Model to analyze chest X-rays, answer clinical questions, and localize relevant regions using bounding boxes.

The project combines medical visual question answering with visual grounding, enabling the model to generate both a textual response and the corresponding region of interest in the X-ray.

🚀 Project Overview

Insight-CXR takes a chest X-ray image along with a clinical question as input and generates:

💬 A clinical response
📦 Bounding-box coordinates corresponding to the relevant image region
Pipeline
Chest X-Ray + Clinical Question
              │
              ▼
      Qwen2.5-VL-3B Model
              │
              ▼
       LoRA / PEFT Fine-Tuning
              │
              ▼
    ┌───────────────────────┐
    │  Clinical Response    │
    │           +           │
    │ Bounding Box Location │
    └───────────────────────┘
              │
              ▼
       Evaluation Pipeline
              │
       ┌──────┴──────┐
       ▼             ▼
 Text Metrics   Localization Metrics
🧠 Model

The project uses:

Qwen2.5-VL-3B
Hugging Face Transformers
PEFT
LoRA Fine-Tuning
4-bit Quantization using BitsAndBytes

The base Vision-Language Model is combined with a fine-tuned LoRA adapter for inference.

Input
Chest X-ray Image
        +
Clinical Question
Output
Clinical Answer
        +
Bounding Box Coordinates

Bounding boxes are generated in the following format:

<box>[x1, y1, x2, y2]</box>

The coordinates correspond to regions on a 448 × 448 image.

🔄 Workflow
1. Input Processing

The model receives a chest X-ray along with a clinical question.

Example:

Image: Chest X-ray

Question:
Is there evidence of pleural effusion?
2. Model Inference

The image and question are processed by the fine-tuned Vision-Language Model.

Example output:

A pleural effusion is present.

<box>[x1, y1, x2, y2]</box>

The generated response is parsed into:

Clinical answer
Predicted bounding box
3. Visual Grounding

The predicted bounding box is compared with the corresponding ground-truth bounding box using Intersection over Union (IoU).

Ground Truth Box
       vs
Predicted Box
       │
       ▼
Intersection over Union (IoU)

The repository includes visualizations comparing:

🟢 Ground-truth bounding boxes
🔴 Predicted bounding boxes
📊 Evaluation

The model was evaluated on 10,102 test samples.

Text Generation Metrics

Generated clinical responses were evaluated using:

BLEU-1
BLEU-4
ROUGE-L

Additional evaluation experiments include:

BERTScore
CheXbert-based clinical evaluation
Localization Metrics

Bounding-box localization was evaluated using:

Mean IoU
Median IoU
IoU ≥ 0.50
IoU ≥ 0.75
IoU ≥ 0.90
Bounding Box Generation Rate
📈 Results
Metric	Score
Test Samples	10,102
BLEU-1	0.5110
BLEU-4	0.1142
ROUGE-L	0.7121
Mean IoU	0.7004
Median IoU	0.8117
IoU @ 0.50	77.38%
IoU @ 0.75	58.96%
IoU @ 0.90	31.39%
Bounding Box Generation Rate	98.71%
Understanding IoU@0.50

An IoU score of 0.50 or higher indicates that the predicted bounding box has sufficient overlap with the ground-truth region.

Therefore:

77.38% of evaluated samples achieved an IoU of at least 0.50.

The model generated a bounding box for 98.71% of the evaluated samples.

⚙️ Evaluation Pipeline

The evaluation pipeline performs the following steps:

Load Fine-Tuned Model
          │
          ▼
Load Test Dataset
          │
          ▼
Run Inference on Chest X-Ray
          │
          ▼
Generate Clinical Answer
          +
Generate Bounding Box
          │
          ▼
Parse Model Output
          │
     ┌────┴────┐
     ▼         ▼
Text Eval   Box Eval
     │         │
     ▼         ▼
BLEU       IoU
ROUGE      IoU@0.50
           IoU@0.75
           IoU@0.90
           BBox Rate

The evaluation script supports checkpoint-based processing, allowing the evaluation process to resume from previously processed samples.

📁 Repository Contents
Insight-CXR/
│
├── Untitled.ipynb
│   Main notebook containing model experiments,
│   inference, visualization, and bounding-box analysis.
│
├── Untitled1.ipynb
│   Advanced evaluation notebook containing additional
│   evaluation metrics and analysis.
│
├── eval_full.py
│   Full test-set evaluation script.
│
├── full_test_metrics.json
│   Final evaluation metrics for 10,102 test samples.
│
├── bad_localization_cases.csv
│   Examples of poor localization cases for analysis.
│
├── eval_full_log.txt
│   Log generated during full test-set evaluation.
│
├── sample_1.png
├── sample_2.png
├── sample_3.png
│   Example visualizations comparing predicted and
│   ground-truth bounding boxes.
│
└── multi_bbox_1.png
    Example containing multiple bounding-box regions.
🛠️ Technologies Used
Python
PyTorch
Hugging Face Transformers
Qwen2.5-VL-3B
PEFT
LoRA
BitsAndBytes
Pandas
NumPy
NLTK
ROUGE Score
PIL / Pillow
📷 Sample Visualizations

The repository includes examples of:

Single bounding-box localization
Multiple bounding-box localization
Ground-truth vs predicted regions
Localization failure analysis
🟢 Ground Truth Region

🔴 Predicted Region
🔬 Future Work

Potential improvements include:

Improving multi-bounding-box matching and evaluation
Building an interactive inference demo
Deploying the model as a web application
Expanding clinical evaluation
Further fine-tuning and experimentation
Improving explainability and visualization
⚠️ Disclaimer

Insight-CXR is an academic and research capstone project and is not intended for clinical diagnosis or real-world medical decision-making.

The model's outputs should not be used as a substitute for professional medical advice or radiologist interpretation.
