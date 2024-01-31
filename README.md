# MedTranslate

## Overview
This is a study focused on simplifying medical texts to improve health literacy, especially in under-resourced regions. The study uses the MedEasi corpus and the ctrlSIM model, employing a T5-Large model for simplification. To address computational limitations in third-world healthcare settings, a novel knowledge distillation approach is used, with a T5-Small model as a student model emulating the T5-Large teacher model. The performance of the student model is evaluated using various metrics, including SARI, ROUGE scores, and readability tests. While conventional metrics show satisfactory results, human evaluations reveal that the student model sometimes fails to simplify complex medical jargon. The research suggests the need for more user-centered evaluation methods and discusses future directions for improving text simplification and evaluation frameworks.

Report [paper](./CS_544_Final_Report.pdf).

## Training the Teacher Model

```
cd CTRL-SIMP
python training.py
```

The teacher model utilizes Basu et al. dataset and model. We finetune T5-large with multi-angle approach. We have modified the source code to run our settings to finetune T5-large. 

## Knowledge Distillation for the Student Model

Using a novel Knowledge Distillation function, inspired from the Mini-LM paper, we use:

<img src="https://latex.codecogs.com/gif.latex?\[\mathcal{L}_{ENC} = \frac{1}{A^{S}_{h}|x|}\Sigma_{i=1}^{A^{S}_{h}}\Sigma_{t=1}^{|x|}D_{KL}(A^{T}_{E,a,t} || A^{S}_{E,a,t})\] \\
\[\mathcal{L}_{DEC} = \frac{1}{A^{S}_{h}|x|}\Sigma_{i=1}^{A^{S}_{h}}\Sigma_{t=1}^{|x|}D_{KL}(A^{T}_{D,a,t} || A^{S}_{D,a,t})\] \\
\[\mathcal{L}_{total} = \mathcal{L}_{ENC} + \mathcal{L}_{DEC}\] \\"/>

to minimize the total Loss. 

You can train the student model by 
```
python kdMiniLM.py -tr
```

We made the evaluation through the python notebook at [evaluation](./metrics_notebook.ipynb). 

