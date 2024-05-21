<h1> Supplementary material for "Reshaping free-text radiology notes into structured reports with generative transformers" paper </h1>

1.  **Pre-processing** phase carried out on radiology reports and the related human-expert annotations: [colab notebook](https://colab.research.google.com/drive/1Ek5iphsgqEY7mvemarQ68hF9GBmNye_b?usp=sharing)

2.  **SR registry filling pipeline**: [SCRIPTS](https://github.com/laurabergomi/NLP_radiology/tree/d6b766bea7c732b0164301f5be428d193c97ed29/SCRIPTS)

3.  **Comparative analysis of models performances** and **human-expert evaluation analysis**: [colab notebook](https://colab.research.google.com/drive/1sOLWuxg_Srh6VRlFvxzCKPBGVgy6Z9ro?usp=sharing)


<h1> About the project:</h1>

<h2>Abstract</h2>
BACKGROUND: Radiology reports are typically written in a free-text format, making clinical information difficult to extract and use. Recently the adoption of structured reporting (SR) has been recommended by various medical societies thanks to the advantages it offers, e.g. standardization, completeness and information retrieval. We propose a pipeline to extract information from free-text radiology reports, that fits with the items of the reference SR registry proposed by a national society of interventional and medical radiology, focusing on CT staging of patients with lymphoma.

METHODS: Our work aims to leverage the potential of Natural Language Processing (NLP) and Transformer-based models to deal with automatic SR registry filling. With the availability of 174 radiology reports, we investigate a rule-free generative Question Answering approach based on a domain-specific version of T5 (IT5). Two strategies (batch-truncation and ex-post combination) are implemented to comply with the model’s context length limitations. Performance is evaluated in terms of strict accuracy, f1, and format accuracy, and compared with the widely used GPT-3.5 Large Language Model. A 5-point Likert scale questionnaire is used to collect human-expert feedback on the similarity between medical annotations and generated answers.

RESULTS: The combination of fine-tuning and batch splitting allows IT5 to achieve notable results; it performs on par with GPT-3.5 albeit its size being a thousand times smaller in terms of parameters. Human-based assessment scores show a high correlation (Spearman’s correlation coefficients>0.88, p-values<0.001) with AI performance metrics (f1) and confirm the superior ability of LLMs (i.e., GPT-3.5, 175B of parameters) in generating plausible human-like statements.

CONCLUSIONS: In our experimental setting, a smaller fine-tuned Transformer-based model with a modest number of parameters (i.e., IT5, 220M) performs well as a clinical information extraction system for automatic SR registry filling task, with superior ability to discern when an N.A. answer is the most correct result to a user query.

