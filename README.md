<h1> Supplementary material for "Reshaping free-text radiology notes into structured reports with generative question answering transformers" paper </h1>

1.  **Pre-processing** phase carried out on radiology reports and the related human-expert annotations: [colab notebook](https://colab.research.google.com/drive/1Ek5iphsgqEY7mvemarQ68hF9GBmNye_b?usp=sharing)

2.  **SR registry filling pipeline**: [SCRIPTS](https://github.com/bmi-labmedinfo/radiology-qa-transformer/tree/main/SCRIPTS)

3.  **Comparative analysis of models performances** and **human-expert evaluation analysis**: [colab notebook](https://colab.research.google.com/drive/1sOLWuxg_Srh6VRlFvxzCKPBGVgy6Z9ro?usp=sharing)


<h1> About the project:</h1>

<h2>Abstract</h2>
<h3>Background</h3>
Radiology reports are typically written in a free-text format, making clinical information difficult to extract and use. Recently, the adoption of structured reporting (SR) has been recommended by various medical societies thanks to the advantages it offers, e.g. standardization, completeness, and information retrieval. We propose a pipeline to extract information from Italian free-text radiology reports that fits with the items of the reference SR registry proposed by a national society of interventional and medical radiology, focusing on CT staging of patients with lymphoma.

<h3>Methods</h3>
Our work aims to leverage the potential of Natural Language Processing and Transformer-based models to deal with automatic SR registry filling. With the availability of 174 Italian radiology reports, we investigate a rule-free generative Question Answering approach based on the Italian-specific version of T5: IT5. To address information content discrepancies, we focus on the six most frequently filled items in the annotations made on the reports: three categorical (multichoice), one free-text (free-text), and two continuous numerical (factual). In the preprocessing phase, we encode also information that is not supposed to be entered. Two strategies (batch-truncation and ex-post combination) are implemented to comply with the IT5 context length limitations. Performance is evaluated in terms of strict accuracy, f1, and format accuracy, and compared with the widely used GPT-3.5 Large Language Model. Unlike multichoice and factual, free-text answers do not have 1-to-1 correspondence with their reference annotations. For this reason, we collect human-expert feedback on the similarity between medical annotations and generated free-text answers, using a 5-point Likert scale questionnaire (evaluating the criteria of correctness and completeness).

<h3>Results</h3>
The combination of fine-tuning and batch splitting allows IT5 ex-post combination to achieve notable results in terms of information extraction of different types of structured data, performing on par with GPT-3.5. Human-based assessment scores of free-text answers show a high correlation with the AI performance metrics f1 (Spearman's correlation coefficients>0.5, p-values<0.001) for both IT5 ex-post combination and GPT-3.5. The latter is better at generating plausible human-like statements, even if it systematically provides answers even when they are not supposed to be given.

<h3>Conclusions</h3>
In our experimental setting, a fine-tuned Transformer-based model with a modest number of parameters (i.e., IT5, 220 M) performs well as a clinical information extraction system for automatic SR registry filling task. It can extract information from more than one place in the report, elaborating it in a manner that complies with the response specifications provided by the SR registry (for multichoice and factual items), or that closely approximates the work of a human-expert (free-text items); with the ability to discern when an answer is supposed to be given or not to a user query.

<h3>Keywords</h3>
Natural language processingClinical textGenerative artificial intelligenceRadiologyLymphomaBiomedical information extraction

<h1> Cite this paper </h1>

```
@article{BERGOMI2024102924,
  title = {Reshaping free-text radiology notes into structured reports with generative question answering transformers},
  journal = {Artificial Intelligence in Medicine},
  pages = {102924},
  year = {2024},
  issn = {0933-3657},
  doi = {https://doi.org/10.1016/j.artmed.2024.102924},
  url = {https://www.sciencedirect.com/science/article/pii/S0933365724001660},
  author = {Laura Bergomi and Tommaso M. Buonocore and Paolo Antonazzo and Lorenzo Alberghi and Riccardo Bellazzi and Lorenzo Preda and Chandra Bortolotto and Enea Parimbelli},
  keywords = {Natural language processing, Clinical text, Generative artificial intelligence, Radiology, Lymphoma, Biomedical information extraction}
}
```
