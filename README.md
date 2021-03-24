# BERT-FAQ
BERT-FAQ is an extensible, open-source Python-based toolkit for building and evaluating a transformer-based FAQ retrieval system. This repository contains FAQ publicly available datasets such as: FAQIRv1.0 [1], Stack-exchange FAQ [2], and more recent Covid-19 FAQ [3]. The goal of this project is to compare and evaluate various supervised & unsupervised deep learning-based FAQ retrieval techniques, as well as, for computing the retrieval effectiveness among different techniques of interest.

Technologies used: Python 3.8.3, Elasticsearch 7.10.2, Jupyter Notebook, Google Colaboratory

## Evaluation:
```
|-----------------------------------------------------------------------------------------------------------------|
|                                                   CovidFAQ                                                      |
|------------|----------------------|-------|---------|--------|------|------|--------|------|------|------|------|
|   Method   |        Matching      |       |Training |Negative|      |      |        |      |      |      |      |
|            |          Field       | Loss  |   Data  |Sampling|NDCG@3|NDCG@5|NDCG@10 |  P@3 |  P@5 | P@10 |  MAP |
|:----------:|:--------------------:|:-----:|:-------:|:------:|:----:|:----:|:------:|:----:|:----:|:----:|:----:|
|Unsupervised|answer                |       |         |        |0.2536|0.1221| 0.2940 |0.0985|0.3466|0.0730|0.2846| 
|Unsupervised|question              |       |         |        |0.7844|0.3451| 0.8104 |0.2286|0.8147|0.1324|0.7422| 
|Unsupervised|question_answer       |       |         |        |0.6976|0.3080| 0.7246 |0.2046|0.7465|0.1154|0.6604|  
|Unsupervised|question_answer_concat|       |         |        |0.5756|0.2588| 0.6067 |0.1846|0.6392|0.1073|0.5705| 
| Supervised |answer                |triplet|   faq   | simple |0.4896|0.2427| 0.5308 |0.1740|0.5680|0.1030|0.4937| 
| Supervised |question              |triplet|   faq   | simple |0.8142|0.3646| 0.8357 |0.2364|0.8383|0.1347|0.7840| 
| Supervised |question_answer       |triplet|   faq   | simple |0.7455|0.3377| 0.7674 |0.2180|0.7903|0.1199|0.7209| 
| Supervised |question_answer_concat|triplet|   faq   | simple |0.6754|0.3089| 0.7129 |0.2119|0.7444|0.1194|0.6763| 
| Supervised |answer                |triplet|   faq   |  hard  |0.5277|0.2616| 0.5642 |0.1872|0.5960|0.1108|0.5127| 
| Supervised |question              |triplet|   faq   |  hard  |0.8249|0.3677| 0.8450 |0.2494|0.8497|0.1362|0.7946| 
| Supervised |question_answer       |triplet|   faq   |  hard  |0.7671|0.3423| 0.7958 |0.2327|0.8025|0.1262|0.7408| 
| Supervised |question_answer_concat|triplet|   faq   |  hard  |0.7173|0.3265| 0.7508 |0.2263|0.7689|0.1252|0.7029| 
| Supervised |answer                |triplet|faq+query| simple |0.7859|0.3803| 0.7980 |0.2421|0.8009|0.1234|0.7533| 
| Supervised |question              |triplet|faq+query| simple |0.9588|0.4310| 0.9604 |0.2644|0.9604|0.1377|0.9429|
| Supervised |question_answer       |triplet|faq+query| simple |0.9377|0.4258| 0.9417 |0.2631|0.9414|0.1321|0.9168|
| Supervised |question_answer_concat|triplet|faq+query| simple |0.9242|0.4239| 0.9300 |0.2629|0.9304|0.1321|0.9040|
| Supervised |answer                |triplet|faq+query|  hard  |0.8595|0.3995| 0.8630 |0.2462|0.8630|0.1234|0.8380|
| Supervised |question              |triplet|faq+query|  hard  |0.9755|0.4335| 0.9753 |0.2648|0.9753|0.1377|0.9626|
| Supervised |question_answer       |triplet|faq+query|  hard  |0.9747|0.4338| 0.9743 |0.2638|0.9743|0.1321|0.9613|
| Supervised |question_answer_concat|triplet|faq+query|  hard  |0.9734|0.4363| 0.9736 |0.2640|0.9736|0.1321|0.9631|
| Supervised |answer                |softmax|   faq   | simple |0.2622|0.1277| 0.2957 |0.0991|0.3472|0.0730|0.2859|
| Supervised |question              |softmax|   faq   | simple |0.7832|0.3358| 0.8091 |0.2258|0.8140|0.1321|0.7460|
| Supervised |question_answer       |softmax|   faq   | simple |0.6959|0.3071| 0.7242 |0.2046|0.7461|0.1154|0.6607|
| Supervised |question_answer_concat|softmax|   faq   | simple |0.5762|0.2585| 0.6079 |0.1846|0.6404|0.1073|0.5739| 
| Supervised |answer                |softmax|   faq   |  hard  |0.2866|0.1373| 0.3191 |0.1045|0.3653|0.0740|0.3058| 
| Supervised |question              |softmax|   faq   |  hard  |0.7881|0.3454| 0.8162 |0.2290|0.8201|0.1324|0.7554| 
| Supervised |question_answer       |softmax|   faq   |  hard  |0.7002|0.3086| 0.7279 |0.2052|0.7497|0.1155|0.6655| 
| Supervised |question_answer_concat|softmax|   faq   |  hard  |0.5931|0.2625| 0.6222 |0.1855|0.6538|0.1076|0.5912|  
| Supervised |answer                |softmax|faq+query| simple |0.4739|0.2260| 0.5043 |0.1568|0.5501|0.1009|0.4661|  
| Supervised |question              |softmax|faq+query| simple |0.8821|0.3723| 0.8846 |0.2453|0.8899|0.1342|0.8465|  
| Supervised |question_answer       |softmax|faq+query| simple |0.8193|0.3485| 0.8365 |0.2245|0.8470|0.1214|0.7785|  
| Supervised |question_answer_concat|softmax|faq+query| simple |0.7024|0.3052| 0.7224 |0.2063|0.7554|0.1207|0.6893| 
| Supervised |answer                |softmax|faq+query|  hard  |0.8361|0.3766| 0.8389 |0.2302|0.8434|0.1178|0.8224|  
| Supervised |question              |softmax|faq+query|  hard  |0.9481|0.4212| 0.9530 |0.2603|0.9547|0.1369|0.9393|  
| Supervised |question_answer       |softmax|faq+query|  hard  |0.9290|0.4156| 0.9346 |0.2551|0.9390|0.1297|0.9198|  
| Supervised |question_answer_concat|softmax|faq+query|  hard  |0.9254|0.4097| 0.9307 |0.2521|0.9351|0.1288|0.9157|
|-----------------------------------------------------------------------------------------------------------------|

|-----------------------------------------------------------------------------------------------------------------|
|                                                   StackFAQ                                                      |
|------------|----------------------|-------|---------|--------|------|------|--------|------|------|------|------|
|   Method   |        Matching      |       |Training |Negative|      |      |        |      |      |      |      |
|            |          Field       | Loss  |   Data  |Sampling|NDCG@3|NDCG@5|NDCG@10 |  P@3 |  P@5 | P@10 |  MAP |
|:----------:|:--------------------:|:-----:|:-------:|:------:|:----:|:----:|:------:|:----:|:----:|:----:|:----:|
|Unsupervised|answer                |       |         |        |0.6012|0.3608| 0.6269 |0.2865|0.6427|0.1946|0.3972| 
|Unsupervised|question              |       |         |        |0.5803|0.5645| 0.5923 |0.5039|0.6222|0.3468|0.6648| 
|Unsupervised|question_answer       |       |         |        |0.6803|0.5161| 0.7067 |0.4524|0.7253|0.3196|0.5893|  
|Unsupervised|question_answer_concat|       |         |        |0.7615|0.6138| 0.7831 |0.5284|0.7893|0.3637|0.6366| 
| Supervised |answer                |triplet|   faq   | simple |0.7713|0.5426| 0.7859 |0.4484|0.7839|0.3081|0.6472| 
| Supervised |question              |triplet|   faq   | simple |0.7582|0.7128| 0.7707 |0.6360|0.7928|0.4354|0.7605| 
| Supervised |question_answer       |triplet|   faq   | simple |0.8250|0.6768| 0.8372 |0.5986|0.8400|0.4160|0.7436| 
| Supervised |question_answer_concat|triplet|   faq   | simple |0.8511|0.7134| 0.8578 |0.6224|0.8578|0.4284|0.7593| 
| Supervised |answer                |triplet|   faq   |  hard  |0.7863|0.6066| 0.7998 |0.4967|0.8022|0.3198|0.7066| 
| Supervised |question              |triplet|   faq   |  hard  |0.7473|0.7040| 0.7628 |0.6296|0.7800|0.4267|0.7527| 
| Supervised |question_answer       |triplet|   faq   |  hard  |0.8193|0.7006| 0.8306 |0.6208|0.8408|0.4178|0.7635| 
| Supervised |question_answer_concat|triplet|   faq   |  hard  |0.8360|0.7326| 0.8448 |0.6432|0.8556|0.4322|0.7777| 
| Supervised |answer                |triplet|faq+query| simple |0.9080|0.7230| 0.9059 |0.5888|0.9006|0.3679|0.8406| 
| Supervised |question              |triplet|faq+query| simple |0.8408|0.8092| 0.8519 |0.7265|0.8630|0.4785|0.8479|
| Supervised |question_answer       |triplet|faq+query| simple |0.9348|0.8313| 0.9375 |0.7279|0.9371|0.4768|0.8997|
| Supervised |question_answer_concat|triplet|faq+query| simple |0.9466|0.8580| 0.9446 |0.7473|0.9443|0.4911|0.9048|
| Supervised |answer                |triplet|faq+query|  hard  |0.9712|0.8353| 0.9696 |0.6549|0.9684|0.3748|0.9558|
| Supervised |question              |triplet|faq+query|  hard  |0.8933|0.8636| 0.8960 |0.7629|0.8977|0.4875|0.8929|
| Supervised |question_answer       |triplet|faq+query|  hard  |0.9821|0.9127| 0.9818 |0.7829|0.9812|0.4863|0.9739|
| Supervised |question_answer_concat|triplet|faq+query|  hard  |0.9893|0.9309| 0.9879 |0.8045|0.9864|0.5034|0.9781|
| Supervised |answer                |softmax|   faq   | simple |0.6724|0.4534| 0.6943 |0.3712|0.7063|0.2518|0.5174|
| Supervised |question              |softmax|   faq   | simple |0.7127|0.6216| 0.7303 |0.5388|0.7411|0.3781|0.6522|
| Supervised |question_answer       |softmax|   faq   | simple |0.7353|0.5559| 0.7531 |0.4679|0.7617|0.3340|0.5932|
| Supervised |question_answer_concat|softmax|   faq   | simple |0.7631|0.5863| 0.7793 |0.4942|0.7820|0.3480|0.6111| 
| Supervised |answer                |softmax|   faq   |  hard  |0.6018|0.3624| 0.6285 |0.2881|0.6447|0.1964|0.3998| 
| Supervised |question              |softmax|   faq   |  hard  |0.5808|0.5645| 0.5921 |0.5034|0.6224|0.3465|0.6373| 
| Supervised |question_answer       |softmax|   faq   |  hard  |0.6805|0.5169| 0.7074 |0.4533|0.7254|0.3201|0.5696| 
| Supervised |question_answer_concat|softmax|   faq   |  hard  |0.7619|0.6146| 0.7827 |0.5284|0.7891|0.3639|0.6362|  
| Supervised |answer                |softmax|faq+query| simple |0.8800|0.7011| 0.8840 |0.5821|0.8832|0.3613|0.8098|  
| Supervised |question              |softmax|faq+query| simple |0.8397|0.8060| 0.8567 |0.7284|0.8659|0.4731|0.8371|  
| Supervised |question_answer       |softmax|faq+query| simple |0.9096|0.7934| 0.9162 |0.7074|0.9170|0.4646|0.8595|  
| Supervised |question_answer_concat|softmax|faq+query| simple |0.9176|0.8156| 0.9233 |0.7222|0.9235|0.4788|0.8629| 
| Supervised |answer                |softmax|faq+query|  hard  |0.8943|0.7206| 0.8958 |0.5649|0.8934|0.3369|0.8075|  
| Supervised |question              |softmax|faq+query|  hard  |0.8359|0.7748| 0.8428 |0.6762|0.8484|0.4421|0.7941|  
| Supervised |question_answer       |softmax|faq+query|  hard  |0.9222|0.8143| 0.9242 |0.6913|0.9206|0.4430|0.8473|  
| Supervised |question_answer_concat|softmax|faq+query|  hard  |0.9353|0.8308| 0.9346 |0.7110|0.9287|0.4567|0.8522|
|-----------------------------------------------------------------------------------------------------------------|

|-----------------------------------------------------------------------------------------------------------------|
|                                                   FAQIR                                                         |
|------------|----------------------|-------|---------|--------|------|------|--------|------|------|------|------|
|   Method   |        Matching      |       |Training |Negative|      |      |        |      |      |      |      |
|            |          Field       | Loss  |   Data  |Sampling|NDCG@3|NDCG@5|NDCG@10 |  P@3 |  P@5 | P@10 |  MAP |
|:----------:|:--------------------:|:-----:|:-------:|:------:|:----:|:----:|:------:|:----:|:----:|:----:|:----:|
|Unsupervised|answer                |       |         |        |0.2667|0.1431|0.2905  |0.1212|0.3069|0.0916|0.1884|
|Unsupervised|question              |       |         |        |0.3798|0.2060|0.4045  |0.1726|0.4258|0.1303|0.2781|
|Unsupervised|question_answer       |       |         |        |0.3761|0.2029|0.4008  |0.1708|0.4225|0.1327|0.2664|
|Unsupervised|question_answer_concat|       |         |        |0.4318|0.2482|0.4573  |0.2130|0.4783|0.1606|0.3106|
| Supervised |answer                |triplet|   faq   | simple |0.4501|0.2787|0.4724  |0.2455|0.4915|0.1801|0.3909| 
| Supervised |question              |triplet|   faq   | simple |0.5409|0.3347|0.5643  |0.2872|0.5774|0.2198|0.4571| 
| Supervised |question_answer       |triplet|   faq   | simple |0.5378|0.3350|0.5642  |0.2921|0.5811|0.2286|0.4561|
| Supervised |question_answer_concat|triplet|   faq   | simple |0.5740|0.3652|0.5938  |0.3167|0.6095|0.2474|0.4724| 
| Supervised |answer                |triplet|   faq   |  hard  |0.3991|0.2291|0.4156  |0.1858|0.4386|0.1368|0.3202|
| Supervised |question              |triplet|   faq   |  hard  |0.4889|0.2818|0.5145  |0.2294|0.5335|0.1680|0.3788|
| Supervised |question_answer       |triplet|   faq   |  hard  |0.4970|0.2877|0.5169  |0.2355|0.5330|0.1753|0.3803|
| Supervised |question_answer_concat|triplet|   faq   |  hard  |0.5255|0.3178|0.5472  |0.2595|0.5683|0.1933|0.4017| 
| Supervised |answer                |triplet|faq+query| simple |0.5144|0.3266|0.5337  |0.2778|0.5497|0.2009|0.4594|
| Supervised |question              |triplet|faq+query| simple |0.6601|0.4342|0.6733  |0.3697|0.6769|0.2707|0.5925|
| Supervised |question_answer       |triplet|faq+query| simple |0.6412|0.4210|0.6584  |0.3582|0.6668|0.2669|0.5670|
| Supervised |question_answer_concat|triplet|faq+query| simple |0.6658|0.4418|0.6875  |0.3866|0.6943|0.2905|0.5843|
| Supervised |answer                |triplet|faq+query|  hard  |0.7136|0.5151|0.7130  |0.3988|0.7126|0.2480|0.7059|
| Supervised |question              |triplet|faq+query|  hard  |0.8122|0.5962|0.8111  |0.4773|0.8105|0.3172|0.8020|
| Supervised |question_answer       |triplet|faq+query|  hard  |0.8247|0.6238|0.8239  |0.5035|0.8232|0.3331|0.8158|
| Supervised |question_answer_concat|triplet|faq+query|  hard  |0.8549|0.6653|0.8541  |0.5459|0.8533|0.3707|0.8454|
| Supervised |answer                |softmax|   faq   | simple |0.3386|0.1986|0.3648  |0.1680|0.3864|0.1309|0.2706|
| Supervised |question              |softmax|   faq   | simple |0.4118|0.2451|0.4466  |0.2103|0.4722|0.1620|0.3199|
| Supervised |question_answer       |softmax|   faq   | simple |0.3893|0.2257|0.4187  |0.1975|0.4446|0.1568|0.2972|
| Supervised |question_answer_concat|softmax|   faq   | simple |0.4241|0.2513|0.4566  |0.2154|0.4769|0.1668|0.3140|
| Supervised |answer                |softmax|   faq   |  hard  |0.2690|0.1440|0.2938  |0.1224|0.3096|0.0921|0.1908|
| Supervised |question              |softmax|   faq   |  hard  |0.3786|0.2054|0.4044  |0.1723|0.4250|0.1300|0.2783|
| Supervised |question_answer       |softmax|   faq   |  hard  |0.3757|0.2043|0.4017  |0.1713|0.4230|0.1332|0.2675|
| Supervised |question_answer_concat|softmax|   faq   |  hard  |0.4318|0.2488|0.4560  |0.2120|0.4776|0.1605|0.3114| 
| Supervised |answer                |softmax|faq+query| simple |0.4941|0.3102|0.5173  |0.2654|0.5297|0.1877|0.4333|
| Supervised |question              |softmax|faq+query| simple |0.6533|0.4119|0.6656  |0.3454|0.6702|0.2513|0.5671|
| Supervised |question_answer       |softmax|faq+query| simple |0.6174|0.3962|0.6385  |0.3410|0.6487|0.2516|0.5328|
| Supervised |question_answer_concat|softmax|faq+query| simple |0.6606|0.4283|0.6743  |0.3662|0.6801|0.2711|0.5540|
| Supervised |answer                |softmax|faq+query|  hard  |0.6791|0.4776|0.6797  |0.3716|0.6785|0.2418|0.6505| 
| Supervised |question              |softmax|faq+query|  hard  |0.7686|0.5413|0.7710  |0.4382|0.7683|0.2995|0.7287|
| Supervised |question_answer       |softmax|faq+query|  hard  |0.7960|0.5827|0.7950  |0.4719|0.7920|0.3215|0.7587|  
| Supervised |question_answer_concat|softmax|faq+query|  hard  |0.8345|0.6329|0.8339  |0.5194|0.8313|0.3595|0.8030|
|-----------------------------------------------------------------------------------------------------------------|

```

## Setup
```
1. Clone repository

2. install conda library 
   pip3 install conda

3. create conda environment
   conda create --name faq
   conda activate faq
   
4. install required libraries
   conda install elasticsearch
   conda install elasticsearch-dsl
   conda install flask
   conda install pandas
   conda install numpy
   conda install pytorch
   conda install scikit-learn
   conda install xmltodict
   conda install sentence-transformers
   conda install -c conda-forge notebook
```

## Project Outline
This section presents general steps in performing preprocessing, ground-truth creation, model training 
and evaluation for an FAQ retrieval system. These steps are illustrated below using the aformentioned
publicly available datasets: CovidFAQ, StackFAQ, and FAQIR datasets.

1. Parsing FAQ Dataset
    * [CovidFAQ](notebook/CovidFAQ/01.Parsing_aligned_question_question_answer.csv_file.ipynb)
    * [StackFAQ](notebook/StackFAQ/01.Parsing_stackExchange-FAQ.xml_file.ipynb)
    * [FAQIR](notebook/FAQIR/01.Parsing_FAQIRv1.0.xml_file.ipynb)
2. Ingesting Data To Elasticsearch
    * [CovidFAQ](notebook/CovidFAQ/02.Ingesting_Data_to_Elasticsearch.ipynb)
    * [StackFAQ](notebook/StackFAQ/02.Ingesting_Data_to_Elasticsearch.ipynb)
    * [FAQIR](notebook/FAQIR/02.Ingesting_Data_to_Elasticsearch.ipynb)
3. Generating Negative Samples
    * [CovidFAQ](notebook/CovidFAQ/03.Generating_Hard_Negatives.ipynb)
    * [StackFAQ](notebook/StackFAQ/03.Generating_Hard_Negatives.ipynb)
    * [FAQIR](notebook/FAQIR/03.Generating_Hard_Negatives.ipynb)
4. Generating Triplet Dataset
    * [CovidFAQ](notebook/CovidFAQ/04.Generating_Ground_Truth_Dataset.ipynb)
    * [StackFAQ](notebook/StackFAQ/04.Generating_Ground_Truth_Dataset.ipynb)
    * [FAQIR](notebook/FAQIR/04.Generating_Ground_Truth_Dataset.ipynb)
5. Generating Elasticsearch Top-k For Re-ranking
    * [CovidFAQ](notebook/CovidFAQ/05.Generating_ES_Topk_Results_For_Reranking.ipynb)
    * [StackFAQ](notebook/StackFAQ/05.Generating_ES_Topk_Results_For_Reranking.ipynb)
    * [FAQIR](notebook/FAQIR/05.Generating_ES_Topk_Results_For_Reranking.ipynb)
6. Model Training
    * [CovidFAQ](notebook/CovidFAQ/06.Model_Training.ipynb)
    * [StackFAQ](notebook/StackFAQ/06.Model_Training.ipynb)
    * [FAQIR](notebook/FAQIR/06.Model_Training.ipynb)
7. Generating Re-ranked results
    * [CovidFAQ](notebook/CovidFAQ/07.Generating_Reranked_Results.ipynb)
    * [StackFAQ](notebook/StackFAQ/07.Generating_Reranked_Results.ipynb)
    * [FAQIR](notebook/FAQIR/07.Generating_Reranked_Results.ipynb)
8. Evaluation
    * [CovidFAQ](notebook/CovidFAQ/08.Evaluation.ipynb)
    * [StackFAQ](notebook/StackFAQ/08.Evaluation.ipynb)
    * [FAQIR](notebook/FAQIR/08.Evaluation.ipynb)

```
To run project outline steps do the following: 
  - navigate to "notebook" directory 
    cd notebook
  - open notebook directory in Jupyter Notebook
    jupyter notebook
    
Start running the notebooks in the following order: 
 
 (1) Local Machine (CPU)
    * to parse CovidFAQ, StackFAQ, FAQIR datasets 
      open the following notebooks:
          01.Parsing_aligned_question_question_answer.csv_file.ipynb
          01.Parsing_stackExchange-FAQ.xml_file.ipynb
          01.Parsing_FAQIRv1.0.xml_file.ipynb

      The convention used below in directory tree structure
        CovidFAQ / StackFAQ / FAQIR  represents directory name for a given dataset name
      
    After running the notebook script it will generate the following files:
          |
          |  |-- data
          |  |
          |  |  |-- CovidFAQ
          |  |  |  |  |-- query_answer_pairs.json
          |  |  |       
          |  |  |-- StackFAQ
          |  |  |  |  |-- stackExchange-FAQ.json
          |  |  |  |  |-- query_answer_pairs.json
          |  |
          |  |  |-- FAQIR
          |  |  |  |  |-- FAQIRv1.0.json
          |  |  |  |  |-- query_answer_pairs.json
          
   * to ingest data to Elasticseach (CovidFAQ, StackFAQ, FAQIR)
     open notebook
         02.Ingesting_Data_To_Elasticsearch.ipynb
        
   * to generate hard negative samples (CovidFAQ, StackFAQ, FAQIR)
     open notebook
         03.Generating_Hard_Negatives.ipynb
        
     After running the notebook script it will generate the following files:
             
          |  |-- data
          |  |
          |  |  |-- CovidFAQ / StackFAQ / FAQIR
          |  |  |  |  |-- hard_negatives_faq.json
          |  |  |  |  |-- hard_negatives_user_query.json
          |  |  |       
          
  * to generate triplet dataset for BERT finetuning (CovidFAQ, StackFAQ, FAQIR)
     open notebook
        04.Generating_Ground_Truth_Dataset.ipynb
       
    After running the notebook script it will generate the following directories and files:
     
          |  |-- data
          |  |
          |  |  |-- CovidFAQ / StackFAQ / FAQIR 
          |  |  |         
          |  |  |  |-- dataset
          |  |  |  |
          |  |  |  |  |-- softmax
          |  |  |  |  |  |-- faq
          |  |  |  |  |  |    |-- hard_faq_dataset.csv
          |  |  |  |  |  |    |-- simple_faq_dataset.csv
          |  |  |  |  |  |
          |  |  |  |  |  |-- user_query
          |  |  |  |  |  |    |-- hard_faq_dataset.csv
          |  |  |  |  |  |    |-- simple_faq_dataset.csv
          |  |  |  |  |  |  
          |  |  |  |  |-- triplet
          |  |  |  |  |  |-- faq
          |  |  |  |  |  |  |  |-- hard_faq_dataset.csv
          |  |  |  |  |  |  |  |-- simple_faq_dataset.csv
          |  |  |  |  |  |
          |  |  |  |  |  |-- user_query
          |  |  |  |  |  |  |  |-- hard_faq_dataset.csv
          |  |  |  |  |  |  |  |-- simple_faq_dataset.csv
          

  * to generate ES top-k results for re-ranking (CovidFAQ, StackFAQ, FAQIR)
     open notebook
       05.Generating_ES_Topk_Results_For_Reranking.ipynb
      
    After running the notebook script it will generate the following directories and files:
      
          |  |-- data
          |  |  |
          |  |  |-- CovidFAQ / StackFAQ / FAQIR      
          |  |  |  |  |-- rank_results
          |  |  |  |  |  |  |  |-- unsupervised
          |  |  |  |  |  |  |  |  |  |  |-- es_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |-- es_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |-- es_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |-- es_query_by_question_answer_concat.json
          
          
  (2) Google Collaboratory (GPU)
    * to finetune BERT model
      - copy BERT-FAQ project directory + generated triplet datasets to Google Drive
      - run Google Colaboratory notebook
      - give permission control using credentials 
          06.Model_Training
      
      After running the notebook script it will generate the following directories and files:
        * models: directory containing PyTorch models & evaluation results
        * evaluation: directory containing triplet dataset split into train.csv, test.csv, val.csv sets
    
       |  |-- output
       |  |  |
       |  |  |  |-- CovidFAQ / StackFAQ / FAQIR 
       |  |  |  |         
       |  |  |  |  |-- models
       |  |  |  |  |  |  |-- softmax_simple_faq_1.1
       |  |  |  |  |  |  |-- softmax_simple_user_query_1.1
       |  |  |  |  |  |  |-- softmax_hard_faq_1.1
       |  |  |  |  |  |  |-- softmax_hard_user_query_1.1
       |  |  |  |  |  |  |-- triplet_simple_faq_1.1
       |  |  |  |  |  |  |-- triplet_simple_user_query_1.1
       |  |  |  |  |  |  |-- triplet_hard_faq_1.1
       |  |  |  |  |  |  |-- triplet_hard_user_query_1.1
       |  |  |  |  |-- evaluation
       |  |  |  |  |  |  |-- softmax_simple_faq_1.1
       |  |  |  |  |  |  |-- softmax_simple_user_query_1.1
       |  |  |  |  |  |  |-- softmax_hard_faq_1.1
       |  |  |  |  |  |  |-- softmax_hard_user_query_1.1
       |  |  |  |  |  |  |-- triplet_simple_faq_1.1
       |  |  |  |  |  |  |-- triplet_simple_user_query_1.1
       |  |  |  |  |  |  |-- triplet_hard_faq_1.1
       |  |  |  |  |  |  |-- triplet_hard_user_query_1.1
          
          
    * to perform re-ranking
      - copy BERT_FAQ project directory + generated rank_results directory (CovidFAQ, StackFAQ, FAQIR)
      - run Google Colaboratory notebook
           07.Generating_Reranked_Results
      
      After running the notebook script it will generate the following files:
        
          |  |-- data
          |  |  |
          |  |  |-- CovidFAQ / StackFAQ / FAQIR      
          |  |  |  |  |-- rank_results
          |  |  |  |  |  |  |  |-- unsupervised
          |  |  |  |  |  |  |  |  |  |  |-- es_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |-- es_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |-- es_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |-- es_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |-- supervised
          |  |  |  |  |  |  |  |  |  |  |-- softmax
          |  |  |  |  |  |  |  |  |  |  |  |  |-- faq
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- simple
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- hard
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |-- user_query
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- simple
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- hard
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |-- triplet
          |  |  |  |  |  |  |  |  |  |  |  |  |-- faq
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- simple
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- hard
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |-- user_query
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- simple
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- hard
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- bert_query_by_question_answer_concat.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer.json
          |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |-- reranked_query_by_question_answer_concat.json
          
    
      * to generate evaluation metrics: NDCG@3, NDCG@5, NDCG@10, P@3, P@5, P@10, MAP
        copy & override "rank_results" directory from Google Drive/BERT-FAQ to BERT-FAQ (local machine)
            e.g. BERT-FAQ/data/CovidFAQ/rank_results
            e.g. BERT-FAQ/data/StackFAQ/rank_results
            e.g. BERT-FAQ/data/FAQIR/rank_results
        
        run notebook (CovidFAQ, StackFAQ, FAQIR)
            08.Evaluation.ipynb
        
        it generates results.csv file:
          
          |  |-- data
          |  |  |
          |  |  |-- CovidFAQ / StackFAQ / FAQIR      
          |  |  |  |  |-- rank_results
          |  |  |  |  |  |  |  |-- results.csv
          
```

## FAQ BERT Ranker
```python

from elasticsearch_dsl.connections import connections
from faq_bert_ranker import FAQ_BERT_Ranker

try:
    es = connections.create_connection(hosts=['localhost'])
except TransportError as e:
    e.info()

top_k = 100
dataset = 'CovidFAQ'
fields = ['question_answer']

# Define model parameters
loss_type = 'triplet'; neg_type = 'hard'; query_type = 'user_query'; version = '1.1'

model_name = "{}_{}_{}_{}".format(loss_type, neg_type, query_type, version)
bert_model_path = bert_model_path = "output" + "/" + dataset + "/models/" + model_name

faq_bert_ranker = FAQ_BERT_Ranker(
  es=es, index=index, fields=fields, top_k=top_k, bert_model_path=bert_model_path
)

ranked_results = faq_bert_ranker.rank_results("Where did COVID-19 come from?")
ranked_result = ranked_results[0]
print(ranked_result)

'''
    [
       {
        'answer': 'It was first found in Wuhan City, Hubei Province, China. The '
                  'first cases are linked to a live animal market but now COVID-19 '
                  'is able to spread person-to-person.',
        'bert_score': 0.6462,
        'es_score': 1.0,
        'question': 'Where did COVID-19 come from?',
        'score': 1.6462
       }
    ]
 '''
```

## Citations
```
@inproceedings{karan2015faqir,
      title={FAQIR -- a Frequently Asked Questions Retrieval Test Collection},
      author={Karan, Mladen and {\v{S}}najder, Jan},
      booktitle={Proceedings of the 10th edition of the Language Resources and Evaluation Conference, LREC 2016},
      year={2015},
      organization={ELRA}
  }
  
 @article{karan2018paraphrase,
      title={Paraphrase-focused learning to rank for domain-specific frequently asked questions retrieval},
      author={Karan, Mladen and {\v{S}}najder, Jan},
      journal={Expert Systems with Applications},
      volume={91},
      pages={418--433},
      year={2018},
      publisher={Elsevier}
}

@inproceedings{Collecting+COVID_NLP20202,
      title={Collecting Verified COVID-19 Question Answer Pairs},
      author={Poliak, Adam and Fleming, Max and Costello, Cash and Murray, Kenton W and Yarmohammadi, Mahsa and Pandya, Shivani and Irani, Darius and Agarwal,
      Milind and Sharma, Udit and Sun, Shuo and Ivanov, Nicola and Shang, Lingxi and Srinivasan, Kaushik and Lee, Seolhwa and Han, Xu and Agarwal, Smisha 
      and Sedoc, João},
      year={2020},
      booktitle={NLP COVID-19 Workshop @EMNLP},
      url={https://openreview.net/forum?id=GR03UfD2OZk}
}
```
