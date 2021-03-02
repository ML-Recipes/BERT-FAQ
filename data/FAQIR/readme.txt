FAQIR -- a test collection for building and evaluating Frequently Asked Questions (FAQ) retrieval models.

Provided by:
Text Analysis and Knowledge Engineering Lab, FER, University of Zagreb
http://takelab.fer.hr

Version: 1.0
Release date: November 2, 2015

1. Description

FAQ retrieval is an interesting area at the intersection of question answering,
semantic search and information retrieval. We provide here a data set for the
FAQ retrieval task in English language.
The construction of the dataset is described in:

   Mladen Karan and Jan Šnajder (2015). FAQIR -- a Frequently Asked Questions Retrieval Test Collection. Proceedings of the 10th edition 
   of the Language Resources and Evaluation Conference (LREC2016). In press.

If you use this dataset for your own work, please cite the above paper.

The BibTeX citation is:

  @inproceedings{karan2015faqir,
    title={FAQIR -- a Frequently Asked Questions Retrieval Test Collection},
    author={Karan, Mladen and {\v{S}}najder, Jan},
    booktitle={Proceedings of the 10th edition of the Language Resources and Evaluation Conference, LREC 2016},
    year={2015},
    organization={ELRA}
  }

2. Data set

The data set contains 4133 FAQ-pairs and 1233 queries. It is organised
into an xml file as follows:

Beneath the upper most "IRSet" tag there are three tags grouping the 
most important parts of the data set, namely the queries, documents and
relevance judgements:

1. <queries>
   Contains a list of queries in <Query> tags containing the following information:
   <Author> - the author of the query
   <infneed> - the query template on which the query is based
   <Original> - the original query from which the query template was devised
   <exWords> - list of important words in the query (these were not used 
               for the paper) 
   <QueryGroupID> - id for the query template
   <QueryID> - id for the query
   <QueryString> - text of the query 
   + some other tags which can be ignored *

2. <Pairs>
   Contains a list of FAQ-pairs in <qapair> tags containing the following information:
   <Id> - id of the FAQ pair
   <Question> - question text of the FAQ-pair
   <Answer> - answer text of the FAQ-pair
   <Categories> - list of categories into which the FAQ-pair was submitted
   + some other tags which can be ignored *
   
3. <relCandidates> - a list of relevance judgements
   This tag contains for each of the 50 query templates an <IRCandidateList> tag
   which represents an annotated list of FAQ-pair candidates that were pooled for
   that query template. An <IRCandidateList> tag contains.

   <grpId> - query template id (corresponding to <QueryGroupID> above)
   several <IRCandidate> tags (one for each pooled document) containing
   a FAQ-pair id (<Id> tag) and a list of annotations for that document
   (<Annotations> tag). Individual Annotations further contain data on the 
   time (<AnnotationTime> tag), annotator (<Annotator> tag) and value
   (<Val> tag) of the annotation. The value coding is:
    1 - relevant 
    2 - useful
    3 - useless 
    4 - irrelevant

Each query has at least one FAQ-pair annotated as "relevant".  However, it is
possible for a FAQ-pair to be irrelevant for all queries. 
 
* extra tags are there because the xml file was generate through C# object serialization
  There should be no problem reading it in any environment. Moreover, you can mail the 
  authors to obtain a C# class that will load the file and provide an interface to the data.   


3. License 

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.


