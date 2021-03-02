This archive contains an XML file representing two FAQ retrieval dataset.
 
The file represents threads on stackexchange. Each thread is within a "qaPair" tag, for each thread three things
are given in the file:

First, a list of rephrasements of the information need from the original StackExchange question ("rephr" tags)
Second, the original StackExchange question ("question" tag)
Third, top answers to the questions ("answer" tags)

From this data it is straigthforward to generate a FAQ retrieval data set, as described in the below paper.

If you use this data-set please cite:

@article{karan2018paraphrase,
  title={Paraphrase-focused learning to rank for domain-specific frequently asked questions retrieval},
  author={Karan, Mladen and {\v{S}}najder, Jan},
  journal={Expert Systems with Applications},
  volume={91},
  pages={418--433},
  year={2018},
  publisher={Elsevier}
}

In case you would like to get the code that generates the XML file by crawling StackExchange you
can send a mail to info (at) takelab (dot) fer (dot) hr.

