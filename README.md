# UnipaGPT-23

Unipa-GPT: Large Language Models for university-oriented QA in Italian

Please cite our work as follows if you use Unipa-corpus

```
@misc{siragusa2024unipagptlargelanguagemodels,
      title={Unipa-GPT: Large Language Models for university-oriented QA in Italian}, 
      author={Irene Siragusa and Roberto Pirrone},
      year={2024},
      eprint={2407.14246},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.14246}, 
}
```

In **corpora** folder we report:
  * _unipa-corpus-clear-docs_ documents of clear split of unipa-corpus
  * _unipa-corpus-emb-docs_ documents of emb split of unipa-corpus
  * _unipa-corpus-full-docs_ documents of full split of unipa-corpus
  * _unipa-corpus-finetuning_ QA pairs generated from the clear split of unipa-corpus used for fine-tuning purposes divided in train and validation split. 
  * _embeddings_ a FAISS vectordatabase created with all document belonging to unipa-corpus-clear-docs
  * _qa_golden.json_ contains 6 manually generated QA pairs
  * _qa_retrieved.json_ contains 4 automatically retrieved documents for each of 6 manually generated QA pairs


In **code** folder we report:
  *  _ingest_data.py_ to generate vectordatabase starting from documents' folder
  *  _query_model.py_ to infer LLM
  *  _eval.py_ to evaluate generated answers
