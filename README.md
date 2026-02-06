# IS469 GenAI with LLMs

NLP vectorization and fine-tuning workflows for the Origami GPU Cluster.

## Structure

```
notebooks/          NLP embedding notebooks
  tfidf_bow_corpus  TF-IDF and Bag-of-Words basics
  word2vec          Word2Vec training and exploration
  bert              BERT contextual embeddings and sentence similarity

scripts/            GPU job scripts
  test_finetune.py  DistilBERT sentiment fine-tuning (SST-2)
  run_finetune.sh   Sbatch template for GPU job submission

docs/               Origami cluster documentation
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install scipy numpy gensim transformers torch sentence-transformers scikit-learn matplotlib
jupyter lab
```

## GPU Cluster Usage

1. SSH into the cluster: `ssh <username>@origami.smu.edu.sg`
2. Check your allocation: `myinfo`
3. Edit `scripts/run_finetune.sh` with your partition, account, and QOS
4. Submit a job:
   ```bash
   cd scripts
   chmod +x run_finetune.sh
   sbatch run_finetune.sh
   ```
5. Monitor: `myqueue` / `cat bert-finetune-test.<JOBID>.out`

See [docs/](docs/README.md) for full cluster documentation.

## Notes

- `word2vec.ipynb` downloads the 20-newsgroups dataset at runtime
- `bert.ipynb` downloads BERT models from Hugging Face on first run (~400MB)
- `test_finetune.py` fine-tunes DistilBERT on 2000 samples — runs in ~11s on a 2080 Ti
