CUDA_VISIBLE_DEVICES=0 python src/bert_cooccur.py --corpus_name xab \
--corpus_path /data/ganleilei/BertGloVe/wiki/ \
--model_name roberta-large \
--bert_path /data/ganleilei/bert/ \
--save_path /data/ganleilei/BertGloVe/wiki/ \
--divide \
--window_size 25 \
--mlm_glove \