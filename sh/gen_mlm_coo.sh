CUDA_VISIBLE_DEVICES=0 python src/bert_cooccur.py \
--corpus_path /data/ganleilei/BertGloVe/wiki/ \
--model_name roberta-large \
--bert_path /data/ganleilei/bert/ \
--save_path /data/ganleilei/BertGloVe/wiki/ \
--divide \
--window_size 10 \
--text2bin \
--word_pair_path /data/ganleilei/BertGloVe/wiki/roberta-large/mlm/cooccur/window10/word_coo/word.filter0.5.merge.coo.txt \
# --word_pair_path /data/ganleilei/BertGloVe/wiki/glove/cooccurrence.wiki.word.dim300 \
# --word_bpe_pair_path data/vocab/wiki.roberta.word.bpe.txt \
# --mlm_glove \