docker run -it -v $(pwd):/root -w /root -p 8008:8008 --rm python_autovc python inference2.py \
--output ./output.wav \
--src-wav ./VCTK-Corpus/wav48/p226/p226_001.wav \
--src-emb ./data/train/p226/emb.npy \
--tgt-emb ./data/train/p230/emb.npy \
--src-world ./data/train/p226/world.npz \
--tgt-world ./data/train/p230/world.npz \
--vocoder ./checkpoint_step001000000_ema.pth \
--autovc ./checkpoints/checkpoint_mceps_step000260.pth