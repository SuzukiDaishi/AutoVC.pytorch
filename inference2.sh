python inference2.py --output ./output_p226_p226.wav \
                     --src-wav ./data/test/p226/p226_336/0000.wav \
                     --src-emb ./data/train/p226/emb.npy \
                     --tgt-emb ./data/train/p226/emb.npy \
                     --src-world ./data/train/p226/world.npz \
                     --tgt-world ./data/train/p226/world.npz \
                     --vocoder ./checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth


python inference2.py --output ./output_p226_p230.wav \
                     --src-wav ./data/test/p226/p226_336/0000.wav \
                     --src-emb ./data/train/p226/emb.npy \
                     --tgt-emb ./data/train/p230/emb.npy \
                     --src-world ./data/train/p226/world.npz \
                     --tgt-world ./data/train/p230/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth


python inference2.py --output ./output_p226_p231.wav \
                     --src-wav ./data/test/p226/p226_336/0000.wav \
                     --src-emb ./data/train/p226/emb.npy \
                     --tgt-emb ./data/train/p231/emb.npy \
                     --src-world ./data/train/p226/world.npz \
                     --tgt-world ./data/train/p231/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth


python inference2.py --output ./output_p226_p232.wav \
                     --src-wav ./data/test/p226/p226_336/0000.wav \
                     --src-emb ./data/train/p226/emb.npy \
                     --tgt-emb ./data/train/p232/emb.npy \
                     --src-world ./data/train/p226/world.npz \
                     --tgt-world ./data/train/p232/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

####################################################################

python inference2.py --output ./output_p230_p226.wav \
                     --src-wav ./data/test/p230/p230_377/0000.wav \
                     --src-emb ./data/train/p230/emb.npy \
                     --tgt-emb ./data/train/p226/emb.npy \
                     --src-world ./data/train/p230/world.npz \
                     --tgt-world ./data/train/p226/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

python inference2.py --output ./output_p230_p230.wav \
                     --src-wav ./data/test/p230/p230_377/0000.wav \
                     --src-emb ./data/train/p230/emb.npy \
                     --tgt-emb ./data/train/p230/emb.npy \
                     --src-world ./data/train/p230/world.npz \
                     --tgt-world ./data/train/p230/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth


python inference2.py --output ./output_p230_p231.wav \
                     --src-wav ./data/test/p230/p230_377/0000.wav \
                     --src-emb ./data/train/p230/emb.npy \
                     --tgt-emb ./data/train/p231/emb.npy \
                     --src-world ./data/train/p230/world.npz \
                     --tgt-world ./data/train/p231/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth


python inference2.py --output ./output_p230_p232.wav \
                     --src-wav ./data/test/p230/p230_377/0000.wav \
                     --src-emb ./data/train/p230/emb.npy \
                     --tgt-emb ./data/train/p232/emb.npy \
                     --src-world ./data/train/p230/world.npz \
                     --tgt-world ./data/train/p232/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

####################################################################

python inference2.py --output ./output_p231_p226.wav \
                     --src-wav ./data/test/p231/p231_431/0000.wav \
                     --src-emb ./data/train/p231/emb.npy \
                     --tgt-emb ./data/train/p226/emb.npy \
                     --src-world ./data/train/p231/world.npz \
                     --tgt-world ./data/train/p226/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

python inference2.py --output ./output_p231_p230.wav \
                     --src-wav ./data/test/p231/p231_431/0000.wav \
                     --src-emb ./data/train/p231/emb.npy \
                     --tgt-emb ./data/train/p230/emb.npy \
                     --src-world ./data/train/p231/world.npz \
                     --tgt-world ./data/train/p230/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

python inference2.py --output ./output_p231_p231.wav \
                     --src-wav ./data/test/p231/p231_431/0000.wav \
                     --src-emb ./data/train/p231/emb.npy \
                     --tgt-emb ./data/train/p231/emb.npy \
                     --src-world ./data/train/p231/world.npz \
                     --tgt-world ./data/train/p231/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

python inference2.py --output ./output_p231_p232.wav \
                     --src-wav ./data/test/p231/p231_431/0000.wav \
                     --src-emb ./data/train/p231/emb.npy \
                     --tgt-emb ./data/train/p232/emb.npy \
                     --src-world ./data/train/p231/world.npz \
                     --tgt-world ./data/train/p232/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

####################################################################

python inference2.py --output ./output_p232_p226.wav \
                     --src-wav ./data/test/p232/p232_375/0000.wav \
                     --src-emb ./data/train/p232/emb.npy \
                     --tgt-emb ./data/train/p226/emb.npy \
                     --src-world ./data/train/p232/world.npz \
                     --tgt-world ./data/train/p226/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

python inference2.py --output ./output_p232_p230.wav \
                     --src-wav ./data/test/p232/p232_375/0000.wav \
                     --src-emb ./data/train/p232/emb.npy \
                     --tgt-emb ./data/train/p230/emb.npy \
                     --src-world ./data/train/p232/world.npz \
                     --tgt-world ./data/train/p230/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

python inference2.py --output ./output_p232_p231.wav \
                     --src-wav ./data/test/p232/p232_375/0000.wav \
                     --src-emb ./data/train/p232/emb.npy \
                     --tgt-emb ./data/train/p231/emb.npy \
                     --src-world ./data/train/p232/world.npz \
                     --tgt-world ./data/train/p231/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

python inference2.py --output ./output_p232_p232.wav \
                     --src-wav ./data/test/p232/p232_375/0000.wav \
                     --src-emb ./data/train/p232/emb.npy \
                     --tgt-emb ./data/train/p232/emb.npy \
                     --src-world ./data/train/p232/world.npz \
                     --tgt-world ./data/train/p232/world.npz \
                     --vocoder checkpoint_step001000000_ema.pth \
                     --autovc ./checkpoints/checkpoint_stargan_step000600.pth

####################################################################