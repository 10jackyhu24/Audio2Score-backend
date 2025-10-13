# config.py
DATASET_PATH = "./maestro-v3.0.0"
SAMPLE_RATE = 16000
N_MELS = 229
HOP_LENGTH = 512
CLIP_SECONDS = 10  # 每段固定 10 秒

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

CHECKPOINT_DIR = "./output/checkpoints"
RESULT_DIR = "./output/results"
