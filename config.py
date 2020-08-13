import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(name)s')
logger = logging.getLogger()

MODEL_PATH = 'resources/attention_3d_block_model_epoch_03_vl_0.2241.h5'
