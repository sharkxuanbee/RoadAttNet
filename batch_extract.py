import os
import glob
import logging
from tqdm import tqdm
from config import Config, load_config
from Feature_Extration import feature_extraction

def setup_logger():
    # Set up a logger that writes to both file and console
    log_file = "batch_extraction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logger()
    logger.info("Initializing Feature Extraction pipeline...")
    
    # Load configuration
    cfg = Config()
    
    rgb_dir = cfg.rgb_dir
    feature1_dir = cfg.feature1_dir # blurred
    feature2_dir = cfg.feature2_dir # cwl
    
    logger.info("Loaded config paths:")
    logger.info(f"  RGB Source: {rgb_dir}")
    logger.info(f"  Blurred Output: {feature1_dir}")
    logger.info(f"  CWL Output: {feature2_dir}")

    # Create directories if they do not exist
    os.makedirs(feature1_dir, exist_ok=True)
    os.makedirs(feature2_dir, exist_ok=True)
    
    # Get all TIFF images from the source directory
    tiff_files = glob.glob(os.path.join(rgb_dir, "*.tiff")) + glob.glob(os.path.join(rgb_dir, "*.tif"))
    
    if not tiff_files:
        logger.error(f"No TIFF images found in {rgb_dir}. Please check your configuration.")
        return

    logger.info(f"Found {len(tiff_files)} images. Starting feature extraction...")
    
    # Process each image with a progress bar
    success_count = 0
    error_count = 0
    for img_path in tqdm(tiff_files, desc="Extracting Features"):
        try:
            # We updated feature_extraction to take two specific output paths for cwl and blurred respectively
            feature_extraction(img_path, feature2_dir, feature1_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            error_count += 1
            
    logger.info(f"Extraction completed! Successfully processed: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()
