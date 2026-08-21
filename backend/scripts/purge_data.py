import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.database import clear_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("purge_data")

def purge():
    logger.info("Purging requests logging and metrics history from database...")
    try:
        clear_db()
        logger.info("Database metrics and query logs successfully purged.")
    except Exception as e:
        logger.error(f"Failed to purge database: {e}")

if __name__ == "__main__":
    purge()
