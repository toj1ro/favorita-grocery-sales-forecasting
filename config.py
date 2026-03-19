import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
STORES_FILE = os.path.join(DATA_DIR, "stores.csv")
ITEMS_FILE = os.path.join(DATA_DIR, "items.csv")
OIL_FILE = os.path.join(DATA_DIR, "oil.csv")
HOLIDAYS_FILE = os.path.join(DATA_DIR, "holidays_events.csv")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")

N_SERIES = 150
RANDOM_SEED = 42

TRAIN_END = "2017-07-15"
VAL_END = "2017-07-31"

HORIZON = 16
SEASON_LENGTH = 7

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

MAX_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
