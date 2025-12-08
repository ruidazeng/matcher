import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

HEADER_LEN = 17

# Part type directories
BEAM_DIR = os.path.join(DATA_DIR, "BEAM")  # beam structure measurements
BOX_DIR = os.path.join(DATA_DIR, "BOX")  # box container measurements
BRK_DIR = os.path.join(DATA_DIR, "BRK")  # bracket measurements
CONLID_DIR = os.path.join(DATA_DIR, "CONLID")  # container with glued lid measurements
CONTAINER_DIR = os.path.join(DATA_DIR, "CONTAINER")  # container only measurements
FLG_DIR = os.path.join(DATA_DIR, "FLG")  # flange measurements
IMP_DIR = os.path.join(DATA_DIR, "IMP")  # impeller measurements
LID_DIR = os.path.join(DATA_DIR, "LID")  # lid only measurements
SEN_DIR = os.path.join(DATA_DIR, "SEN")  # sensor only measurements
TUBE_DIR = os.path.join(DATA_DIR, "TUBE")  # plastic tube measurements
VNT_DIR = os.path.join(DATA_DIR, "VNT")  # vent measurements

# If you want to add data for a new part type, it has to go here
ALL_PART_TYPES = [
    "BEAM",       # Beam structure measurements
    "BOX",        # Box container measurements
    "BRK",        # Bracket measurements
    "CONLID",     # Container with glued lid measurements
    "CONTAINER",  # Container only measurements
    "FLG",        # Flange measurements
    "IMP",        # Impeller measurements
    "LID",        # Lid only measurements
    "SEN",        # Sensor only measurements
    "TUBE",       # Plastic tube measurements
    "VNT",        # Vent measurements
]