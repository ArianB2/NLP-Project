import os
import json
import base64
import asyncio
import numpy as np
from collections import deque

import sounddevice as sd
from dotenv import load_dotenv
from websockets.asyncio.client import connect


SAMPLE_RATE = 24_000
FRAME_MS = 20
