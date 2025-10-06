from tinygrad import Tensor, nn, Device, GlobalCounters
from extra.models.llama import Transformer, convert_from_huggingface
from tinygrad.helpers import Timing, getenv
from transformers import auto
